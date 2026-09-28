// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// A 2-bit matrix-vector product for two to four rows at once: MLX's own qmv_fast, with every
// thread holding each row's slice of the activation, so a weight byte is read and decoded once
// and used for all of them. MLX runs a few-row product as that many separate passes over the
// weights, and VerifyMatmul spends a simdgroup matrix of eight columns on two; at the one-draft
// verify of an MTP head, both cost 1.4 to 2 times a single row on an M6 (docs/qwen38-27b/16 §14).

import Foundation
import MLX
import MLXFast

public enum FewRowQMV {
  public static let supportedRows = 2...4

  /// Off with `ISHIZUKI_FEW_ROW_QMV=0`, for comparing against the paths it replaces.
  nonisolated(unsafe) public static var enabled =
    ProcessInfo.processInfo.environment["ISHIZUKI_FEW_ROW_QMV"] != "0"

  /// Nil when the shape is not one this kernel covers: 2-bit affine, a group of 32 to 512 that
  /// divides the 512-wide block, whole blocks of input and whole blocks of eight outputs.
  public static func apply(
    _ x: MLXArray, _ w: MLXArray, scales: MLXArray, biases: MLXArray, groupSize: Int, bits: Int
  ) -> MLXArray? {
    #if canImport(Metal)
      guard enabled, bits == 2, x.ndim == 2, (1...4).contains(x.dim(0)) else { return nil }
      let m = x.dim(0)
      let k = x.dim(1)
      let n = w.dim(0)
      guard k % 512 == 0, n % 8 == 0, [32, 64, 128, 256, 512].contains(groupSize),
        w.dim(1) * 16 == k, scales.dim(1) * groupSize == k
      else { return nil }
      return kernel(
        [x, w, scales, biases],
        template: [("XT", x.dtype), ("R", m), ("K", k), ("N", n), ("GS", groupSize)],
        grid: (64 * (n / 8), 1, 1),
        threadGroup: (64, 1, 1),
        outputShapes: [[m, n]],
        outputDTypes: [x.dtype])[0]
    #else
      return nil
    #endif
  }

  /// The same product with the 2-bit codes turned into numbers two at a time: one AND and one OR
  /// on a 32-bit word give a half2 of 1024 + code (times a power of four), which the multiply
  /// reads as a float, so a weight costs one integer op and one FMA per row instead of MLX's
  /// mask, convert and FMA. The 1024 comes back out once per block through the activation's sum.
  /// Rows 1 to 4. Off with `ISHIZUKI_QMV_MAGIC=0`.
  nonisolated(unsafe) public static var magicEnabled =
    ProcessInfo.processInfo.environment["ISHIZUKI_QMV_MAGIC"] != "0"
  public static let magicRows = 1...4
  /// Variants for finding where the two-row time goes (16 §16), `ISHIZUKI_QMV_MODE`: 0 the kernel,
  /// 1 the second row's FMAs left out, 2 the second row's activation preparation left out,
  /// 4 only the loads (a floor), 5 both rows in one half2 multiply per code.
  nonisolated(unsafe) public static var magicMode =
    ProcessInfo.processInfo.environment["ISHIZUKI_QMV_MODE"].flatMap { Int($0) } ?? 0
  nonisolated(unsafe) public static var magicRolled =
    ProcessInfo.processInfo.environment["ISHIZUKI_QMV_ROLLED"] == "1"
  /// One-row products through the magic kernel too (`ISHIZUKI_QMV_MAGIC1=1`), for comparing it
  /// with MLX's qmv on the real forward. Off: one row stays on MLX.
  nonisolated(unsafe) public static var magicSingleRow =
    ProcessInfo.processInfo.environment["ISHIZUKI_QMV_MAGIC1"] == "1"
  /// 32-bit words of codes each thread reads per pass: 1 or 2 (`ISHIZUKI_QMV_WPT`).
  nonisolated(unsafe) public static var magicWordsPerThread =
    ProcessInfo.processInfo.environment["ISHIZUKI_QMV_WPT"].flatMap { Int($0) } ?? 1
  nonisolated(unsafe) public static var magicGridY =
    ProcessInfo.processInfo.environment["ISHIZUKI_QMV_GRIDY"] == "1"
  /// Output rows per simdgroup: the activation's per-block preparation is shared by this many.
  nonisolated(unsafe) public static var magicRowsPerSimdgroup =
    ProcessInfo.processInfo.environment["ISHIZUKI_QMV_RPS"].flatMap { Int($0) } ?? 4

  public static func applyMagic(
    _ x: MLXArray, _ w: MLXArray, scales: MLXArray, biases: MLXArray, groupSize: Int, bits: Int
  ) -> MLXArray? {
    #if canImport(Metal)
      guard magicEnabled, bits == 2, x.ndim == 2, magicRows.contains(x.dim(0)) else { return nil }
      let m = x.dim(0)
      let k = x.dim(1)
      let n = w.dim(0)
      guard k % 512 == 0, n % 8 == 0, [32, 64, 128, 256, 512].contains(groupSize),
        w.dim(1) * 16 == k, scales.dim(1) * groupSize == k
      else { return nil }
      let rps = magicRowsPerSimdgroup
      let wpt = k % 1024 == 0 && groupSize >= 32 * magicWordsPerThread ? magicWordsPerThread : 1
      guard n % (2 * rps) == 0 else { return nil }
      let rolled = magicRolled
      let gridY = magicGridY
      return (rolled ? magicKernelRolled : magicKernel)(
        [x, w, scales, biases],
        template: [
          ("XT", x.dtype), ("R", m), ("K", k), ("N", n), ("GS", groupSize), ("RPS", rps),
          ("MODE", magicMode), ("GY", gridY ? 1 : 0), ("WPT", wpt),
        ],
        grid: gridY ? (32 * m, 2 * (n / (2 * rps)), 1) : (64 * (n / (2 * rps)), 1, 1),
        threadGroup: gridY ? (32, 2, 1) : (64, 1, 1),
        outputShapes: [[m, n]],
        outputDTypes: [x.dtype])[0]
    #else
      return nil
    #endif
  }

  #if canImport(Metal)
    private static let magicSource: String = """
            // WPT 32-bit words (16 codes each) per thread per pass, 32 * 16 * WPT codes per
            // simdgroup pass, RPS output rows per simdgroup, 2 simdgroups per threadgroup, as
            // qmv_fast. Word bits 2i..2i+1 hold code i. The half2 lanes are bits 0-15 (codes
            // 0-7) and 16-31 (codes 8-15); a code at lane bits 2j (j <= 4) ORed with 0x6400 is
            // the half 1024 + 4^j * code, exact. Codes 5-7 and 13-15 come from the word >> 10.
            constexpr int VPT = 16 * WPT;
            constexpr int BLOCK = VPT * 32;
            constexpr int KW = K / 16;           // words per weight row
            constexpr int KG = K / GS;
            constexpr int STEP = GS / VPT;
            constexpr uint MAGIC = 0x64006400u;

            uint lane = thread_index_in_simdgroup;
            uint sg = simdgroup_index_in_threadgroup;
            int out_row = int(GY ? threadgroup_position_in_grid.y : threadgroup_position_in_grid.x) * (2 * RPS) + int(sg) * RPS;
            // GY: (32, 2, 1) threadgroups on a (rows of x, output blocks) grid, qmv_fast's own
            // launch shape. Only for comparing; the rows of x then do not share a weight read.

            device const uint *ws = (device const uint *)w + out_row * KW + lane * WPT;
            device const XT *sc = scales + out_row * KG + lane / STEP;
            device const XT *bi = biases + out_row * KG + lane / STEP;
            device const XT *xp = x + lane * VPT;

            // Position i of a word's 16 sits at 4^shift[i] in its half.
            const float inv[16] = {1.0f, 0.25f, 0.0625f, 0.015625f, 0.00390625f, 1.0f, 0.25f, 0.0625f,
                                   1.0f, 0.25f, 0.0625f, 0.015625f, 0.00390625f, 1.0f, 0.25f, 0.0625f};

            float xt[R][VPT];
            float result[R][RPS];
            for (int r = 0; r < R; ++r)
                for (int j = 0; j < RPS; ++j) result[r][j] = 0.0f;

            for (int k = 0; k < K; k += BLOCK) {
                uint words[RPS][WPT];
                float sc_[RPS], bi_[RPS];
                for (int row = 0; row < RPS; ++row) {
                    if (WPT == 2) {
                        uint2 pair = *((device const uint2 *)(ws + row * KW));
                        words[row][0] = pair.x;
                        words[row][WPT - 1] = pair.y;
                    } else {
                        words[row][0] = ws[row * KW];
                    }
                    sc_[row] = float(sc[row * KG]);
                    bi_[row] = float(bi[row * KG]);
                }
                float sum[R];
                float corr[R];
                for (int r = 0; r < R; ++r) {
                    device const vec<XT, 4> *x4 = (device const vec<XT, 4> *)(xp + r * K + k);
                    float s = 0.0f, c = 0.0f;
                    for (int q = 0; q < VPT / 4; ++q) {
                        float4 a = float4(x4[q]);
                        s += a.x + a.y + a.z + a.w;
                        for (int e = 0; e < 4; ++e) {
                            xt[r][4 * q + e] = a[e] * inv[(4 * q + e) % 16];
                            c += xt[r][4 * q + e];
                        }
                    }
                    sum[r] = s;
                    corr[r] = 1024.0f * c;
                }
                for (int row = 0; row < RPS; ++row) {
                    if (MODE == 4) {
                        for (int r = 0; r < R; ++r)
                            result[r][row] += float(words[row][0] & 0xffu) * sc_[row] + bi_[row] + float(words[row][WPT - 1] >> 24);
                        continue;
                    }
                    float acc[R];
                    for (int r = 0; r < R; ++r) acc[r] = 0.0f;
                    for (int p = 0; p < WPT; ++p) {
                        uint u = words[row][p];
                        uint v = u >> 10;
                        half2 h0 = as_type<half2>((u & 0x00030003u) | MAGIC);
                        half2 h1 = as_type<half2>((u & 0x000c000cu) | MAGIC);
                        half2 h2 = as_type<half2>((u & 0x00300030u) | MAGIC);
                        half2 h3 = as_type<half2>((u & 0x00c000c0u) | MAGIC);
                        half2 h4 = as_type<half2>((u & 0x03000300u) | MAGIC);
                        half2 h5 = as_type<half2>((v & 0x00030003u) | MAGIC);
                        half2 h6 = as_type<half2>((v & 0x000c000cu) | MAGIC);
                        half2 h7 = as_type<half2>((v & 0x00300030u) | MAGIC);
                        for (int r = 0; r < R; ++r) {
                            thread const float *xq = xt[r] + 16 * p;
                            acc[r] += xq[0] * float(h0.x) + xq[8] * float(h0.y)
                                + xq[1] * float(h1.x) + xq[9] * float(h1.y)
                                + xq[2] * float(h2.x) + xq[10] * float(h2.y)
                                + xq[3] * float(h3.x) + xq[11] * float(h3.y)
                                + xq[4] * float(h4.x) + xq[12] * float(h4.y)
                                + xq[5] * float(h5.x) + xq[13] * float(h5.y)
                                + xq[6] * float(h6.x) + xq[14] * float(h6.y)
                                + xq[7] * float(h7.x) + xq[15] * float(h7.y);
                        }
                    }
                    for (int r = 0; r < R; ++r)
                        result[r][row] += sc_[row] * (acc[r] - corr[r]) + sum[r] * bi_[row];
                }
                ws += BLOCK / 16;
                sc += BLOCK / GS;
                bi += BLOCK / GS;
            }

            for (int r = 0; r < R; ++r) {
                for (int row = 0; row < RPS; ++row) {
                    float v = simd_sum(result[r][row]);
                    if (lane == 0) y[r * N + out_row + row] = XT(v);
                }
            }
        """

    private static let magicKernel: MLXFast.MLXFastKernel = MLXFast.metalKernel(
      name: "ishizuki_few_row_qmv2_magic",
      inputNames: ["x", "w", "scales", "biases"],
      outputNames: ["y"],
      source: magicSource)

    /// The k loop kept rolled (`ISHIZUKI_QMV_ROLLED=1`), for comparing against the unrolled one.
    private static let magicKernelRolled: MLXFast.MLXFastKernel = MLXFast.metalKernel(
      name: "ishizuki_few_row_qmv2_magic_rolled",
      inputNames: ["x", "w", "scales", "biases"],
      outputNames: ["y"],
      source: magicSource.replacingOccurrences(
        of: "for (int k = 0; k < K; k += BLOCK) {",
        with: "_Pragma(\"clang loop unroll(disable)\")\nfor (int k = 0; k < K; k += BLOCK) {"))
  #endif

  #if canImport(Metal)
    private static let kernel: MLXFast.MLXFastKernel = MLXFast.metalKernel(
      name: "ishizuki_few_row_qmv2",
      inputNames: ["x", "w", "scales", "biases"],
      outputNames: ["y"],
      source: """
            // qmv_fast for bits 2: 16 values (4 bytes) per thread, 512 per simdgroup pass,
            // 4 output rows per simdgroup, 2 simdgroups per threadgroup.
            constexpr int VPT = 16;
            constexpr int BLOCK = VPT * 32;
            constexpr int RPS = 4;
            constexpr int KW = K / 4;            // bytes per weight row
            constexpr int KG = K / GS;           // groups per weight row
            constexpr int STEP = GS / VPT;       // threads sharing a scale

            uint lane = thread_index_in_simdgroup;
            uint sg = simdgroup_index_in_threadgroup;
            int out_row = int(threadgroup_position_in_grid.x) * (2 * RPS) + int(sg) * RPS;

            device const uint8_t *ws = (device const uint8_t *)w + out_row * KW + lane * 4;
            device const XT *sc = scales + out_row * KG + lane / STEP;
            device const XT *bi = biases + out_row * KG + lane / STEP;
            device const XT *xp = x + lane * VPT;

            float xt[R][VPT];
            float result[R][RPS];
            for (int r = 0; r < R; ++r)
                for (int j = 0; j < RPS; ++j) result[r][j] = 0.0f;

            for (int k = 0; k < K; k += BLOCK) {
                float sum[R];
                for (int r = 0; r < R; ++r) {
                    device const XT *xr = xp + r * K + k;
                    float s = 0.0f;
                    for (int i = 0; i < VPT; i += 4) {
                        float a = float(xr[i]), b = float(xr[i + 1]);
                        float c = float(xr[i + 2]), d = float(xr[i + 3]);
                        s += a + b + c + d;
                        xt[r][i] = a;
                        xt[r][i + 1] = b / 4.0f;
                        xt[r][i + 2] = c / 16.0f;
                        xt[r][i + 3] = d / 64.0f;
                    }
                    sum[r] = s;
                }
                for (int row = 0; row < RPS; ++row) {
                    device const uint8_t *wl = ws + row * KW;
                    float s = float(sc[row * KG]);
                    float b = float(bi[row * KG]);
                    float acc[R];
                    for (int r = 0; r < R; ++r) acc[r] = 0.0f;
                    for (int i = 0; i < VPT / 4; ++i) {
                        uint8_t q = wl[i];
                        float q0 = float(q & 0x03), q1 = float(q & 0x0c);
                        float q2 = float(q & 0x30), q3 = float(q & 0xc0);
                        for (int r = 0; r < R; ++r) {
                            acc[r] += xt[r][4 * i] * q0 + xt[r][4 * i + 1] * q1
                                + xt[r][4 * i + 2] * q2 + xt[r][4 * i + 3] * q3;
                        }
                    }
                    for (int r = 0; r < R; ++r) result[r][row] += s * acc[r] + sum[r] * b;
                }
                ws += BLOCK / 4;
                sc += BLOCK / GS;
                bi += BLOCK / GS;
            }

            for (int r = 0; r < R; ++r) {
                for (int row = 0; row < RPS; ++row) {
                    float v = simd_sum(result[r][row]);
                    if (lane == 0) y[r * N + out_row + row] = XT(v);
                }
            }
        """)
  #endif
}
