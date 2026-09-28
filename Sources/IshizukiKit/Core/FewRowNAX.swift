// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// A 2-bit affine product for two to eight rows on the Metal 4 tensor ops (MSL 4.0, `matmul2d`),
// for the verify of a drafted token. The GPU's plain lanes only decode the weights into
// threadgroup memory; the multiply goes to the tensor units, so a second row costs no lane work.
// MLX's own tensor-op qmm is built for tall activations and, at a handful of rows, took about four
// times a single-row qmv on an M6 (docs/qwen38-27b/16 §14-3).

import Foundation
import MLX
import MLXFast

public enum FewRowNAX {
  public static let supportedRows = 2...8

  /// Off with `ISHIZUKI_FEW_ROW_NAX=0`.
  nonisolated(unsafe) public static var enabled =
    ProcessInfo.processInfo.environment["ISHIZUKI_FEW_ROW_NAX"] != "0"

  /// Weight rows per threadgroup and the K step, overridable for tuning.
  nonisolated(unsafe) public static var rowsPerGroup =
    ProcessInfo.processInfo.environment["ISHIZUKI_NAX_ROWS"].flatMap { Int($0) } ?? 64

  public static func apply(
    _ x: MLXArray, _ w: MLXArray, scales: MLXArray, biases: MLXArray, groupSize: Int, bits: Int
  ) -> MLXArray? {
    #if canImport(Metal)
      guard enabled, bits == 2, groupSize == 128, x.ndim == 2, supportedRows.contains(x.dim(0)),
        x.dtype == .float16, scales.dtype == .float16
      else { return nil }
      let m = x.dim(0)
      let k = x.dim(1)
      let n = w.dim(0)
      let nr0 = rowsPerGroup
      let tokens = ProcessInfo.processInfo.environment["ISHIZUKI_NAX_TOKENS"].flatMap { Int($0) } ?? 8
      let swap = ProcessInfo.processInfo.environment["ISHIZUKI_NAX_SWAP"] == "1"
      guard m <= tokens else { return nil }
      let mode = ProcessInfo.processInfo.environment["ISHIZUKI_NAX_MODE"].flatMap { Int($0) } ?? 0
      guard k % 128 == 0, n % nr0 == 0, w.dim(1) * 16 == k else { return nil }
      return kernel(
        [x, w, scales, biases, m],
        template: [("K", k), ("N", n), ("NR0", nr0), ("MODE", mode), ("NR1", tokens), ("SWAP", swap)],
        grid: (128 * (n / nr0), 1, 1),
        threadGroup: (128, 1, 1),
        outputShapes: [[m, n]],
        outputDTypes: [.float16])[0]
    #else
      return nil
    #endif
  }

  #if canImport(Metal)
    private static let kernel: MLXFast.MLXFastKernel = MLXFast.metalKernel(
      name: "ishizuki_few_row_nax",
      inputNames: ["x", "w", "scales", "biases", "M"],
      outputNames: ["y"],
      source: """
            // NR0 weight rows x 8 activation rows per threadgroup (4 simdgroups), K steps of one
            // group (128). Each step every thread decodes NR0 * 128 / 128 codes of one row into
            // threadgroup memory as half (scale * q + bias), the activations are staged beside
            // them, and the tensor op accumulates in float.
            constexpr int NK = 128, GS = 128;
            constexpr int KW = K / 16;                 // uint32 words per weight row
            constexpr int KG = K / GS;
            constexpr int TPR = 128 / NR0;             // threads per weight row
            constexpr int CPT = NK / TPR;              // codes per thread per step
            threadgroup half As[NR0 * NK];
            threadgroup half Bs[NR1 * NK];
            threadgroup float Cs[NR1 * NR0];

            uint tid = thread_position_in_threadgroup.x;
            uint row0 = threadgroup_position_in_grid.x * NR0;
            int rows = int(M);

            auto tA = tensor(As, dextents<int32_t, 2>(NK, NR0));
            auto tB = tensor(Bs, dextents<int32_t, 2>(NK, NR1));

            uint ar = tid / TPR, aq = tid % TPR;
            device const uint *wr = w + (row0 + ar) * KW + aq * (CPT / 16);
            device const half *sr = scales + (row0 + ar) * KG;
            device const half *br = biases + (row0 + ar) * KG;
            threadgroup half *dst = As + ar * NK + aq * CPT;

            // One K step: decode this thread's codes, stage the activations, then the product.
            #define FEW_ROW_STEP(RUN)                                                          \
            for (int k0 = 0; k0 < K; k0 += NK) {                                               \
                int g = k0 / GS;                                                               \
                float s = float(sr[g]);                                                        \
                float b = float(br[g]);                                                        \
                if (MODE != 2) for (int j = 0; j < CPT / 16; ++j) {                            \
                    uint word = wr[k0 / 16 + j];                                               \
                    for (int i = 0; i < 16; i += 2) {                                          \
                        half2 v = half2(float2(float((word >> (2 * i)) & 3u),                  \
                                               float((word >> (2 * i + 2)) & 3u)) * s + b);    \
                        *(threadgroup half2 *)(dst + j * 16 + i) = v;                          \
                    }                                                                          \
                }                                                                              \
                for (uint e = tid; e < uint(NR1 * NK); e += 128) {                             \
                    uint t = e / NK, kk = e % NK;                                              \
                    Bs[e] = int(t) < rows ? x[t * K + k0 + kk] : half(0);                      \
                }                                                                              \
                threadgroup_barrier(mem_flags::mem_threadgroup);                               \
                auto mB = tB.slice(0, 0);                                                      \
                auto mA = tA.slice(0, 0);                                                      \
                if (MODE != 1) RUN;                                                            \
                threadgroup_barrier(mem_flags::mem_threadgroup);                               \
            }

            if constexpr (SWAP) {
                // C[rows x tokens] = A[rows x K] * B[tokens x K]^T
                matmul2d<matmul2d_descriptor(NR0, NR1, NK, false, true, false,
                                             matmul2d_descriptor::mode::multiply_accumulate),
                         execution_simdgroups<4>> mm;
                auto cT = mm.template get_destination_cooperative_tensor<decltype(tA), decltype(tB), float>();
                for (uint16_t i = 0; i < cT.get_capacity(); ++i) { if (cT.is_valid_element(i)) cT[i] = 0.0f; }
                FEW_ROW_STEP(mm.run(mA, mB, cT))
                auto tC = tensor(Cs, dextents<int32_t, 2>(NR1, NR0));
                auto mC = tC.slice(0, 0);
                cT.store(mC);
                threadgroup_barrier(mem_flags::mem_threadgroup);
                for (uint e = tid; e < uint(rows * NR0); e += 128) {
                    uint t = e / NR0, r = e % NR0;
                    y[t * N + row0 + r] = half(Cs[r * NR1 + t]);
                }
            } else {
                // C[tokens x rows] = B[tokens x K] * A[rows x K]^T
                matmul2d<matmul2d_descriptor(NR1, NR0, NK, false, true, false,
                                             matmul2d_descriptor::mode::multiply_accumulate),
                         execution_simdgroups<4>> mm;
                auto cT = mm.template get_destination_cooperative_tensor<decltype(tB), decltype(tA), float>();
                for (uint16_t i = 0; i < cT.get_capacity(); ++i) { if (cT.is_valid_element(i)) cT[i] = 0.0f; }
                FEW_ROW_STEP(mm.run(mB, mA, cT))
                auto tC = tensor(Cs, dextents<int32_t, 2>(NR0, NR1));
                auto mC = tC.slice(0, 0);
                cT.store(mC);
                threadgroup_barrier(mem_flags::mem_threadgroup);
                for (uint e = tid; e < uint(rows * NR0); e += 128) {
                    uint t = e / NR0, r = e % NR0;
                    y[t * N + row0 + r] = half(Cs[t * NR0 + r]);
                }
            }
        """,
      header: """
        #include <metal_tensor>
        #include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
        using namespace mpp::tensor_ops;

        """)
  #endif
}
