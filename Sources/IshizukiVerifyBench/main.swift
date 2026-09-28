// SPDX-License-Identifier: AGPL-3.0-or-later
// Few-row matmul timing for the MTP verify on the M6 (docs/qwen38-27b/16 §14). No model is
// loaded: every weight is random, quantized the way the Bonsai 2 27B pack holds it (2-bit
// affine, group 128), at each shape its backbone multiplies once per token. The output head
// (248320x5120) is left out: it would need 5 GB of random floats beside the resident server,
// and the kernel's own table already takes it at two rows.
//
//   ishizuki-verify-bench [iterations]
//
// Per shape and row count M: MLX's quantizedMM, and ishizuki's VerifyMatmul with its row
// floor lowered to M (mtp.patch), and FewRowQMV (mtp.patch) for M 2 to 4. Each timing chains 16 products into one eval and divides,
// the median of `iterations` such evals. The kernel's output is checked against MLX's.

import Foundation
import IshizukiKit
import MLX
import MLXRandom

let iterations = CommandLine.arguments.count > 1 ? Int(CommandLine.arguments[1]) ?? 20 : 20
let chain = 16

// (label, output n, input k, count per token in the 64-layer backbone)
let shapes: [(String, Int, Int, Int)] = [
  ("mlp.gate/up      17408x5120", 17408, 5120, 128),
  ("mlp.down         5120x17408", 5120, 17408, 64),
  ("gdn.in_proj_qkv  10240x5120", 10240, 5120, 48),
  ("gdn.in_proj_z    6144x5120", 6144, 5120, 48),
  ("gdn/attn.out     5120x6144", 5120, 6144, 64),
  ("attn.q           12288x5120", 12288, 5120, 16),
  ("attn.k/v         1024x5120", 1024, 5120, 32),
]

func median(_ xs: [Double]) -> Double { xs.sorted()[xs.count / 2] }

func time(_ body: () -> [MLXArray]) -> Double {
  eval(body())
  var runs: [Double] = []
  for _ in 0..<iterations {
    let start = DispatchTime.now().uptimeNanoseconds
    eval(body())
    runs.append(Double(DispatchTime.now().uptimeNanoseconds - start) / 1e3 / Double(chain))
  }
  return median(runs)
}

MLXRandom.seed(7)
// The quantized-KV attention of one full-attention layer (16 §16), as Attention.quantizedAttention
// runs it on the house cache (keys 3-bit, values 4-bit, group 64, a dense window of 128), for one
// and two queries: 24 heads over 4 KV heads, head 256, VB_ATTN_T cached tokens. Chains 16 layers.
if ProcessInfo.processInfo.environment["VB_ATTN"] == "1" {
  let total = Int(ProcessInfo.processInfo.environment["VB_ATTN_T"] ?? "2316") ?? 2316
  let win = 128
  let t = total - win
  let (kvH, rep, d) = (4, 6, 256)
  let keysQ = quantized((MLXRandom.normal([1, kvH, t, d])).asType(.float16), groupSize: 64, bits: 3)
  let valsQ = quantized((MLXRandom.normal([1, kvH, t, d])).asType(.float16), groupSize: 64, bits: 4)
  let k3 = (keysQ.wq.expandedDimensions(axis: 2), keysQ.scales.expandedDimensions(axis: 2), keysQ.biases!.expandedDimensions(axis: 2))
  let v4 = (valsQ.wq.expandedDimensions(axis: 2), valsQ.scales.expandedDimensions(axis: 2), valsQ.biases!.expandedDimensions(axis: 2))
  let wk = MLXRandom.normal([1, kvH, 1, win, d]).asType(.float16)
  let wv = MLXRandom.normal([1, kvH, 1, win, d]).asType(.float16)
  eval(k3.0, k3.1, k3.2, v4.0, v4.1, v4.2, wk, wv)
  print("quantized attention, \(total) tokens (us per layer): l / full / scores qmm / values qmm / softmax+mask")
  for l in [1, 2] {
    let qs = (0..<chain).map { _ in (MLXRandom.normal([1, kvH, rep, l, d]) * 0.06).asType(.float16) }
    eval(qs)
    let mask: MLXArray? = l == 1 ? nil : causalMask(length: l, offset: total - l, dtype: .float16)
    func scores(_ q: MLXArray) -> MLXArray {
      quantizedMM(q, k3.0, scales: k3.1, biases: k3.2, transpose: true, groupSize: 64, bits: 3, mode: .affine)
    }
    func attend(_ q: MLXArray) -> MLXArray {
      var s = scores(q)
      var w = matmul(q, wk.swappedAxes(-1, -2))
      if let mask {
        s = s + mask[.ellipsis, 0..<t]
        w = w + mask[.ellipsis, t...]
      }
      let weights = softmax(concatenated([s, w], axis: -1), axis: -1, precise: true)
      let out = quantizedMM(weights[.ellipsis, 0..<t], v4.0, scales: v4.1, biases: v4.2, transpose: false, groupSize: 64, bits: 4, mode: .affine)
      return out + matmul(weights[.ellipsis, t...], wv)
    }
    let full = time { qs.map(attend) }
    let sc = time { qs.map(scores) }
    let ws = qs.map { softmax(concatenated([scores($0), matmul($0, wk.swappedAxes(-1, -2))], axis: -1), axis: -1, precise: true) }
    eval(ws)
    let vals = time { ws.map { quantizedMM($0[.ellipsis, 0..<t], v4.0, scales: v4.1, biases: v4.2, transpose: false, groupSize: 64, bits: 4, mode: .affine) } }
    let sm = time { qs.map { q -> MLXArray in
      let s0 = MLXRandom.normal([1, kvH, rep, l, total]).asType(.float16)
      return softmax(mask.map { s0 + $0 } ?? s0, axis: -1, precise: true)
    } }
    // Folded: the 6 query heads of a KV head and the l queries as rows of one product, M = 6l.
    let k3f = (keysQ.wq, keysQ.scales, keysQ.biases!)
    let v4f = (valsQ.wq, valsQ.scales, valsQ.biases!)
    let wkf = wk.squeezed(axis: 2)
    let wvf = wv.squeezed(axis: 2)
    let maskF: MLXArray? = mask.map { MLX.tiled($0, repetitions: [rep, 1]) }
    func attendFolded(_ q5: MLXArray) -> MLXArray {
      let q = q5.reshaped([1, kvH, rep * l, d])
      var s = quantizedMM(q, k3f.0, scales: k3f.1, biases: k3f.2, transpose: true, groupSize: 64, bits: 3, mode: .affine)
      var w = matmul(q, wkf.swappedAxes(-1, -2))
      if let maskF {
        s = s + maskF[.ellipsis, 0..<t]
        w = w + maskF[.ellipsis, t...]
      }
      let weights = softmax(concatenated([s, w], axis: -1), axis: -1, precise: true)
      let out = quantizedMM(weights[.ellipsis, 0..<t], v4f.0, scales: v4f.1, biases: v4f.2, transpose: false, groupSize: 64, bits: 4, mode: .affine)
      return (out + matmul(weights[.ellipsis, t...], wvf)).reshaped([1, kvH, rep, l, d])
    }
    let folded = time { qs.map(attendFolded) }
    let scF = time { qs.map { quantizedMM($0.reshaped([1, kvH, rep * l, d]), k3f.0, scales: k3f.1, biases: k3f.2, transpose: true, groupSize: 64, bits: 3, mode: .affine) } }
    let diff = abs(attend(qs[0]).asType(.float32) - attendFolded(qs[0]).asType(.float32)).max().item(Float.self)
    // Reference in float32 from the dequantized cache.
    let kd = concatenated([dequantized(keysQ.wq, scales: keysQ.scales, biases: keysQ.biases!, groupSize: 64, bits: 3).asType(.float32), wk.squeezed(axis: 2).asType(.float32)], axis: 2)
    let vd = concatenated([dequantized(valsQ.wq, scales: valsQ.scales, biases: valsQ.biases!, groupSize: 64, bits: 4).asType(.float32), wv.squeezed(axis: 2).asType(.float32)], axis: 2)
    let q32 = qs[0].reshaped([1, kvH, rep * l, d]).asType(.float32)
    var s32 = matmul(q32, kd.swappedAxes(-1, -2))
    if let maskF { s32 = s32 + maskF.asType(.float32) }
    let ref = matmul(softmax(s32, axis: -1, precise: true), vd).reshaped([1, kvH, rep, l, d])
    let eA = abs(attend(qs[0]).asType(.float32) - ref).max().item(Float.self)
    let eF = abs(attendFolded(qs[0]).asType(.float32) - ref).max().item(Float.self)
    print(String(format: "l=%d  %.0f  %.0f  %.0f  %.0f | folded: full %.0f scores %.0f, max|diff| %.3g | against f32: now %.3g, folded %.3g", l, full, sc, vals, sm, folded, scF, diff, eA, eF))
    // Which product: the scores of the 5-D (broadcast) form against f32, each query row alone.
    let kq32 = dequantized(keysQ.wq, scales: keysQ.scales, biases: keysQ.biases!, groupSize: 64, bits: 3).asType(.float32)
    let sNow = scores(qs[0]).asType(.float32)                                   // [1,4,6,l,t]
    let sRef = matmul(qs[0].asType(.float32), kq32.expandedDimensions(axis: 2).swappedAxes(-1, -2))
    for j in 0..<l {
      let e = abs(sNow[.ellipsis, j, 0...] - sRef[.ellipsis, j, 0...]).max().item(Float.self)
      let one = quantizedMM(qs[0][.ellipsis, j..<(j + 1), 0...], k3.0, scales: k3.1, biases: k3.2, transpose: true, groupSize: 64, bits: 3, mode: .affine).asType(.float32)
      let e1 = abs(one - sRef[.ellipsis, j..<(j + 1), 0...]).max().item(Float.self)
      print(String(format: "  scores row %d: in the l=%d product %.3g, alone %.3g", j, l, e, e1))
    }
    let wNow = softmax(sRef, axis: -1)
    let vq32 = dequantized(valsQ.wq, scales: valsQ.scales, biases: valsQ.biases!, groupSize: 64, bits: 4).asType(.float32)
    let oNow = quantizedMM(wNow.asType(.float16), v4.0, scales: v4.1, biases: v4.2, transpose: false, groupSize: 64, bits: 4, mode: .affine).asType(.float32)
    let oRef = matmul(wNow, vq32.expandedDimensions(axis: 2))
    for j in 0..<l {
      print(String(format: "  values row %d: %.3g", j, abs(oNow[.ellipsis, j, 0...] - oRef[.ellipsis, j, 0...]).max().item(Float.self)))
    }
  }
  if ProcessInfo.processInfo.environment["VB_ONLY_ATTN"] == "1" { exit(0) }
}

// Cold weights (16 §16): the chain above multiplies one weight 16 times, so part of it can stay
// in the system cache between products. Here the chain walks `copies` different weights of the
// shape (4 × 25 MB for gate/up, past the cache), as a forward does, which reads each weight once.
// Columns: M=1 MLX, M=1 magic, M=2 FewRowQMV, M=2 magic, M=2 MLX, then per ISHIZUKI_QMV_MODE in VB_MODES.
let copies = Int(ProcessInfo.processInfo.environment["VB_COPIES"] ?? "4") ?? 4
let modes = (ProcessInfo.processInfo.environment["VB_MODES"] ?? "").split(separator: ",").compactMap { Int($0) }
print("cold chain over \(copies) weights (us, max|diff| against MLX): M=1 mlx, M=1 magic, M=2 few-row, M=2 magic, M=2 mlx" + (modes.isEmpty ? "" : ", M=2 modes \(modes)"))
var coldPerToken = [Double](repeating: 0, count: 5)
for (label, n, k, count) in shapes {
  var ws: [(MLXArray, MLXArray, MLXArray)] = []
  for _ in 0..<copies {
    let packed = quantized(
      (MLXRandom.normal([n, k]) * 0.02).asType(.float16), groupSize: 128, bits: 2)
    ws.append((packed.wq, packed.scales, packed.biases!))
    eval(packed.wq, packed.scales, packed.biases!)
  }
  var cells: [String] = []
  var column = 0
  let cases: [(Int, Int, Int)] = [(1, 0, 0), (1, 1, 0), (2, 2, 0), (2, 1, 0), (2, 0, 0)] + modes.flatMap { [(1, 1, $0), (2, 1, $0)] }
  for (m, which, mode) in cases {
    FewRowQMV.magicMode = mode
    let xs = (0..<chain).map { _ in MLXRandom.normal([m, k]).asType(.float16) }
    eval(xs)
    func run(_ x: MLXArray, _ c: Int) -> MLXArray {
      let (w, scales, biases) = ws[c % copies]
      switch which {
      case 0: return quantizedMM(x, w, scales: scales, biases: biases, transpose: true, groupSize: 128, bits: 2, mode: .affine)
      case 1: return FewRowQMV.applyMagic(x, w, scales: scales, biases: biases, groupSize: 128, bits: 2)!
      default: return FewRowQMV.apply(x, w, scales: scales, biases: biases, groupSize: 128, bits: 2)!
      }
    }
    let (w0, s0, b0) = ws[0]
    let ref = quantizedMM(xs[0], w0, scales: s0, biases: b0, transpose: true, groupSize: 128, bits: 2, mode: .affine)
    let diff = abs(run(xs[0], 0).asType(.float32) - ref.asType(.float32)).max().item(Float.self)
    let t = time { xs.enumerated().map { run($0.element, $0.offset) } }
    if column < 5 { coldPerToken[column] += t * Double(count) }
    column += 1
    cells.append(String(format: "%.0f (%.3g)", t, diff))
  }
  FewRowQMV.magicMode = 0
  print("\(label)  \(cells.joined(separator: "  "))")
}
print(String(format: "cold, one backbone pass of matmuls: M=1 mlx %.1f ms, M=1 magic %.1f ms, M=2 few-row %.1f ms, M=2 magic %.1f ms, M=2 mlx %.1f ms",
  coldPerToken[0] / 1e3, coldPerToken[1] / 1e3, coldPerToken[2] / 1e3, coldPerToken[3] / 1e3, coldPerToken[4] / 1e3))
if ProcessInfo.processInfo.environment["VB_ONLY_COLD"] == "1" { exit(0) }

// FewRowQMV.applyMagic (mtp.patch): codes decoded two at a time through half2 (16 §15).
print("magic qmv (us, max|diff| against MLX): shape / M=1 mlx, M=1 magic, M=2 few-row, M=2 magic, M=3 magic, M=1 few-row")
var magicPerToken = (mlx1: 0.0, magic1: 0.0, few2: 0.0, magic2: 0.0)
for (label, n, k, count) in shapes {
  let packed = quantized(
    (MLXRandom.normal([n, k]) * 0.02).asType(.float16), groupSize: 128, bits: 2)
  let (w, scales, biases) = (packed.wq, packed.scales, packed.biases!)
  eval(w, scales, biases)
  var cells: [String] = []
  var got: [Double] = []
  for (m, which) in [(1, 0), (1, 1), (2, 2), (2, 1), (3, 1), (1, 2)] {
    let xs = (0..<chain).map { _ in MLXRandom.normal([m, k]).asType(.float16) }
    eval(xs)
    let ref = quantizedMM(xs[0], w, scales: scales, biases: biases, transpose: true, groupSize: 128, bits: 2, mode: .affine)
    func run(_ x: MLXArray) -> MLXArray {
      switch which {
      case 0: return quantizedMM(x, w, scales: scales, biases: biases, transpose: true, groupSize: 128, bits: 2, mode: .affine)
      case 1: return FewRowQMV.applyMagic(x, w, scales: scales, biases: biases, groupSize: 128, bits: 2)!
      default: return FewRowQMV.apply(x, w, scales: scales, biases: biases, groupSize: 128, bits: 2)!
      }
    }
    let diff = abs(run(xs[0]).asType(.float32) - ref.asType(.float32)).max().item(Float.self)
    let t = time { xs.map(run) }
    got.append(t)
    cells.append(String(format: "%.0f (%.3g)", t, diff))
  }
  magicPerToken.mlx1 += got[0] * Double(count)
  magicPerToken.magic1 += got[1] * Double(count)
  magicPerToken.few2 += got[2] * Double(count)
  magicPerToken.magic2 += got[3] * Double(count)
  print("\(label)  \(cells.joined(separator: "  "))")
}
print(String(format: "one backbone pass of matmuls: M=1 mlx %.1f ms, M=1 magic %.1f ms, M=2 few-row %.1f ms, M=2 magic %.1f ms",
  magicPerToken.mlx1 / 1e3, magicPerToken.magic1 / 1e3, magicPerToken.few2 / 1e3, magicPerToken.magic2 / 1e3))
// ISHIZUKI_QMV_MODE variants of the two-row magic kernel (16 §16): time and max|diff| against MLX.
if let modes = ProcessInfo.processInfo.environment["VB_MODES"] {
  print("magic M=2 by mode (us, max|diff|): shape / " + modes)
  for (label, n, k, _) in shapes.prefix(3) {
    let packed = quantized(
      (MLXRandom.normal([n, k]) * 0.02).asType(.float16), groupSize: 128, bits: 2)
    let (w, scales, biases) = (packed.wq, packed.scales, packed.biases!)
    eval(w, scales, biases)
    let xs = (0..<chain).map { _ in MLXRandom.normal([2, k]).asType(.float16) }
    eval(xs)
    let ref = quantizedMM(xs[0], w, scales: scales, biases: biases, transpose: true, groupSize: 128, bits: 2, mode: .affine)
    var cells: [String] = []
    for mode in modes.split(separator: ",").compactMap({ Int($0) }) {
      FewRowQMV.magicMode = mode
      let run = { (x: MLXArray) in FewRowQMV.applyMagic(x, w, scales: scales, biases: biases, groupSize: 128, bits: 2)! }
      let diff = abs(run(xs[0]).asType(.float32) - ref.asType(.float32)).max().item(Float.self)
      let t = time { xs.map(run) }
      cells.append(String(format: "%d: %.0f (%.3g)", mode, t, diff))
    }
    FewRowQMV.magicMode = 0
    print("\(label)  \(cells.joined(separator: "  "))")
  }
}
if ProcessInfo.processInfo.environment["VB_ONLY_MAGIC"] == "1" { exit(0) }

var perToken: [Int: (mlx: Double, kernel: Double, few: Double)] = [:]
print("shape                         M   mlx us   verify us   max|diff|   few-row us   max|diff|")
for (label, n, k, count) in shapes {
  let packed = quantized(
    (MLXRandom.normal([n, k]) * 0.02).asType(.float16), groupSize: 128, bits: 2)
  let (w, scales, biases) = (packed.wq, packed.scales, packed.biases!)
  eval(w, scales, biases)
  for m in 1...3 {
    let xs = (0..<chain).map { _ in MLXRandom.normal([m, k]).asType(.float16) }
    eval(xs)
    let mlx = time {
      xs.map {
        quantizedMM($0, w, scales: scales, biases: biases, transpose: true, groupSize: 128, bits: 2, mode: .affine)
      }
    }
    var kernel = Double.nan
    var diff = Float.nan
    if m >= 2 {
      VerifyMatmul.rowFloor = m
      if let got = VerifyMatmul.apply(xs[0], w, scales: scales, biases: biases, groupSize: 128, bits: 2) {
        let ref = quantizedMM(xs[0], w, scales: scales, biases: biases, transpose: true, groupSize: 128, bits: 2, mode: .affine)
        diff = abs(got.asType(.float32) - ref.asType(.float32)).max().item(Float.self)
        kernel = time {
          xs.map { VerifyMatmul.apply($0, w, scales: scales, biases: biases, groupSize: 128, bits: 2)! }
        }
      }
    }
    var few = Double.nan
    var fewDiff = Float.nan
    if let got = FewRowQMV.apply(xs[0], w, scales: scales, biases: biases, groupSize: 128, bits: 2) {
      let ref = quantizedMM(xs[0], w, scales: scales, biases: biases, transpose: true, groupSize: 128, bits: 2, mode: .affine)
      fewDiff = abs(got.asType(.float32) - ref.asType(.float32)).max().item(Float.self)
      few = time {
        xs.map { FewRowQMV.apply($0, w, scales: scales, biases: biases, groupSize: 128, bits: 2)! }
      }
    }
    print(String(format: "%@  %d  %8.1f  %9.1f   %9.3g   %10.1f   %.3g", label, m, mlx, kernel, diff, few, fewDiff))
    let sofar = perToken[m] ?? (0, 0, 0)
    perToken[m] = (
      sofar.mlx + mlx * Double(count),
      sofar.kernel + (kernel.isNaN ? mlx : min(mlx, kernel)) * Double(count),
      sofar.few + (few.isNaN ? mlx : few) * Double(count))
  }
}
for m in 1...3 {
  let t = perToken[m]!
  print(String(format: "M=%d  one backbone pass of matmuls: mlx %.1f ms, best of mlx/verify %.1f ms, few-row %.1f ms", m, t.mlx / 1e3, t.kernel / 1e3, t.few / 1e3))
}

// FewRowNAX (tensor ops, mtp.patch): the verify's rows on matmul2d, the lanes only decoding.
print("\nFewRowNAX (us, max|diff| against MLX): shape / M=1 qmv, M=2 nax, M=4 nax, M=8 nax")
for (label, n, k, _) in shapes {
  let packed = quantized(
    (MLXRandom.normal([n, k]) * 0.02).asType(.float16), groupSize: 128, bits: 2)
  let (w, scales, biases) = (packed.wq, packed.scales, packed.biases!)
  eval(w, scales, biases)
  var cells: [String] = []
  for m in [1, 2, 4, 8] {
    let xs = (0..<chain).map { _ in MLXRandom.normal([m, k]).asType(.float16) }
    eval(xs)
    if m == 1 {
      let t = time {
        xs.map { quantizedMM($0, w, scales: scales, biases: biases, transpose: true, groupSize: 128, bits: 2, mode: .affine) }
      }
      cells.append(String(format: "%.0f", t))
      continue
    }
    guard let got = FewRowNAX.apply(xs[0], w, scales: scales, biases: biases, groupSize: 128, bits: 2) else {
      cells.append("declined"); continue
    }
    let ref = quantizedMM(xs[0], w, scales: scales, biases: biases, transpose: true, groupSize: 128, bits: 2, mode: .affine)
    let diff = abs(got.asType(.float32) - ref.asType(.float32)).max().item(Float.self)
    let t = time { xs.map { FewRowNAX.apply($0, w, scales: scales, biases: biases, groupSize: 128, bits: 2)! } }
    cells.append(String(format: "%.0f (%.3g)", t, diff))
  }
  print("\(label)  \(cells.joined(separator: "  "))")
}

// MLX alone at larger row counts: from its qmv batch limit on, quantizedMM is qmm, which on a GPU
// with neural accelerators is the NAX (tensor ops) kernel. A verify padded up to such a row count
// pays that kernel's time, whatever the padding.
print("\nMLX quantizedMM by rows (us): shape / M=1 2 4 6 8 12 16 32 64")
for (label, n, k, _) in shapes {
  let packed = quantized(
    (MLXRandom.normal([n, k]) * 0.02).asType(.float16), groupSize: 128, bits: 2)
  let (w, scales, biases) = (packed.wq, packed.scales, packed.biases!)
  eval(w, scales, biases)
  var cells: [String] = []
  for m in [1, 2, 4, 6, 8, 12, 16, 32, 64] {
    let xs = (0..<chain).map { _ in MLXRandom.normal([m, k]).asType(.float16) }
    eval(xs)
    let t = time {
      xs.map {
        quantizedMM($0, w, scales: scales, biases: biases, transpose: true, groupSize: 128, bits: 2, mode: .affine)
      }
    }
    cells.append(String(format: "%.0f", t))
  }
  print("\(label)  \(cells.joined(separator: " "))")
}
