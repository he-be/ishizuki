// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// Target-token log-probabilities through the output head, a slice of rows at a time, so the
// full [tokens × vocab] logits never exist in either pass.

import MLX

public enum OutputHead {
  case dense(MLXArray)
  case quantized(weight: MLXArray, scales: MLXArray, biases: MLXArray?, groupSize: Int, bits: Int)

  var arrays: [MLXArray] {
    switch self {
    case .dense(let w): [w]
    case .quantized(let w, let s, let b, _, _): [w, s] + (b.map { [$0] } ?? [])
    }
  }

  func rebuilt(from a: ArraySlice<MLXArray>) -> OutputHead {
    let a = Array(a)
    switch self {
    case .dense: return .dense(a[0])
    case .quantized(_, _, _, let groupSize, let bits):
      return .quantized(
        weight: a[0], scales: a[1], biases: a.count > 2 ? a[2] : nil, groupSize: groupSize,
        bits: bits)
    }
  }

  func logits(_ h: MLXArray) -> MLXArray {
    switch self {
    case .dense(let w):
      matmul(h, w.transposed()).asType(.float32)
    case .quantized(let w, let s, let b, let groupSize, let bits):
      quantizedMatmul(
        h, w, scales: s, biases: b, transpose: true, groupSize: groupSize, bits: bits
      ).asType(.float32)
    }
  }

  func inputGrad(_ g: MLXArray, dtype: DType) -> MLXArray {
    switch self {
    case .dense(let w):
      matmul(g.asType(w.dtype), w)
    case .quantized(let w, let s, let b, let groupSize, let bits):
      quantizedMatmul(
        g.asType(dtype), w, scales: s, biases: b, transpose: false, groupSize: groupSize,
        bits: bits)
    }
  }
}

public func tokenLogProbs(
  _ hidden: MLXArray, head: OutputHead, targets: MLXArray, rowChunk: Int = 256
) -> MLXArray {
  let shape = targets.shape
  let dim = hidden.dim(-1)
  let h = hidden.reshaped(-1, dim)
  let t = targets.reshaped(-1).asType(.int32)
  let rows = h.dim(0)
  let bounds = stride(from: 0, to: rows, by: rowChunk).map { ($0, min($0 + rowChunk, rows)) }

  let f = CustomFunction {
    Forward { inputs in
      let (h, t, head) = (inputs[0], inputs[1], head.rebuilt(from: inputs[2...]))
      let parts = bounds.map { lo, hi in
        let z = head.logits(h[lo ..< hi])
        let zt = takeAlong(z, t[lo ..< hi].expandedDimensions(axis: -1), axis: -1).squeezed(
          axis: -1)
        return zt - logSumExp(z, axis: -1)
      }
      return [concatenated(parts, axis: 0)]
    }
    VJP { primals, cotangents in
      let (h, t, head) = (primals[0], primals[1], head.rebuilt(from: primals[2...]))
      let c = cotangents[0].asType(.float32)
      var dh: [MLXArray] = []
      var dw: MLXArray? = nil
      for (lo, hi) in bounds {
        let hc = h[lo ..< hi]
        let z = head.logits(hc)
        let vocab = MLXArray(0 ..< Int32(z.dim(-1)))
        let onehot = (vocab .== t[lo ..< hi].expandedDimensions(axis: -1)).asType(.float32)
        let g = c[lo ..< hi].expandedDimensions(axis: -1) * (onehot - softmax(z, axis: -1))
        dh.append(head.inputGrad(g, dtype: h.dtype))
        if case .dense = head {
          let part = matmul(g.transposed(), hc.asType(.float32))
          dw = dw.map { $0 + part } ?? part
        }
      }
      var grads = [concatenated(dh, axis: 0).asType(h.dtype), zeros(like: t)]
      switch head {
      case .dense(let w): grads.append(dw!.asType(w.dtype))
      case .quantized: grads += primals[2...].map { zeros(like: $0) }
      }
      return grads
    }
  }
  return f([h, t] + head.arrays)[0].reshaped(shape)
}
