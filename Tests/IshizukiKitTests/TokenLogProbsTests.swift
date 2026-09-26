// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// The chunked head against the whole-vocabulary one, values and gradients both.

import Foundation
import MLX
import MLXRandom
import Testing

@testable import IshizukiKit

@Suite("Token log-probs")
struct TokenLogProbsTests {
  private let rows = 37
  private let dim = 64
  private let vocab = 300

  private func reference(_ h: MLXArray, _ w: MLXArray, _ t: MLXArray) -> MLXArray {
    let z = matmul(h, w.transposed()).asType(.float32)
    let logp = z - logSumExp(z, axis: -1, keepDims: true)
    return takeAlong(logp, t.expandedDimensions(axis: -1), axis: -1).squeezed(axis: -1)
  }

  private func maxDiff(_ a: MLXArray, _ b: MLXArray) -> Float {
    abs(a.asType(.float32) - b.asType(.float32)).max().item(Float.self)
  }

  @Test("dense head: values and both gradients match autodiff", arguments: [8, 16, 256])
  func dense(rowChunk: Int) {
    MLXRandom.seed(7)
    let h = MLXRandom.normal([rows, dim])
    let w = MLXRandom.normal([vocab, dim]) * 0.1
    let t = MLXRandom.randInt(0 ..< vocab, [rows]).asType(.int32)
    let c = MLXRandom.normal([rows])

    let fused = valueAndGrad(
      { a in [(tokenLogProbs(a[0], head: .dense(a[1]), targets: t, rowChunk: rowChunk) * c).sum()] },
      argumentNumbers: [0, 1])
    let naive = valueAndGrad(
      { a in [(reference(a[0], a[1], t) * c).sum()] }, argumentNumbers: [0, 1])

    let (fv, fg) = fused([h, w])
    let (nv, ng) = naive([h, w])
    #expect(maxDiff(fv[0], nv[0]) < 1e-3)
    #expect(maxDiff(fg[0], ng[0]) < 1e-4)
    #expect(maxDiff(fg[1], ng[1]) < 1e-4)
    #expect(
      maxDiff(tokenLogProbs(h, head: .dense(w), targets: t, rowChunk: rowChunk), reference(h, w, t))
        < 1e-4)
  }

  @Test("quantized head: input gradient matches the dequantized reference")
  func quantized() {
    MLXRandom.seed(11)
    let h = MLXRandom.normal([rows, dim])
    let (wq, s, b) = MLXArray.quantizedTriple(MLXRandom.normal([vocab, dim]) * 0.1)
    let w = dequantized(wq, scales: s, biases: b, groupSize: 64, bits: 4)
    let t = MLXRandom.randInt(0 ..< vocab, [rows]).asType(.int32)
    let c = MLXRandom.normal([rows])
    let head = OutputHead.quantized(weight: wq, scales: s, biases: b, groupSize: 64, bits: 4)

    let fused = valueAndGrad(
      { a in [(tokenLogProbs(a[0], head: head, targets: t, rowChunk: 16) * c).sum()] })
    let naive = valueAndGrad({ a in [(reference(a[0], w, t) * c).sum()] })
    let (fv, fg) = fused([h])
    let (nv, ng) = naive([h])
    #expect(maxDiff(fv[0], nv[0]) < 1e-3)
    #expect(maxDiff(fg[0], ng[0]) < 1e-4)
  }

  @Test("batched shapes pass through")
  func shapes() {
    let h = MLXRandom.normal([2, 5, dim])
    let w = MLXRandom.normal([vocab, dim])
    let t = MLXRandom.randInt(0 ..< vocab, [2, 5]).asType(.int32)
    #expect(tokenLogProbs(h, head: .dense(w), targets: t, rowChunk: 3).shape == [2, 5])
  }
}

extension MLXArray {
  fileprivate static func quantizedTriple(_ w: MLXArray) -> (MLXArray, MLXArray, MLXArray) {
    let (wq, s, b) = MLX.quantized(w, groupSize: 64, bits: 4)
    return (wq, s, b!)
  }
}
