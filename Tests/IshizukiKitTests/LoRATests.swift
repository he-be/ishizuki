// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// Adapters on a rotated, quantized projection: the forward, every gradient, and a round trip.

import Foundation
import MLX
import MLXRandom
import Testing

@testable import IshizukiKit

@Suite("LoRA", .serialized)
struct LoRATests {
  private let rows = 12
  private let inputDim = 128
  private let outputDim = 96
  private let block = 64

  private func projection() throws -> (PackedLinear, MLXArray, MLXArray) {
    let w = MLXRandom.normal([outputDim, inputDim]) * 0.05
    let signs = MLX.where(MLXRandom.bernoulli(0.5, [inputDim]), Float(1), Float(-1))
    let (wq, s, b) = MLX.quantized(w, groupSize: 64, bits: 4)
    let linear = try PackedLinear(
      weight: wq, scales: s, biases: b!, signs: signs, block: block, groupSize: 64, bits: 4)
    let dense = dequantized(wq, scales: s, biases: b, groupSize: 64, bits: 4)
    return (linear, dense, signs)
  }

  private func maxDiff(_ a: MLXArray, _ b: MLXArray) -> Float {
    abs(a.asType(.float32) - b.asType(.float32)).max().item(Float.self)
  }

  @Test("a fresh adapter leaves the projection unchanged")
  func freshIsIdentity() throws {
    MLXRandom.seed(3)
    let (linear, _, _) = try projection()
    let x = MLXRandom.normal([rows, inputDim])
    let before = linear(x)
    linear.lora = LoRA(inputDim: inputDim, outputDim: outputDim, rank: 8, scale: 2)
    #expect(maxDiff(linear(x), before) == 0)
  }

  @Test("gradients through the rotated quantized base match a dense reference")
  func gradients() throws {
    MLXRandom.seed(5)
    BonsaiRuntime.differentiable = true
    defer { BonsaiRuntime.differentiable = false }
    let (linear, dense, signs) = try projection()
    let lora = LoRA(inputDim: inputDim, outputDim: outputDim, rank: 8, scale: 2)
    lora.b = MLXRandom.normal([8, outputDim]) * 0.1
    linear.lora = lora
    let x = MLXRandom.normal([rows, inputDim])
    let c = MLXRandom.normal([rows, outputDim])

    let fused = valueAndGrad(
      { p in
        lora.a = p[1]
        lora.b = p[2]
        return [(linear(p[0]) * c).sum()]
      }, argumentNumbers: [0, 1, 2])
    let naive = valueAndGrad(
      { p in
        let rotated = hadamardRotate(p[0], block: block, signs: signs)
        let y = matmul(rotated, dense.T) + matmul(matmul(p[0], p[1]), p[2]) * 2
        return [(y * c).sum()]
      }, argumentNumbers: [0, 1, 2])

    let (a0, b0) = (lora.a, lora.b)
    let (fv, fg) = fused([x, a0, b0])
    let (nv, ng) = naive([x, a0, b0])
    #expect(maxDiff(fv[0], nv[0]) < 1e-2)
    for (f, n) in zip(fg, ng) { #expect(maxDiff(f, n) < 1e-3) }
  }

  @Test("adapters save and load by module path, in mlx-lm's layout")
  func roundTrip() throws {
    MLXRandom.seed(9)
    let (q, _, _) = try projection()
    let (o, _, _) = try projection()
    let linears = ["model.layers.0.self_attn.q_proj": q, "model.layers.0.mlp.o_proj": o]
    let config = LoRAConfig(rank: 4, alpha: 8, targets: ["q_proj"])
    let adapters = Adapters.attach(to: linears, config: config)
    #expect(adapters.modules.map(\.path) == ["model.layers.0.self_attn.q_proj"])
    #expect(o.lora == nil)
    adapters.modules[0].lora.b = MLXRandom.normal([4, outputDim])

    let dir = FileManager.default.temporaryDirectory.appending(path: UUID().uuidString)
    defer { try? FileManager.default.removeItem(at: dir) }
    try adapters.save(to: dir)
    let json = try String(contentsOf: dir.appending(path: "adapter_config.json"), encoding: .utf8)
    #expect(json.contains("\"self_attn.q_proj\""))

    let x = MLXRandom.normal([rows, inputDim])
    let expected = q(x)
    adapters.detach(from: linears)
    let loaded = try Adapters.load(from: dir, into: linears)
    #expect(loaded.config.scale == 2)
    #expect(maxDiff(q(x), expected) == 0)
  }
}
