// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// Adapters trained through the whole hybrid stack: gradients against finite differences,
// and a batch the model learns.

import Foundation
import MLX
import MLXRandom
import Testing

@testable import IshizukiKit

@Suite("SFT", .serialized)
struct SFTTests {
  private var fixture: URL {
    URL(fileURLWithPath: #filePath)
      .deletingLastPathComponent()
      .appending(path: "Fixtures/qwen4-exp")
  }

  private func model(scratch: URL) throws -> TextModel {
    let source = try SourceCheckpoint(directory: fixture)
    _ = try EngramRepack.run(source: source, destination: scratch)
    var arrays: [String: MLXArray] = [:]
    for name in source.tensorNames
    where !name.hasPrefix("model.ngram_embedding.") && !name.hasPrefix("model.ple_embedding.") {
      arrays[name] = TensorNaming.relayout(
        name, try source.tensor(name), zeroCentredNorms: true
      ).asType(.float32)
    }
    let store = try WeightStore(arrays: arrays).openingEngrams(at: scratch)
    let config = try BonsaiConfig.load(directory: fixture)
    let factory = PackedModuleFactory(
      store: store, config: config, tensorPrefix: "", dense: true, activationDType: .float32)
    return try TextModel(config: config, factory: factory, store: store)
  }

  private func withModel(_ body: (TextModel) throws -> Void) throws {
    let scratch = URL(fileURLWithPath: NSTemporaryDirectory())
      .appending(path: "sft-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: scratch, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: scratch) }
    try body(try model(scratch: scratch))
  }

  private var batch: SFTBatch {
    let a: [Int32] = [1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45]
    let b: [Int32] = [2, 4, 8, 16, 32, 3, 6, 12, 24, 48]
    return SFTBatch([
      SFTExample(tokens: a, trained: a.indices.map { $0 >= 3 }),
      SFTExample(tokens: b, trained: b.indices.map { $0 >= 2 }),
    ])
  }

  @Test("adapters reach every kind of layer, and the gradient matches finite differences")
  func gradientCheck() throws {
    try withModel { model in
      MLXRandom.seed(21)
      let adapters = Adapters.attach(
        to: model.linears, config: LoRAConfig(rank: 4, alpha: 4, targets: ["q_proj", "o_proj", "out_proj", "down_proj", "in_proj_qkv"]))
      #expect(adapters.modules.count > 4)
      for module in adapters.modules {
        module.lora.b = MLXRandom.normal(module.lora.b.shape) * 0.05
      }
      let trainer = SFTTrainer(model: model, adapters: adapters, learningRate: 1e-3)
      let (_, grads) = trainer.gradients(batch)
      #expect(grads.allSatisfy { !isNaN($0).any().item(Bool.self) })

      BonsaiRuntime.differentiable = true
      defer { BonsaiRuntime.differentiable = false }
      let eps: Float = 1e-2
      for (index, module) in adapters.modules.enumerated() where index % 2 == 0 {
        let original = module.lora.a
        let bump = MLXArray.zeros(like: original)
        bump[0, 0] = MLXArray(eps)
        module.lora.a = original + bump
        let up = trainer.loss(batch).item(Float.self)
        module.lora.a = original - bump
        let down = trainer.loss(batch).item(Float.self)
        module.lora.a = original
        let numeric = (up - down) / (2 * eps)
        let analytic = grads[2 * index][0, 0].item(Float.self)
        #expect(
          abs(numeric - analytic) < 2e-3 + 0.05 * abs(numeric),
          "\(module.path): numeric \(numeric) analytic \(analytic)")
      }
    }
  }

  @Test("sixty steps drive a fixed batch's loss well down")
  func learns() throws {
    try withModel { model in
      MLXRandom.seed(4)
      let adapters = Adapters.attach(to: model.linears, config: LoRAConfig(rank: 8, alpha: 16))
      let trainer = SFTTrainer(model: model, adapters: adapters, learningRate: 5e-2)
      let first = trainer.step(batch).loss
      var last = first
      for _ in 0 ..< 60 { last = trainer.step(batch).loss }
      #expect(last < first - 0.5, "loss \(first) → \(last)")
    }
  }
}
