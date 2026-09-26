// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// Supervised fine-tuning of the adapters alone: masked next-token loss, AdamW, clipped norm.

import Foundation
import MLX
import MLXOptimizers

public struct SFTExample: Sendable {
  public let tokens: [Int32]
  public let trained: [Bool]

  public init(tokens: [Int32], trained: [Bool]) {
    precondition(tokens.count == trained.count)
    self.tokens = tokens
    self.trained = trained
  }
}

public struct SFTBatch {
  public let inputs: MLXArray
  public let targets: MLXArray
  public let mask: MLXArray

  public init(_ examples: [SFTExample], padding: Int32 = 0) {
    let width = (examples.map(\.tokens.count).max() ?? 1) - 1
    var inputs: [Int32] = []
    var targets: [Int32] = []
    var mask: [Float] = []
    for example in examples {
      let n = example.tokens.count - 1
      let pad = width - n
      inputs += example.tokens.dropLast() + Array(repeating: padding, count: pad)
      targets += example.tokens.dropFirst() + Array(repeating: padding, count: pad)
      mask += example.trained.dropFirst().map { $0 ? 1 : 0 } + Array(repeating: 0, count: pad)
    }
    let shape = [examples.count, width]
    self.inputs = MLXArray(inputs, shape)
    self.targets = MLXArray(targets, shape)
    self.mask = MLXArray(mask, shape)
  }

  public var trainedTokens: Int { Int(mask.sum().item(Float.self)) }
}

extension OutputHead {
  public init(_ linear: PackedLinear) {
    precondition(linear.ggml == nil && linear.exl3 == nil, "coded output heads cannot train yet")
    if linear.isDense {
      self = .dense(linear.weight)
    } else {
      self = .quantized(
        weight: linear.weight, scales: linear.scales, biases: linear.biases,
        groupSize: linear.groupSize, bits: linear.bits)
    }
  }
}

extension TextModel {
  public func tokenLogProbs(inputs: MLXArray, targets: MLXArray, rowChunk: Int = 256) -> MLXArray
  {
    let h = hidden(inputs: inputs)
    return IshizukiKit.tokenLogProbs(
      lmHead.rotate(h), head: OutputHead(lmHead), targets: targets, rowChunk: rowChunk)
  }
}

public final class SFTTrainer {
  public let model: TextModel
  public let adapters: Adapters
  public let optimizer: AdamW
  public var maxGradNorm: Float?
  private var states: [AdamState]

  public init(
    model: TextModel, adapters: Adapters, learningRate: Float, weightDecay: Float = 0,
    maxGradNorm: Float? = 1
  ) {
    self.model = model
    self.adapters = adapters
    let optimizer = AdamW(
      learningRate: learningRate, weightDecay: weightDecay, biasCorrection: true)
    self.optimizer = optimizer
    self.maxGradNorm = maxGradNorm
    self.states = adapters.parameters.map { optimizer.newState(parameter: $0) }
  }

  public func loss(_ batch: SFTBatch) -> MLXArray {
    let total = (0 ..< batch.inputs.dim(0)).map { row in
      let logp = model.tokenLogProbs(
        inputs: batch.inputs[row ..< row + 1], targets: batch.targets[row ..< row + 1])
      return (logp * batch.mask[row ..< row + 1]).sum()
    }.reduce(MLXArray(Float(0)), +)
    return -total / maximum(batch.mask.sum(), 1)
  }

  public func gradients(_ batch: SFTBatch) -> (loss: MLXArray, grads: [MLXArray]) {
    BonsaiRuntime.differentiable = true
    defer { BonsaiRuntime.differentiable = false }
    let parameters = adapters.parameters
    let f = valueAndGrad(
      { [self] (p: [MLXArray]) -> [MLXArray] in
        adapters.assign(p)
        return [loss(batch)]
      }, argumentNumbers: parameters.indices)
    let (value, grads) = f(parameters)
    adapters.assign(parameters)
    return (value[0], grads)
  }

  @discardableResult
  public func step(_ batch: SFTBatch) -> (loss: Float, gradNorm: Float) {
    var (loss, grads) = gradients(batch)
    let norm = sqrt(grads.map { square($0).sum() }.reduce(MLXArray(Float(0)), +))
    if let maxGradNorm {
      let factor = minimum(MLXArray(maxGradNorm) / (norm + 1e-6), 1)
      grads = grads.map { $0 * factor }
    }
    var updated: [MLXArray] = []
    for (i, (g, p)) in zip(grads, adapters.parameters).enumerated() {
      let (next, state) = optimizer.applySingle(gradient: g, parameter: p, state: states[i])
      updated.append(next)
      states[i] = state
    }
    adapters.assign(updated)
    eval([loss, norm] + updated + states.flatMap { $0.innerState() })
    return (loss.item(Float.self), norm.item(Float.self))
  }
}
