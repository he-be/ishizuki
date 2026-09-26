// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// Low-rank adapters on the packed projections, stored the way mlx-lm stores them so a pair
// trained here loads there and back.

import Foundation
import MLX
import MLXRandom

public final class LoRA: @unchecked Sendable {
  public var a: MLXArray
  public var b: MLXArray
  public let scale: Float
  public let dropout: Float

  public var rank: Int { a.dim(1) }

  public init(inputDim: Int, outputDim: Int, rank: Int, scale: Float, dropout: Float = 0) {
    let bound = 1 / Float(inputDim).squareRoot()
    self.a = MLXRandom.uniform(low: -bound, high: bound, [inputDim, rank], dtype: .float32)
    self.b = MLXArray.zeros([rank, outputDim], dtype: .float32)
    self.scale = scale
    self.dropout = dropout
  }

  public init(a: MLXArray, b: MLXArray, scale: Float, dropout: Float = 0) {
    self.a = a
    self.b = b
    self.scale = scale
    self.dropout = dropout
  }

  public func callAsFunction(_ x: MLXArray) -> MLXArray {
    var h = x
    if dropout > 0, BonsaiRuntime.differentiable {
      let keep = MLXRandom.bernoulli(MLXArray(1 - dropout), h.shape)
      h = MLX.where(keep, h / (1 - dropout), MLXArray.zeros(like: h))
    }
    return matmul(matmul(h, a.asType(h.dtype)), b.asType(h.dtype)) * scale
  }
}

public struct LoRAConfig: Codable, Sendable {
  public var rank: Int
  public var alpha: Float
  public var dropout: Float
  public var targets: [String]
  var layerKeys: [String]?

  public init(
    rank: Int = 16, alpha: Float = 32, dropout: Float = 0,
    targets: [String] = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
  ) {
    self.rank = rank
    self.alpha = alpha
    self.dropout = dropout
    self.targets = targets
  }

  public var scale: Float { alpha / Float(rank) }

  private enum Keys: String, CodingKey {
    case fineTuneType = "fine_tune_type", numLayers = "num_layers", parameters = "lora_parameters"
  }

  private enum ParameterKeys: String, CodingKey { case rank, scale, dropout, keys }

  public init(from decoder: Decoder) throws {
    let outer = try decoder.container(keyedBy: Keys.self)
    let inner = try outer.nestedContainer(keyedBy: ParameterKeys.self, forKey: .parameters)
    rank = try inner.decode(Int.self, forKey: .rank)
    alpha = try inner.decode(Float.self, forKey: .scale) * Float(rank)
    dropout = try inner.decodeIfPresent(Float.self, forKey: .dropout) ?? 0
    targets = Array(
      Set((try inner.decodeIfPresent([String].self, forKey: .keys) ?? []).map {
        String($0.split(separator: ".").last ?? "")
      })
    ).sorted()
  }

  public func encode(to encoder: Encoder) throws {
    var outer = encoder.container(keyedBy: Keys.self)
    try outer.encode("lora", forKey: .fineTuneType)
    try outer.encode(-1, forKey: .numLayers)
    var inner = outer.nestedContainer(keyedBy: ParameterKeys.self, forKey: .parameters)
    try inner.encode(rank, forKey: .rank)
    try inner.encode(scale, forKey: .scale)
    try inner.encode(dropout, forKey: .dropout)
    try inner.encode(layerKeys ?? targets, forKey: .keys)
  }
}

public struct Adapters: Sendable {
  public let config: LoRAConfig
  public let modules: [(path: String, lora: LoRA)]

  public static func attach(to linears: [String: PackedLinear], config: LoRAConfig) -> Adapters {
    let targets = Set(config.targets)
    let chosen = linears.filter { path, _ in
      targets.contains(String(path.split(separator: ".").last ?? ""))
    }.sorted { $0.key < $1.key }
    let modules = chosen.map { path, linear in
      let lora = LoRA(
        inputDim: linear.inputDim, outputDim: linear.outputDim, rank: config.rank,
        scale: config.scale, dropout: config.dropout)
      linear.lora = lora
      return (path, lora)
    }
    return Adapters(config: config, modules: modules)
  }

  public static func load(
    from directory: URL, into linears: [String: PackedLinear]
  ) throws -> Adapters {
    let raw = try Data(contentsOf: directory.appending(path: "adapter_config.json"))
    let config = try JSONDecoder().decode(LoRAConfig.self, from: raw)
    let arrays = try loadArrays(url: directory.appending(path: "adapters.safetensors"))
    var modules: [(String, LoRA)] = []
    for (path, linear) in linears.sorted(by: { $0.key < $1.key }) {
      guard let a = arrays[path + ".lora_a"], let b = arrays[path + ".lora_b"] else { continue }
      let lora = LoRA(
        a: a.asType(.float32), b: b.asType(.float32), scale: config.scale,
        dropout: config.dropout)
      linear.lora = lora
      modules.append((path, lora))
    }
    return Adapters(config: config, modules: modules)
  }

  public func detach(from linears: [String: PackedLinear]) {
    for (path, _) in modules { linears[path]?.lora = nil }
  }

  public var parameters: [MLXArray] { modules.flatMap { [$0.lora.a, $0.lora.b] } }

  public func assign(_ parameters: [MLXArray]) {
    precondition(parameters.count == 2 * modules.count)
    for (i, module) in modules.enumerated() {
      module.lora.a = parameters[2 * i]
      module.lora.b = parameters[2 * i + 1]
    }
  }

  public var parameterCount: Int { parameters.reduce(0) { $0 + $1.size } }

  public func save(to directory: URL) throws {
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    var arrays: [String: MLXArray] = [:]
    for (path, lora) in modules {
      arrays[path + ".lora_a"] = lora.a
      arrays[path + ".lora_b"] = lora.b
    }
    try MLX.save(arrays: arrays, url: directory.appending(path: "adapters.safetensors"))
    let encoder = JSONEncoder()
    encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
    var config = config
    config.layerKeys = Array(
      Set(modules.map { module in
        let parts = module.path.split(separator: ".")
        let layer = parts.firstIndex(where: { Int($0) != nil }).map { $0 + 1 } ?? 0
        return parts[layer...].joined(separator: ".")
      })
    ).sorted()
    try encoder.encode(config).write(to: directory.appending(path: "adapter_config.json"))
  }
}
