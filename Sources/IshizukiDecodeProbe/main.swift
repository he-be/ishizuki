// SPDX-License-Identifier: AGPL-3.0-or-later
// Decode probe for the MTP verify on the M6 (docs/qwen38-27b/16 §14). Loads the pack once and,
// after prefilling a prompt file:
//   1. times one backbone forward of M = 1, 2, 3 tokens (logits at every position, the drafted
//      loop's verify), with the cache restored after each, median of `reps`;
//   2. splits the same forward by layer kind with an eval after every layer: embedding,
//      delta-net layers, attention layers, final norm + output head;
//   3. generates `gen` tokens from the prompt, three seeds each, with drafting off and with the
//      MTP head (whole-block restore and partial keep of a rejected block), the server's official
//      thinking-off sampling (0.7 / 20 / 0.8 / 0.0 / presence 1.5), and reports tok/s and the
//      acceptance.
// Step 1 runs with MLX, FewRowQMV and its half2 variant (mtp.patch) for the two-row products,
// after a check of their verify logits against MLX. Step 2 (split) is defined but not run.
//
//   ishizuki-decode-probe <pack> <prompt.txt> [reps 10] [gen 300]
//
// It is a model process: run it only while no ishizuki-serve holds the model.

import Foundation
import IshizukiKit
import MLX

let args = CommandLine.arguments
guard args.count >= 3 else {
  FileHandle.standardError.write(
    "usage: ishizuki-decode-probe <pack> <prompt.txt> [reps] [gen]\n".data(using: .utf8)!)
  exit(2)
}
let pack = URL(fileURLWithPath: args[1])
let promptText = try String(contentsOfFile: args[2], encoding: .utf8)
let reps = args.count > 3 ? Int(args[3]) ?? 10 : 10
let genTokens = args.count > 4 ? Int(args[4]) ?? 300 : 300

func ms(_ start: UInt64) -> Double { Double(DispatchTime.now().uptimeNanoseconds - start) / 1e6 }
func median(_ xs: [Double]) -> Double { xs.sorted()[xs.count / 2] }
func say(_ s: String) { print(s); fflush(stdout) }

let model = try BonsaiModel(path: pack)
let text = model.text!
say("mtp head: \(model.hasMTP)")
let tokens = model.tokenizer.encode(promptText)

// Prefill once into a cache the forwards extend and give back.
let cache = text.makeCache()
var index = 0
while index < tokens.count {
  let end = min(index + 512, tokens.count)
  let chunk = MLXArray(tokens[index..<end].map { Int32($0) }).reshaped([1, end - index])
  eval(text.trunk(inputs: chunk, cache: cache))
  index = end
}
say("prefilled \(tokens.count) tokens, cache offset \(cache.offset)")

let probe = [tokens[tokens.count / 2], tokens[tokens.count / 3], tokens[tokens.count / 4]]

func forward(_ m: Int) -> Double {
  let ids = MLXArray(probe.prefix(m).map { Int32($0) }).reshaped([1, m])
  var runs: [Double] = []
  for _ in 0..<(reps + 2) {
    let snapshot = cache.snapshot()
    let start = DispatchTime.now().uptimeNanoseconds
    let h = text.trunk(inputs: ids, cache: cache)
    let logits = text.lmHead(text.normed(h))
    eval(logits)
    runs.append(ms(start))
    cache.restore(snapshot)
  }
  return median(Array(runs.dropFirst(2)))
}

func split(_ m: Int) -> [String: Double] {
  let ids = MLXArray(probe.prefix(m).map { Int32($0) }).reshaped([1, m])
  var parts: [String: [Double]] = [:]
  for _ in 0..<(reps + 2) {
    let snapshot = cache.snapshot()
    var t = DispatchTime.now().uptimeNanoseconds
    var h = text.embedTokens(ids)
    eval(h)
    var sums: [String: Double] = ["embed": ms(t)]
    let compute = h.dtype
    let mask = causalMask(length: m, offset: cache.offset, dtype: compute)
    h = h.asType(.float32)
    var linear = 0.0
    var attention = 0.0
    for (i, layer) in text.layers.enumerated() {
      t = DispatchTime.now().uptimeNanoseconds
      h = layer(h, mask: mask, cache: cache.layers[i], positions: nil, compute: compute)
      eval(h)
      if layer.isLinear { linear += ms(t) } else { attention += ms(t) }
    }
    t = DispatchTime.now().uptimeNanoseconds
    let logits = text.lmHead(text.normed(h.asType(compute)))
    eval(logits)
    sums["head"] = ms(t)
    sums["deltanet"] = linear
    sums["attention"] = attention
    for (k, v) in sums { parts[k, default: []].append(v) }
    cache.restore(snapshot)
  }
  return parts.mapValues { median(Array($0.dropFirst(2))) }
}

// The 2-token verify against two 1-token forwards (16 §16): row j of the block should be the
// logits a plain decode gives after the same tokens. With the quantized-KV attention's broadcast
// form (ISHIZUKI_ATTN_FOLD=0) and with the folded one.
FewRowQMV.enabled = true
FewRowQMV.magicEnabled = true
func logitsOf(_ ids: [Int]) -> MLXArray {
  let x = MLXArray(ids.map { Int32($0) }).reshaped([1, ids.count])
  return text.lmHead(text.normed(text.trunk(inputs: x, cache: cache))).asType(.float32)
}
do {
  let snapshot = cache.snapshot()
  let a = logitsOf([probe[0]])
  let b = logitsOf([probe[1]])
  eval(a, b)
  cache.restore(snapshot)
  for fold in [false, true] {
    Attention.foldsVerify = fold
    let block = logitsOf(Array(probe.prefix(2)))
    eval(block)
    cache.restore(snapshot)
    let d0 = abs(block[0..., 0, 0...] - a[0..., 0, 0...]).max().item(Float.self)
    let d1 = abs(block[0..., 1, 0...] - b[0..., 0, 0...]).max().item(Float.self)
    let m0 = (block[0..., 0, 0...].argMax(axis: -1) .== a[0..., 0, 0...].argMax(axis: -1)).all().item(Bool.self)
    let m1 = (block[0..., 1, 0...].argMax(axis: -1) .== b[0..., 0, 0...].argMax(axis: -1)).all().item(Bool.self)
    // Total variation between the two rows' softmax at temperature 0.7, the server's sampling.
    func tv(_ x: MLXArray, _ y: MLXArray) -> Float {
      (abs(softmax(x / 0.7, axis: -1) - softmax(y / 0.7, axis: -1)).sum() / 2).item(Float.self)
    }
    say(String(format: "fold %@: row 0 vs 1-token max|diff| %.4f argmax %@ TV %.4f | row 1 vs 1-token max|diff| %.4f argmax %@ TV %.4f",
      fold ? "on " : "off", d0, m0 ? "=" : "!=", tv(block[0..., 0, 0...], a[0..., 0, 0...]),
      d1, m1 ? "=" : "!=", tv(block[0..., 1, 0...], b[0..., 0, 0...])))
  }
}

for pass in 0..<2 {
  for fold in [false, true] {
    Attention.foldsVerify = fold
    var cells: [String] = []
    for m in 1...2 { cells.append(String(format: "M=%d %.1f ms", m, forward(m))) }
    say("forward pass \(pass + 1) (magic, fold \(fold ? "on" : "off")): \(cells.joined(separator: ", "))")
  }
}
if ProcessInfo.processInfo.environment["PROBE_NO_GEN"] == "1" { exit(0) }

// End to end, as the server samples: thinking off, official values, three seeds per setting,
// partial keep, magic for the two-row products.
BonsaiRuntime.lookupMinMatch = Int(Int32.max)
setenv("ISHIZUKI_SPEC_KEEP", "1", 1)
let settings: [(String, Bool, Bool)] = [
  ("draft off          ", false, true),
  ("mtp, fold off      ", true, false),
  ("mtp, fold on       ", true, true),
]
for (label, drafting, fold) in settings {
  Attention.foldsVerify = fold
  BonsaiRuntime.speculativeDecode = drafting
  var rates: [String] = []
  var accepted = 0
  var proposed = 0
  for seed in [1, 2, 3] as [UInt64] {
    var official = SamplingOptions(
      temperature: 0.7, topP: 0.8, topK: 20, minP: 0.0, presencePenalty: 1.5)
    official.seed = seed
    let result = Generator(model: model).generate(
      promptTokens: tokens, options: official, maxTokens: genTokens)
    rates.append(String(format: "%.2f", result.stats.generationTokensPerSecond))
    accepted += result.speculative?.accepted ?? 0
    proposed += result.speculative?.proposed ?? 0
  }
  say("generate \(label): tok/s \(rates.joined(separator: " / ")), accepted \(accepted)/\(proposed)")
}
