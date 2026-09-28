// SPDX-License-Identifier: AGPL-3.0-or-later
// Prefill driver for ishizuki (struffl/ishizuki), built on the M6 from the he-be/ishizuki fork
// (bench/m6/ishizuki_build_m6.sh) and run there
// (docs/qwen38-27b/15-M6-PREFILL-PLAN.md §3-2, route B). ishizuki has no CLI: its
// benches live in the menu bar app and in `swift test`, neither of which runs on a
// headless Mac without Xcode. This is the pp part of Tests/IshizukiKitTests/
// EngineBenchProbe.swift, fed a prompt file instead of a repeated passage.
//
//   ishizuki-prefill <pack directory> <prompt.txt>
//
// Prints one JSON line: prompt tokens, prefill seconds (Generator's promptSeconds, the
// same number EngineBenchProbe reads), load seconds, peak GPU memory, first token.
// A warm pass over the first 8 tokens runs before the timed one so the timed pass does
// not include the first page-in of the weights (bench/m6/ref_mlxpack_bonsai.sh does the
// same for the pack's own runtime).

import Foundation
import IshizukiKit
import MLX

let args = CommandLine.arguments
guard args.count == 3 else {
  FileHandle.standardError.write("usage: ishizuki-prefill <pack> <prompt.txt>\n".data(using: .utf8)!)
  exit(2)
}
let pack = URL(fileURLWithPath: args[1])
let promptText = try String(contentsOfFile: args[2], encoding: .utf8)

// Drafting is off for the whole process (the generator's own switch is internal to the kit);
// with maxTokens 1 it would not run anyway.
BonsaiRuntime.speculativeDecode = false

let loadStart = Date()
let model = try BonsaiModel(path: pack)
let tokens = model.tokenizer.encode(promptText, addSpecialTokens: false)

let warm = Generator(model: model)
_ = warm.generate(promptTokens: Array(tokens.prefix(8)), options: .greedy, maxTokens: 1)
let loadSeconds = -loadStart.timeIntervalSinceNow

GPU.resetPeakMemory()
let generator = Generator(model: model)
let result = generator.generate(promptTokens: tokens, options: .greedy, maxTokens: 1)
let peak = GPU.snapshot().peakMemory

let row: [String: Any] = [
  "prompt_tok": result.stats.promptTokens,
  "prefill_s": (result.stats.promptSeconds * 10000).rounded() / 10000,
  "pp_tok_s": (result.stats.promptTokensPerSecond * 10).rounded() / 10,
  "load_s": (loadSeconds * 1000).rounded() / 1000,
  "peak_gb": (Double(peak) / 1e9 * 1000).rounded() / 1000,
  "prefill_chunk": generator.prefillChunkSize,
  "first_token": result.tokens.first ?? -1,
  "text": model.tokenizer.decode(result.tokens),
]
let data = try JSONSerialization.data(withJSONObject: row, options: [.sortedKeys])
print(String(decoding: data, as: UTF8.self))
