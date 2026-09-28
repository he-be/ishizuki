// SPDX-License-Identifier: AGPL-3.0-or-later
// Headless OpenAI-compatible server for ishizuki (struffl/ishizuki), built on the M6 from the
// he-be/ishizuki fork (bench/m6/ishizuki_build_m6.sh) and run there
// (docs/qwen38-27b/16-M6-ISHIZUKI-AGENT-PLAN.md S1). The menu bar app does
// not run on a headless Mac; this is its ServerController.start without the UI.
//
//   ishizuki-serve --pack DIR [--port 8128] [--host 127.0.0.1] [--name ternary-bonsai-2-27b]
//                  [--thinking on|off] [--ctx 32768] [--ceiling-gb N] [--slots 2] [--prefix-dir DIR] [--prefix-gb 8]
//                  [--draft mtp|all|off]
//
// Binds to --host only (bind-host.patch), 127.0.0.1 by default: the API has no
// authentication, so it is reached from the MBP over `ssh -N -L 8128:127.0.0.1:8128`.
// Everything else is the app's own default (ServerSettings.init: KV 3.5 bits / window 128,
// idle 120 s, never unload, politeness normal, prefix store 8 GB, preload, no hot vision
// tower). Sampling is the fixed official set of Tsugumi's `.bonsai27b` for the chosen
// thinking mode (AppModelKind); requests can only override temperature.
// --ctx is llama-server's n_ctx (api.patch): a longer prompt is refused with
// exceed_context_size_error and a reply stops at it; 32K is the plan's operating point (§6-6).
// The memory budget is sized to it as well (the app's is 262,144 tokens), and its ceiling is
// --ceiling-gb, by default ishizuki's own fallback of 75% of RAM (12 GB on the M6). ishizuki
// takes the ceiling from iogpu.wired_limit_mb when that is set, and the M6 keeps it at 14336
// for the reference runtimes (m6-prefill/29): the budget then grew to 13 GB on a 16 GB machine
// and the first Offline set swapped 2.5 GiB (16 §10). --slots pins the number of cached
// conversations, 2 by default; 0 keeps ishizuki's growth (doubling up to 8 as prefixes are
// evicted), which under the 12 GB ceiling still grew to 8 and swapped 644 MiB in 20 s (16 §10).
// --draft picks the drafted decode (mtp.patch): `mtp` (default) drafts one token with the pack's
// MTP head, `all` puts ishizuki's prompt lookup in front of it, `off` is the plain pipelined
// loop. A pack without a head drafts nothing under `mtp` and runs the drafted loop for nothing,
// so `off` is the setting for one. Every setting samples with the official values above.
// SIGTERM / SIGINT call stop(), which archives the prefix caches, then exit 0.

import Foundation
import IshizukiKit

func fail(_ message: String) -> Never {
  FileHandle.standardError.write("ishizuki-serve: \(message)\n".data(using: .utf8)!)
  exit(2)
}

var options: [String: String] = [:]
var rest = CommandLine.arguments.dropFirst()
while let key = rest.popFirst() {
  guard key.hasPrefix("--"), let value = rest.popFirst() else { fail("bad argument \(key)") }
  options[String(key.dropFirst(2))] = value
}
let known: Set = [
  "pack", "port", "host", "name", "thinking", "ctx", "ceiling-gb", "slots", "prefix-dir",
  "prefix-gb", "draft",
]
if let unknown = options.keys.first(where: { !known.contains($0) }) { fail("unknown --\(unknown)") }
guard let packPath = options["pack"] else { fail("--pack is required") }
let pack = URL(fileURLWithPath: (packPath as NSString).expandingTildeInPath)
guard let port = UInt16(options["port"] ?? "8128") else { fail("bad --port") }
let host = options["host"] ?? "127.0.0.1"
let name = options["name"] ?? "ternary-bonsai-2-27b"
let thinking: Bool
switch options["thinking"] ?? "off" {
case "on": thinking = true
case "off": thinking = false
default: fail("--thinking is on or off")
}
let prefixDir =
  options["prefix-dir"].map { URL(fileURLWithPath: ($0 as NSString).expandingTildeInPath) }
  ?? IshizukiPaths.prefixCache
guard let context = Int(options["ctx"] ?? "32768"), context > 0 else { fail("bad --ctx") }
let defaultCeilingGB = Double(ProcessInfo.processInfo.physicalMemory) * 0.75 / 1_073_741_824
guard let ceilingGB = Double(options["ceiling-gb"] ?? "\(defaultCeilingGB)"), ceilingGB > 0 else {
  fail("bad --ceiling-gb")
}
guard let slotCount = Int(options["slots"] ?? "2"), slotCount >= 0 else { fail("bad --slots") }
let slots: Int? = slotCount > 0 ? slotCount : nil
guard let prefixGB = Double(options["prefix-gb"] ?? "8") else { fail("bad --prefix-gb") }
let draft = options["draft"] ?? "mtp"
switch draft {
case "mtp": BonsaiRuntime.lookupMinMatch = Int(Int32.max)
case "all": break
case "off": BonsaiRuntime.speculativeDecode = false
default: fail("--draft is mtp, all or off")
}

// Tsugumi AppModelKind.bonsai27b: thinking off 0.7 / 20 / 0.8 / 0.0 / presence 1.5,
// thinking on 1.0 / 20 / 0.95 / 0.0 / 0.0. Not to be tuned.
let sampling =
  thinking
  ? SamplingOptions(temperature: 1.0, topP: 0.95, topK: 20, minP: 0.0, presencePenalty: 0.0)
  : SamplingOptions(temperature: 0.7, topP: 0.8, topK: 20, minP: 0.0, presencePenalty: 1.5)

@Sendable func log(_ line: String) {
  let stamp = ISO8601DateFormatter().string(from: Date())
  print("\(stamp) \(line)")
  fflush(stdout)
}

let kvConfig = KVCacheConfig()
let server: APIServer
do {
  try kvConfig.validate()
  let budget = MemoryBudget(
    kvBits: kvConfig.bits, maxContextTokens: context,
    weights: StreamedPlan.residentBytes(in: pack) ?? MemoryBudget.defaultWeights,
    ceiling: Int(ceilingGB * 1_073_741_824), slots: slots)
  let start = Date()
  server = try APIServer(
    directory: pack, modelName: name, thinking: thinking, samplingOptions: sampling,
    kvConfig: kvConfig,
    residency: ResidencyManager.Options(wiredBytes: 0, idleSeconds: 120, evictSeconds: 0),
    politeness: .normal, ropeScaling: .none, budget: budget,
    prefixStore: prefixGB > 0
      ? PrefixStore(directory: prefixDir, byteLimit: Int(prefixGB * 1_073_741_824)) : nil,
    preload: true, hot: false)
  log(String(format: "ready: %@ from %@ in %.1f s", name, pack.path, -start.timeIntervalSinceNow))
  server.log = { log($0) }
  server.contextLimit = context
  try server.listen(port: port, host: host)
  log(
    "listening on \(host):\(port), n_ctx \(context), "
      + String(format: "budget ceiling %.1f GB, ", ceilingGB)
      + "slots \(slots.map(String.init) ?? "grow to \(MemoryBudget.slotCeiling)"), thinking \(thinking ? "on" : "off"), "
      + "sampling t \(sampling.temperature) top_k \(sampling.topK) top_p \(sampling.topP) "
      + "min_p \(sampling.minP) presence \(sampling.presencePenalty), draft \(draft), KV \(kvConfig.bits ?? 16) bits, "
      + "prefix store \(prefixGB > 0 ? "\(prefixGB) GB at \(prefixDir.path)" : "off")")
} catch {
  fail("\(error)")
}

var sources: [DispatchSourceSignal] = []
for sig in [SIGTERM, SIGINT] {
  signal(sig, SIG_IGN)
  let source = DispatchSource.makeSignalSource(signal: sig, queue: .main)
  source.setEventHandler {
    log("signal \(sig): stopping")
    server.stop()
    exit(0)
  }
  source.resume()
  sources.append(source)
}
dispatchMain()
