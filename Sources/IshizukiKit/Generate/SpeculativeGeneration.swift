// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// Drafted decoding for a served turn: prompt lookup when the reply repeats what came before, the
// pack's own MTP head otherwise, checked a block at a time and exact for greedy and sampled picks.

import Foundation
import MLX
import MLXRandom

extension Generator {
  struct Drafts {
    let lookup: Drafter
    let mtp: MTPDrafter?
    var dspark: DSparkDrafter? = nil
    var dflash: DFlashDrafter? = nil

    /// Whether a head of the model's own drafts, which is what the drafted loop is for.
    var hasHead: Bool { mtp != nil || dspark != nil }
  }

  struct DraftedDecode {
    var generated: [Int] = []
    var text = ""
    var stoppedOnEOS = false
    var cancelled = false
    var stats = SpeculativeStats()
  }

  func drafts(
    options: SamplingOptions, constraint: OutputConstraint?, promptEmbeddings: MLXArray?,
    positions: MLXArray?
  ) -> Drafts? {
    // Sampled turns and penalized ones draft with the MTP head too: `speculate` accepts each
    // drafted token with the target's own probability of it and otherwise draws from the rest,
    // with each row's penalty taken over the tokens before it. The pipelined loop's lookup
    // check has no such windows, so a penalized turn without a head still drafts nothing.
    let penalized = options.repetitionPenalty != 1 || options.presencePenalty != 0
    guard speculativeDecode ?? BonsaiRuntime.speculativeDecode, constraint == nil,
      promptEmbeddings == nil, positions == nil, !penalized || model.mtp != nil
    else { return nil }
    let mtp = model.mtp == nil ? nil : try? MTPDrafter(model: model, kvConfig: kvConfig)
    let dflash =
      model.deepseek == nil ? model.dflash.map { DFlashDrafter(draft: $0, backbone: model.text) } : nil
    return Drafts(
      lookup: lookup?() ?? NgramDrafter(minPatternLength: BonsaiRuntime.lookupMinMatch), mtp: mtp,
      dspark: model.deepseek.flatMap { DSparkDrafter(model: $0) }, dflash: dflash)
  }

  func speculate(
    _ drafts: Drafts, logits first: MLXArray, observed start: (MLXArray, [Int])?,
    cache: ModelCache, sampler: Sampler, promptTokens: [Int], maxTokens: Int,
    detokenizer: inout StreamingDetokenizer,
    isCancelled: (@Sendable () -> Bool)?,
    onProgress: ((GenerationProgress) -> Void)?,
    onToken: ((String) -> Bool)?
  ) -> DraftedDecode {
    var out = DraftedDecode()
    var logits = first
    var observed = start
    var forced: Int?
    var context = promptTokens
    var ngramIdle = 0
    let temperature = sampler.options.temperature

    var landed: (hidden: MLXArray?, start: Int) = (nil, 0)
    func forward(_ tokens: [Int], allPositions: Bool) -> (MLXArray, MLXArray) {
      let ids = MLXArray(tokens.map { Int32($0) }).reshaped([1, tokens.count])
      let h: MLXArray
      if let dspark = drafts.dspark {
        let step = dspark.forward(ids, cache: cache)
        landed = (step.hidden, step.start)
        h = step.trunk
      } else {
        h = model.backbone.trunk(inputs: ids, cache: cache)
      }
      let normed = model.backbone.normed(h)
      return (h, allPositions ? model.backbone.logits(normed) : model.backbone.lastLogits(normed))
    }

    func emit(_ tokens: [Int]) -> Bool {
      for token in tokens {
        if model.tokenizer.eosTokenIds.contains(token) {
          out.stoppedOnEOS = true
          return false
        }
        context.append(token)
        out.generated.append(token)
        onProgress?(.decode(count: out.generated.count))
        let fragment = detokenizer.append(token)
        if !fragment.isEmpty {
          out.text += fragment
          if let onToken, !onToken(fragment) { return false }
        }
        if out.generated.count >= maxTokens { return false }
      }
      return true
    }

    // ISHIZUKI_MTP_DRAFT_PENALTY=1: the head drafts the argmax of its logits under the target's
    // bans and penalties. Off by default: on the M6 it lowered the acceptance (320/581 against
    // 339/562 for the raw argmax, 16 §15).
    let penalizedTurn = sampler.options.repetitionPenalty != 1 || sampler.options.presencePenalty != 0
    if let mtp = drafts.mtp, penalizedTurn,
      ProcessInfo.processInfo.environment["ISHIZUKI_MTP_DRAFT_PENALTY"] == "1"
    {
      mtp.scoring = { logits, tokens in sampler.penalizedScores(logits, recentTokens: tokens) }
    }
    defer { drafts.mtp?.scoring = nil }

    var carry: [Int] = []
    // A rejected draft keeps what was accepted in place: the attention layers move their offset
    // back and the delta-net layers rerun the rule over the kept positions from the state the
    // verify started at (`ModelCache.keep`). Restoring the whole block instead carries the kept
    // tokens into the next verify, which on the M6 made one round in two or three a 3-token
    // forward (110 ms against 79 ms for two; 16 §14). DSpark commits its own window, so it keeps
    // the restore.
    let keepsPart =
      drafts.dspark == nil && ProcessInfo.processInfo.environment["ISHIZUKI_SPEC_KEEP"] != "0"
    if keepsPart { cache.recordsSteps = true }
    defer { if keepsPart { cache.recordsSteps = false } }
    // ISHIZUKI_SPEC_TRACE=1: where a drafted round's time goes (ms, summed over the turn).
    let tracing = ProcessInfo.processInfo.environment["ISHIZUKI_SPEC_TRACE"] == "1"
    var spent = (draft: 0.0, forward: 0.0, check: 0.0, rest: 0.0)
    var mark = DispatchTime.now().uptimeNanoseconds
    func lap() -> Double {
      let now = DispatchTime.now().uptimeNanoseconds
      defer { mark = now }
      return Double(now - mark) / 1e6
    }
    defer {
      if tracing {
        print(
          String(
            format: "spec-trace: rounds %d proposed %d accepted %d | ms draft %.0f forward %.0f check %.0f rest %.0f",
            out.stats.rounds, out.stats.proposed, out.stats.accepted, spent.draft, spent.forward,
            spent.check, spent.rest))
        fflush(stdout)
      }
    }

    while out.generated.count < maxTokens {
      if tracing { spent.rest += lap() }
      if isCancelled?() == true {
        out.cancelled = true
        break
      }
      let confirmed =
        forced ?? sampler.token(logits[0..., -1, 0...], recentTokens: context).item(Int.self)
      forced = nil
      if let mtp = drafts.mtp, let observed {
        mtp.observe(hidden: observed.0, nextTokens: observed.1 + [confirmed])
      }
      observed = nil

      var draft: [Int] = []
      let room = BonsaiRuntime.draftLength
      if ngramIdle == 0 {
        draft = drafts.lookup.propose(context: context + [confirmed], count: room)
      } else {
        ngramIdle -= 1
      }
      let fromNgram = !draft.isEmpty
      if draft.isEmpty, let mtp = drafts.mtp {
        draft = mtp.propose(context: context + [confirmed], count: 1)
      }
      if draft.isEmpty, let dspark = drafts.dspark {
        draft = dspark.propose(after: confirmed, at: cache.offset + carry.count, limit: room)
      }
      if carry.count + 1 + draft.count > 16 { draft = [] }
      if tracing { spent.draft += lap() }
      out.stats.rounds += 1

      let offset = carry.count
      let block = carry + [confirmed] + draft
      carry = []

      if draft.isEmpty {
        guard emit([confirmed]) else { break }
        let (h, next) = forward(block, allPositions: false)
        eval(next)
        drafts.dspark?.commit(landed.hidden, start: landed.start, count: block.count)
        logits = next
        observed = (offset == 0 ? h : h[0..., offset..., 0...], [])
        continue
      }

      out.stats.proposed += draft.count
      let snapshot = cache.snapshot()
      let (h, blockLogits) = forward(block, allPositions: true)
      eval(blockLogits)
      if tracing { spent.forward += lap() }

      var accepted = 0
      // Row i predicts what follows block[offset + i], after the draft tokens before it.
      let recent = Array(context.suffix(sampler.options.repetitionContext)) + [confirmed]
      let windows = (0...draft.count).map { recent + draft.prefix($0) }
      let rows = sampler.truncatedScores(rows: blockLogits[0, offset..., 0...], windows: windows)
      if temperature > 0 {
        let scaled = rows / temperature
        let ids = MLXArray(draft.map { Int32($0) })
        let chance = takeAlong(
          softmax(scaled[..<draft.count], axis: -1), ids.reshaped([-1, 1]), axis: -1
        ).reshaped([-1])
        let rolls = MLXRandom.uniform(0 ..< 1, [draft.count])
        let spoiled = scaled[..<draft.count]
        spoiled[MLXArray(Int32(0)..<Int32(draft.count)), ids] = MLXArray(-Float.infinity)
        let replacements = MLXRandom.categorical(spoiled, axis: -1)
        let bonus = MLXRandom.categorical(scaled[draft.count...], axis: -1)
        eval(chance, rolls, replacements, bonus)
        let chances = chance.asArray(Float.self)
        let draws = rolls.asArray(Float.self)
        while accepted < draft.count, draws[accepted] < chances[accepted] { accepted += 1 }
        forced =
          accepted < draft.count
          ? Int(replacements.asArray(Int32.self)[accepted]) : bonus.item(Int.self)
      } else {
        let predictions = rows.argMax(axis: -1).asArray(Int32.self)
        while accepted < draft.count, Int(predictions[accepted]) == draft[accepted] {
          accepted += 1
        }
        forced = Int(predictions[accepted])
      }
      out.stats.accepted += accepted
      if tracing { spent.check += lap() }
      if fromNgram, accepted == 0 { ngramIdle = 4 }

      let kept = [confirmed] + draft.prefix(accepted)
      drafts.dspark?.commit(
        landed.hidden, start: landed.start,
        count: accepted == draft.count ? block.count : offset + kept.count)
      if accepted == draft.count {
        logits = blockLogits[0..., (blockLogits.dim(1) - 1)..., 0...]
        observed = (offset == 0 ? h : h[0..., offset..., 0...], Array(block[(offset + 1)...]))
      } else {
        out.stats.rollbacks += 1
        if keepsPart {
          cache.keep(offset + kept.count, since: snapshot)
        } else {
          cache.restore(snapshot)
          carry = Array(block[..<offset]) + kept
        }
        if let mtp = drafts.mtp, let forced = forced {
          mtp.observe(
            hidden: h[0..., offset..<(offset + kept.count), 0...],
            nextTokens: Array(kept.dropFirst()) + [forced])
        }
      }

      guard emit(kept) else {
        if accepted == draft.count { cache.restore(snapshot) }
        break
      }

      let delay = Politeness.throttleDelay(for: politeness)
      if delay > 0 { Thread.sleep(forTimeInterval: delay) }
    }
    return out
  }
}
