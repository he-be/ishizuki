// SPDX-FileCopyrightText: 2026 Sarah Truffle <me@heni.lol>
// SPDX-License-Identifier: AGPL-3.0-or-later
//
// A small weight kept in each dtype it is asked for, so a norm stored in fp32 is converted once
// rather than on every layer of every step.

import Foundation
import MLX

final class CastWeight: @unchecked Sendable {
  let base: MLXArray
  private var casts: [DType: MLXArray] = [:]
  private let lock = NSLock()

  init(_ base: MLXArray) {
    self.base = base
  }

  func callAsFunction(_ dtype: DType) -> MLXArray {
    if base.dtype == dtype { return base }
    lock.lock()
    defer { lock.unlock() }
    if let cast = casts[dtype] { return cast }
    let cast = base.asType(dtype)
    casts[dtype] = cast
    return cast
  }
}
