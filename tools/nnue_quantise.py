"""Turn the trained floating-point network into the int16 file the engine ships.

Two things have to be true afterwards and both are checked here rather than assumed:

  * the integer arithmetic agrees with the float model it came from, to within a centipawn or
    two, over real positions rather than random noise; and
  * nothing overflows the types it is stored in.

A quantisation that is quietly wrong produces an engine that still runs, still plays legal
chess, and is weaker for no visible reason, which is the most expensive kind of bug there is.

    python3 tools/nnue_quantise.py <net.npz float> <net.npz int> <data.npz>
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.nnue_train import bucket_of, design_matrix, perspective_indices  # noqa: E402

QA = 255
QB = 16
# The engine squashes the raw output through a table rather than computing a tanh, so the two
# agree exactly and the hot path stays integer. Raw output past four times the bound is already
# saturated, so a table covering +-2048 centipawns is the whole function; with no bound fitted the
# table is the identity and the engine's own limit does the bounding.
TABLE_HALF = 2048


def main() -> int:
    source = Path(sys.argv[1])
    destination = Path(sys.argv[2])
    data = Path(sys.argv[3])

    net = np.load(source)
    W1, b1, W2, b2 = net["W1"], net["b1"], net["W2"], np.atleast_1d(net["b2"])
    out_scale = float(net["out_scale"])
    bound = float(net["bound"]) if "bound" in net.files else 0.0

    qW1 = np.rint(W1 * QA).astype(np.int32)
    qb1 = np.rint(b1 * QA).astype(np.int32)
    qW2 = np.rint(W2 * out_scale * QB).astype(np.int32)
    qb2 = np.rint(np.atleast_1d(b2) * out_scale).astype(np.int32)

    if np.abs(qW1).max() > 32767 or np.abs(qW2).max() > 32767:
        print(f"overflow: W1 {np.abs(qW1).max()}, W2 {np.abs(qW2).max()}")
        return 1
    # The accumulator sums at most 32 rows plus the bias, and is held in int32 in the engine.
    worst = int(np.abs(qb1).max() + 32 * np.abs(qW1).max())
    print(f"largest possible accumulator {worst:,} (int32 holds 2,147,483,647)")

    blob = np.load(data)
    feats, sizes = blob["feats"][:20000], blob["sizes"][:20000]
    stm = blob["stm"][:20000]
    # The net decides how many buckets there are, not the environment the trainer ran in.
    chosen = np.clip(bucket_of(blob["sizes"][:20000]), 0, W2.shape[0] - 1)
    white_view, black_view = perspective_indices(feats, sizes)
    is_white = (stm == 0)[:, None]
    us = design_matrix(np.where(is_white, white_view, black_view))
    them = design_matrix(np.where(is_white, black_view, white_view))

    raw_float = ((
        np.concatenate(
            [
                np.clip(us @ W1 + b1, 0.0, 1.0),
                np.clip(them @ W1 + b1, 0.0, 1.0),
            ],
            axis=1,
        )
        * W2[chosen]
    ).sum(axis=1) + b2[chosen]) * out_scale
    float_out = bound * np.tanh(raw_float / bound) if bound > 0.0 else raw_float

    raw = np.arange(-TABLE_HALF, TABLE_HALF, dtype=np.float64)
    if bound > 0.0:
        table = np.rint(bound * np.tanh(raw / bound)).astype(np.int16)
    else:
        table = np.clip(raw, -32768, 32767).astype(np.int16)

    acc_us = np.clip(us @ qW1 + qb1, 0, QA).astype(np.int64)
    acc_them = np.clip(them @ qW1 + qb1, 0, QA).astype(np.int64)
    total = (
        np.concatenate([acc_us, acc_them], axis=1) * qW2.astype(np.int64)[chosen]
    ).sum(axis=1)
    integer_raw = np.trunc(total / (QA * QB)).astype(np.int64) + qb2[chosen]
    integer_out = table[
        np.clip(integer_raw + TABLE_HALF, 0, 2 * TABLE_HALF - 1)
    ].astype(np.int64)

    difference = integer_out - float_out
    print(
        f"quantisation error over {len(difference):,} positions: "
        f"mean {difference.mean():+.2f}cp  sd {difference.std():.2f}cp  "
        f"worst {np.abs(difference).max():.1f}cp"
    )

    np.savez(
        destination,
        W1=qW1.astype(np.int16),
        b1=qb1.astype(np.int32),
        W2=qW2.astype(np.int16),
        b2=qb2.astype(np.int32),
        table=table,
    )
    print(f"bound {bound:.0f}cp, table {table.min()} to {table.max()}")
    size = destination.stat().st_size
    print(
        f"saved {destination} ({size / 1024:.0f} KB), L1={W1.shape[1]}, "
        f"{qW2.shape[0]} output buckets"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
