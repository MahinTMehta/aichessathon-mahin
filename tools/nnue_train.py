"""Train a small network to predict what our evaluation gets wrong.

Shape. Two accumulators of `L1` numbers, one per side, built by adding up one column of the
input weights for every piece on the board — the side to move's view first, the other side's
second. Clip both into [0, 1], concatenate, take a dot product. That is the whole network:

    a_us   = b1 + sum(W1[f] for f in features seen from the side to move)
    a_them = b1 + sum(W1[f] for f in features seen from the other side)
    out    = OUT_SCALE * (clip([a_us, a_them], 0, 1) . W2[bucket] + b2[bucket])

`out` is in centipawns and is *added to* our handcrafted evaluation. Fitting the residual rather
than the score is the whole reason this is worth trying with only a few hundred thousand
positions: material and piece placement are already priced, and a network that had to relearn
them from scratch would spend all of its capacity there and arrive back where it started.

Loss is measured where it matters. Two evaluations that differ by 50 centipawns in a position
that is +8 play the same move; two that differ by 50 in a position that is level do not. So the
error is taken after a sigmoid, which is flat at the extremes and steep near zero, exactly like
the difference between the two evaluations matters.

    python3 tools/nnue_train.py <data.npz> <out.npz> [L1] [epochs]
"""

import os
import sys
import time
from pathlib import Path

import numpy as np
from scipy import sparse

# Centipawns per unit of sigmoid slope. 1/300 puts a one-pawn edge at 0.58 and a rook at 0.85,
# which is roughly how much either actually decides a game.
K = 1.0 / 300.0
OUT_SCALE = 100.0
# The correction the engine applies is bounded: a network fitted on positions from real games is
# extrapolating in the positions a search actually spends its time in, and unbounded it produced
# corrections of 786 centipawns there. Bounding it afterwards was worth about 120 elo. Bounding it
# *here*, inside the fit, is strictly better: the network is trained knowing its answer will be
# squashed, so it spends its capacity on the range that survives instead of on numbers that get
# thrown away. Set BOUND to 0 to fit the unbounded network instead.
BOUND = float(os.environ.get("NNUE_BOUND", "128"))
# Output buckets. The same arrangement of pieces means different things with thirty men on the
# board and with eight, and one set of output weights has to average over both. Splitting them by
# how much material is left costs nothing at all to evaluate - only one bucket's weights are ever
# read - and eight times as many output weights is 1,024 numbers against the input layer's 49,152.
BUCKETS = int(os.environ.get("NNUE_BUCKETS", "8"))
# The engine only ever applies the network's answer up to a bound, so positions where the
# handcrafted evaluation is already out by more than that teach it about a range it will never be
# allowed to use. Dropping them aims the fit at the range that survives.
MAX_RESIDUAL = float(os.environ.get("NNUE_MAX_RESIDUAL", "0"))


def bucket_of(sizes: np.ndarray) -> np.ndarray:
    """Which output bucket a position with this many pieces belongs to."""
    return np.clip((sizes.astype(np.int64) - 2) // 4, 0, BUCKETS - 1)
HOLDOUT = 0.08
SEED = 20240
BATCH = 8192


def perspective_indices(feats: np.ndarray, sizes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the White-view and Black-view feature index arrays.

    A feature is (piece colour, piece type, square) packed as `piece * 64 + square`. Seen from
    the other side, the same piece is the opposite colour standing on the vertically mirrored
    square, so the index moves by a fixed amount that depends only on the piece's colour.
    """
    white = feats.astype(np.int32)
    colour = white // 384
    square = white % 64
    black = white + (1 - 2 * colour) * 384 + ((square ^ 56) - square)
    padding = feats < 0
    white = np.where(padding, -1, white)
    black = np.where(padding, -1, black)
    _ = sizes
    return white, black


def design_matrix(indices: np.ndarray) -> sparse.csr_matrix:
    """One row per position, one column per feature, a 1 where the feature is set."""
    rows, columns = np.nonzero(indices >= 0)
    data = np.ones(rows.size, dtype=np.float32)
    return sparse.csr_matrix(
        (data, (rows, indices[rows, columns])), shape=(indices.shape[0], 768), dtype=np.float32
    )


def main() -> int:
    source = Path(sys.argv[1])
    destination = Path(sys.argv[2])
    l1 = int(sys.argv[3]) if len(sys.argv) > 3 else 32
    epochs = int(sys.argv[4]) if len(sys.argv) > 4 else 40

    blob = np.load(source)
    feats, sizes = blob["feats"], blob["sizes"]
    labels, hand, stm = blob["labels"], blob["hand"], blob["stm"]
    count = feats.shape[0]
    print(
        f"{count:,} positions, L1={l1}, {BUCKETS} output buckets, bound {BOUND:.0f}cp",
        flush=True,
    )

    white_view, black_view = perspective_indices(feats, sizes)
    # The network answers from the side to move, so "us" is White's view when White is to move.
    is_white = (stm == 0)[:, None]
    us_view = np.where(is_white, white_view, black_view)
    them_view = np.where(is_white, black_view, white_view)
    del white_view, black_view

    sign = np.where(stm == 0, 1.0, -1.0).astype(np.float32)
    target = (labels * sign).astype(np.float32)  # reference score, side to move's point of view
    base = (hand * sign).astype(np.float32)  # our evaluation, same point of view

    usable = np.arange(count)
    if MAX_RESIDUAL > 0.0:
        usable = usable[np.abs(target - base) <= MAX_RESIDUAL]
        print(
            f"{usable.size:,} of {count:,} positions within {MAX_RESIDUAL:.0f}cp of the "
            f"handcrafted evaluation",
            flush=True,
        )

    us = design_matrix(us_view)
    them = design_matrix(them_view)
    del us_view, them_view

    rng = np.random.default_rng(SEED)
    order = rng.permutation(usable)
    split = int(usable.size * (1.0 - HOLDOUT))
    train_rows, held_rows = order[:split], order[split:]

    W1 = rng.normal(0.0, 0.01, size=(768, l1)).astype(np.float32)
    b1 = np.zeros(l1, dtype=np.float32)
    W2 = rng.normal(0.0, 0.1, size=(BUCKETS, 2 * l1)).astype(np.float32)
    b2 = np.zeros(BUCKETS, dtype=np.float32)
    buckets = bucket_of(sizes)

    parameters = [W1, b1, W2, b2]
    moments = [np.zeros_like(p, dtype=np.float32) for p in parameters]
    velocities = [np.zeros_like(p, dtype=np.float32) for p in parameters]
    beta1, beta2, epsilon = 0.9, 0.999, 1e-8
    step = 0

    def forward_on(sub_us, sub_them, rows, W1, b1, W2, b2):
        a_us = sub_us @ W1 + b1
        a_them = sub_them @ W1 + b1
        hidden = np.concatenate([np.clip(a_us, 0.0, 1.0), np.clip(a_them, 0.0, 1.0)], axis=1)
        chosen = buckets[rows]
        raw = ((hidden * W2[chosen]).sum(axis=1) + b2[chosen]) * OUT_SCALE
        out = BOUND * np.tanh(raw / BOUND) if BOUND > 0.0 else raw
        return a_us, a_them, hidden, out

    def forward(rows: np.ndarray, W1, b1, W2, b2):
        return forward_on(us[rows], them[rows], rows, W1, b1, W2, b2)

    def loss_on(rows: np.ndarray, with_net: bool) -> float:
        total = 0.0
        for start in range(0, rows.size, 65536):
            chunk = rows[start : start + 65536]
            predicted = base[chunk]
            if with_net:
                predicted = predicted + forward(chunk, W1, b1, W2, b2)[3]
            error = _sigmoid(K * predicted) - _sigmoid(K * target[chunk])
            total += float(np.sum(error * error))
        return total / rows.size

    baseline_train = loss_on(train_rows, False)
    baseline_held = loss_on(held_rows, False)
    print(
        f"our evaluation alone: train {baseline_train:.6f}  held-out {baseline_held:.6f}",
        flush=True,
    )

    best_held = baseline_held
    best = (W1.copy(), b1.copy(), W2.copy(), b2.copy())
    stale = 0
    started = time.perf_counter()
    learning_rate = 0.003

    for epoch in range(epochs):
        shuffled = rng.permutation(train_rows)
        for start in range(0, shuffled.size, BATCH):
            rows = shuffled[start : start + BATCH]
            size = rows.size
            sub_us, sub_them = us[rows], them[rows]
            a_us, a_them, hidden, out = forward_on(sub_us, sub_them, rows, W1, b1, W2, b2)
            predicted = base[rows] + out
            sigma = _sigmoid(K * predicted)
            error = sigma - _sigmoid(K * target[rows])
            # d(loss)/d(predicted centipawns), averaged over the batch.
            grad_out = (2.0 * error * K * sigma * (1.0 - sigma) / size).astype(np.float32)
            if BOUND > 0.0:
                # d/draw of BOUND * tanh(raw / BOUND) is 1 - tanh(raw / BOUND) ** 2.
                grad_out = grad_out * (1.0 - (out / BOUND) ** 2)

            scaled = (grad_out * OUT_SCALE).astype(np.float32)
            chosen = buckets[rows]
            grad_hidden = scaled[:, None] * W2[chosen]
            gW2 = np.zeros_like(W2)
            gb2 = np.zeros_like(b2)
            for bucket in range(BUCKETS):
                mask = chosen == bucket
                if not mask.any():
                    continue
                gW2[bucket] = hidden[mask].T @ scaled[mask]
                gb2[bucket] = scaled[mask].sum()

            mask_us = ((a_us > 0.0) & (a_us < 1.0)).astype(np.float32)
            mask_them = ((a_them > 0.0) & (a_them < 1.0)).astype(np.float32)
            grad_us = grad_hidden[:, :l1] * mask_us
            grad_them = grad_hidden[:, l1:] * mask_them

            gW1 = sub_us.T @ grad_us + sub_them.T @ grad_them
            gb1 = grad_us.sum(axis=0) + grad_them.sum(axis=0)

            step += 1
            correction1 = 1.0 - beta1**step
            correction2 = 1.0 - beta2**step
            for index, gradient in enumerate((gW1, gb1, gW2, gb2)):
                moments[index] = beta1 * moments[index] + (1 - beta1) * gradient
                velocities[index] = beta2 * velocities[index] + (1 - beta2) * gradient * gradient
                update = (
                    learning_rate
                    * (moments[index] / correction1)
                    / (np.sqrt(velocities[index] / correction2) + epsilon)
                )
                if index == 0:
                    W1 -= update
                elif index == 1:
                    b1 -= update
                elif index == 2:
                    W2 -= update
                else:
                    b2 -= update

        held = loss_on(held_rows, True)
        train = loss_on(train_rows[:200000], True)
        gain = 100.0 * (1.0 - held / baseline_held)
        print(
            f"epoch {epoch + 1:3d}  train {train:.6f}  held-out {held:.6f}  "
            f"({gain:+.1f}% vs our evaluation)  ({time.perf_counter() - started:.0f}s)",
            flush=True,
        )
        if held < best_held - 1e-9:
            best_held = held
            best = (W1.copy(), b1.copy(), W2.copy(), b2.copy())
            stale = 0
        else:
            stale += 1
            if stale >= 4:
                print("held-out loss stopped improving")
                break
        if stale >= 2:
            learning_rate *= 0.5

    W1, b1, W2, b2 = best[0], best[1], best[2], best[3]
    print(f"\nbest held-out {best_held:.6f} from {baseline_held:.6f}")
    print(f"W1 range [{W1.min():.3f}, {W1.max():.3f}]  W2 range [{W2.min():.3f}, {W2.max():.3f}]")

    # How much of our evaluation's error the network actually removes, in centipawns.
    residual_before = target[held_rows] - base[held_rows]
    predicted = np.concatenate(
        [
            forward(held_rows[start : start + 65536], W1, b1, W2, b2)[3]
            for start in range(0, held_rows.size, 65536)
        ]
    )
    residual_after = residual_before - predicted
    print(
        f"error sd on held-out: {residual_before.std():.1f}cp -> {residual_after.std():.1f}cp"
    )
    np.savez(
        destination,
        W1=W1,
        b1=b1,
        W2=W2,
        b2=b2,
        out_scale=np.float32(OUT_SCALE),
        bound=np.float32(BOUND),
        buckets=np.int32(BUCKETS),
    )
    print(f"saved to {destination}")
    return 0


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


if __name__ == "__main__":
    raise SystemExit(main())
