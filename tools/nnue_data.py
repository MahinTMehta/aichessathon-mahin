"""Turn `score|fen` label files into the sparse feature arrays a small network trains on.

The network is the same shape every NNUE uses at its input: one binary feature per
(piece colour, piece type, square), read twice — once from White's point of view and once from
Black's, with the board flipped and the colours swapped. That is what lets one set of weights
serve both sides, and it is why the accumulator for the side to move and the accumulator for the
other side are simply the same weights indexed differently.

Only the White-side indices are stored. The Black-side index of the same piece is a fixed
arithmetic function of it (swap the colour block, flip the rank), so deriving it costs a couple
of integer operations and saves half the file.

Also stored, per position:

    label   the reference engine's score, White's point of view, centipawns
    hand    what our own evaluation says about the same position, same units and point of view
    stm     0 if White is to move

`hand` is there because the network is trained on the *difference*. Our evaluation is already
tuned and already worth its 55 elo; a network that has to rediscover material from scratch would
spend all of its capacity doing that. Fitting the residual means the net starts level with what
we have and only has to learn what our evaluation is missing.

    python3 tools/nnue_data.py <labelled.txt> <out.npz> [limit]
"""

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bitboards import EMPTY, ST_SIDE, WHITE
from evaluate import evaluate
from fen import parse
from position import in_check

MAX_FEATURES = 32
# A static evaluation is asked about quiet positions, so a position where the side to move is in
# check is dropped: what the reference engine thinks of it is about the check.
#
# Decisive positions are kept, and their label is clipped instead. Dropping them was a mistake the
# first time round: the positions where one side has just hung a queen are exactly the ones a
# search spends its time in, and a network that has never been shown one has to guess. The clip
# only stops a forced mate from dominating the fit; the sigmoid the loss goes through is nearly
# flat out there anyway.
SCORE_CAP = 1500


def main() -> int:
    source = Path(sys.argv[1])
    destination = Path(sys.argv[2])
    limit = int(sys.argv[3]) if len(sys.argv) > 3 else 10_000_000

    rows = []
    with source.open() as handle:
        for line in handle:
            head, _, fen = line.partition("|")
            fen = fen.strip()
            if not fen:
                continue
            rows.append((int(head), fen))
            if len(rows) >= limit:
                break
    count = len(rows)
    print(f"{count:,} positions", flush=True)

    feats = np.full((count, MAX_FEATURES), -1, dtype=np.int16)
    sizes = np.zeros(count, dtype=np.int8)
    labels = np.zeros(count, dtype=np.float32)
    hand = np.zeros(count, dtype=np.float32)
    stm = np.zeros(count, dtype=np.int8)

    scratch = np.zeros(16, dtype=np.uint64)
    started = time.perf_counter()
    kept = 0
    skipped_check = 0
    skipped_score = 0
    for i, (score, fen) in enumerate(rows):
        if score > SCORE_CAP:
            score = SCORE_CAP
            skipped_score += 1
        elif score < -SCORE_CAP:
            score = -SCORE_CAP
            skipped_score += 1
        bb, mb, st, _key, _full = parse(fen)
        if in_check(bb, st):
            skipped_check += 1
            continue
        i = kept
        kept += 1
        n = 0
        for square in range(64):
            piece = int(mb[square])
            if piece == EMPTY:
                continue
            feats[i, n] = piece * 64 + square
            n += 1
        sizes[i] = n
        labels[i] = score
        own = int(evaluate(bb, st, scratch))
        # `evaluate` answers from the side to move; the labels are from White's.
        hand[i] = own if st[ST_SIDE] == WHITE else -own
        stm[i] = int(st[ST_SIDE])
        if kept % 50_000 == 0:
            rate = kept / (time.perf_counter() - started)
            print(f"{kept:>8,}  {rate:6.0f}/s", flush=True)

    feats = feats[:kept]
    sizes = sizes[:kept]
    labels = labels[:kept]
    hand = hand[:kept]
    stm = stm[:kept]
    np.savez_compressed(
        destination, feats=feats, sizes=sizes, labels=labels, hand=hand, stm=stm
    )
    residual = labels - hand
    print(
        f"kept {kept:,} of {count:,}  "
        f"(dropped {skipped_check:,} in check, clipped {skipped_score:,} past +-{SCORE_CAP})"
    )
    print(f"saved to {destination}")
    print(
        f"label sd {labels.std():.0f}cp   our error sd {residual.std():.0f}cp   "
        f"mean |error| {np.abs(residual).mean():.0f}cp"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
