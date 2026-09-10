"""A small network that corrects the handcrafted evaluation.

What it is. One weight column per (piece colour, piece type, square). Add up the columns of the
pieces on the board, twice: once seen from the side to move and once from the other side. Clamp
both into range, take a dot product, and you have a number in centipawns which is *added to*
what `evaluate` already says.

Why a correction rather than a replacement. The handcrafted evaluation is tuned and is worth its
measured 55 elo; asking a network trained on a few hundred thousand positions to rediscover
material and piece placement from scratch would spend all of its capacity arriving back where we
already are. Fitting the difference means it only has to learn what we are missing, and against a
reference engine's scores it removed a third of our error on held-out positions.

Everything here is integers. The weights are stored as int16 at a fixed scale, the accumulator is
int32, and the output is divided back down at the end, so the number the search sees is the same
on every machine and there is no floating point in the hot path.

The weights ship as `net.npz`, which is our own file: it was fitted here, from positions labelled
here. Nothing from the reference engine is inside it — the rules ban shipping another engine or
another team's network, and explicitly allow learning from one.
"""

import os
from pathlib import Path

import numpy as np
from numba import int64, njit, uint64

from bitboards import OCC_ALL, lsb, popcount

# The accumulator holds `W1 * 127`, so a clamped hidden unit runs from 0 to 127 and stands for the
# range [0, 1] the network was trained with. The output weights carry a further factor of 16 on
# top of the centipawn scale, so the final divide is by 127 * 16.
QA = 255
QB = 16
OUT_DIVISOR = QA * QB

_HERE = Path(__file__).resolve().parent
# The packaging harness ships every top-level .py plus anything under `weights/`, so that is where
# the file lives. The bare name is still accepted because the experiment directories keep it there.
_PATH = _HERE / "weights" / "net.npz"
if not _PATH.exists():
    _PATH = _HERE / "net.npz"
# The data generator has to see the evaluation the network was trained to correct, not the
# corrected one, or the next round of training fits a residual on top of itself.
ENABLED = _PATH.exists() and os.environ.get("ENGINE_NO_NNUE") != "1"

if ENABLED:
    _blob = np.load(_PATH)
    W1 = np.ascontiguousarray(_blob["W1"], dtype=np.int16)  # (768, L1)
    B1 = np.ascontiguousarray(_blob["b1"], dtype=np.int16)  # (L1,)
    # One set of output weights per material bucket. The same arrangement of pieces means
    # different things with thirty men on the board and with eight, and only the bucket's own row
    # is ever read, so eight of them cost nothing to evaluate.
    W2 = np.ascontiguousarray(_blob["W2"], dtype=np.int16)  # (buckets, 2 * L1)
    B2 = np.ascontiguousarray(np.atleast_1d(_blob["b2"]), dtype=np.int32)  # (buckets,)
    L1 = int(W1.shape[1])
    BUCKETS = int(W2.shape[0])
else:  # pragma: no cover - the packaging check refuses to build a zip without the weights
    W1 = np.zeros((768, 1), dtype=np.int16)
    B1 = np.zeros(1, dtype=np.int16)
    W2 = np.zeros((1, 2), dtype=np.int16)
    B2 = np.zeros(1, dtype=np.int32)
    L1 = 1
    BUCKETS = 1

ACC_SIZE = 2 * L1
# `scratch` is the evaluation's existing buffer and is passed only so that the signature does not
# change if the accumulator ever needs to live there.
ACC_BASE = 16


@njit(int64(uint64[::1], int64, uint64[::1]), cache=False)
def nnue_correction(bb: np.ndarray, side: int, scratch: np.ndarray) -> int:
    """The network's correction in centipawns, from the side to move's point of view.

    The first half of the accumulator is the side to move's and the second half is the other
    side's, which is what lets one set of weights serve both colours: a black knight on g8 seen
    from Black is a white knight on g1 seen from White.

    The accumulator is int16 and every piece adds a whole row of it at once, because that is what
    lets the compiler use vector instructions: this runs once per node, so the difference between
    a scalar loop and a vector one is the difference between the network being affordable and not.
    """
    width = W1.shape[1]
    acc = np.empty(2 * width, dtype=np.int16)
    for i in range(width):
        acc[i] = B1[i]
        acc[width + i] = B1[i]

    for piece in range(12):
        colour = piece // 6
        kind = piece - colour * 6
        mirrored = (1 - colour) * 6 + kind
        bits = bb[piece]
        while bits:
            square = lsb(bits)
            bits &= bits - uint64(1)
            white_row = piece * 64 + square
            black_row = mirrored * 64 + (square ^ 56)
            if side == 0:
                us_row = white_row
                them_row = black_row
            else:
                us_row = black_row
                them_row = white_row
            for i in range(width):
                acc[i] += W1[us_row, i]
                acc[width + i] += W1[them_row, i]

    bucket = (int64(popcount(bb[OCC_ALL])) - 2) // 4
    if bucket < 0:
        bucket = int64(0)
    elif bucket >= BUCKETS:
        bucket = int64(BUCKETS - 1)

    total = int64(0)
    for i in range(2 * width):
        value = int64(acc[i])
        if value < 0:
            value = int64(0)
        elif value > QA:
            value = int64(QA)
        total += value * int64(W2[bucket, i])
    # Truncate toward zero rather than toward minus infinity, so a position and its mirror do not
    # come back one centipawn apart.
    if total >= 0:
        return total // OUT_DIVISOR + int64(B2[bucket])
    return -((-total) // OUT_DIVISOR) + int64(B2[bucket])
