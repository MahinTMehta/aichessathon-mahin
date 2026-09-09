"""Perft, kept out of the engine's import path so the platform never compiles it on the clock."""

import numpy as np
from numba import int8, int32, int64, njit, uint64

from bitboards import KING, lsb
from position import generate, is_attacked, make_move, unmake_move

@njit(
    int64(uint64[::1], int8[::1], int64[::1], uint64[::1], int64[:, ::1], int32[:, ::1], int64, int64),
    cache=False,
)
def perft(
    bb: np.ndarray,
    mb: np.ndarray,
    st: np.ndarray,
    key: np.ndarray,
    undo: np.ndarray,
    buffers: np.ndarray,
    ply: int,
    depth: int,
) -> int:
    if depth == 0:
        return 1
    count = generate(bb, st, buffers[ply], 0)
    side = st[ST_SIDE]
    total = 0
    for i in range(count):
        move = buffers[ply, i]
        make_move(bb, mb, st, key, undo, ply, move)
        if not is_attacked(bb, lsb(bb[side * 6 + KING]), 1 - side):
            total += perft(bb, mb, st, key, undo, buffers, ply + 1, depth - 1)
        unmake_move(bb, mb, st, key, undo, ply, move)
    return total
