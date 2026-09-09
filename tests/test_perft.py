"""Perft: the movegen either matches these node counts exactly or it is wrong."""

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bitboards import MAX_MOVES, MAX_PLY  # noqa: E402
from fen import parse  # noqa: E402
from perft import perft  # noqa: E402

CASES = (
    ("startpos", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
     (20, 400, 8902, 197281, 4865609)),
    ("kiwipete", "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
     (48, 2039, 97862, 4085603)),
    ("position3", "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
     (14, 191, 2812, 43238, 674624, 11030083)),
    ("position4", "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
     (6, 264, 9467, 422333, 15833292)),
    ("position4-mirror", "r2q1rk1/pP1p2pp/Q4n2/bbp1p3/Np6/1B3NBn/pPPP1PPP/R3K2R b KQ - 0 1",
     (6, 264, 9467, 422333, 15833292)),
    ("position5", "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
     (44, 1486, 62379, 2103487)),
    ("position6", "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10",
     (46, 2079, 89890, 3894594)),
)


def main() -> int:
    undo = np.zeros((MAX_PLY, 4), dtype=np.int64)
    buffers = np.zeros((MAX_PLY, MAX_MOVES), dtype=np.int32)
    failures = 0
    total_nodes = 0
    total_time = 0.0
    for name, fen, expected in CASES:
        bb, mb, st, key, _ = parse(fen)
        for depth, want in enumerate(expected, start=1):
            started = time.perf_counter()
            got = perft(bb, mb, st, key, undo, buffers, 0, depth)
            elapsed = time.perf_counter() - started
            if depth == len(expected):
                total_nodes += got
                total_time += elapsed
            status = "ok" if got == want else f"FAIL want {want}"
            if got != want:
                failures += 1
            print(f"{name:18s} depth {depth}  {got:>10d}  {elapsed:7.3f}s  {status}")
    if total_time > 0:
        print(f"\n{total_nodes / total_time / 1e6:.2f} M nodes/s over the deepest runs")
    print("PERFT FAILURES:" if failures else "all perft counts match", failures or "")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
