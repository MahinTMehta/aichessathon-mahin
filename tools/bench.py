"""Nodes per second on a fixed set of positions.

The A/B harness gives both sides the same node budget, which is what makes its results clean —
and also what makes it blind to speed. A change that searches twenty percent faster scores
exactly nothing there, while in a real game it buys most of a ply. This measures that half.

Every position is searched to the same node count, so the only thing being compared is how long
those nodes take.
"""

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bitboards import MAX_MOVES, MAX_PLY
from fen import parse
from search import TT_SIZE, search_root

FAR = 1e18

POSITIONS = (
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
    "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
    "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
    "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
    "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10",
    "2rq1rk1/pp3p2/2pbpn1p/1N1pNbp1/3P4/PB2PQBP/1PP2PP1/2R2RK1 b - - 0 16",
    "8/8/4k3/8/2p5/2P5/4K3/8 w - - 0 1",
    "r1bqk2r/pp1n1ppp/2n1p3/2bpP3/5P2/2NB4/PPP3PP/R1BQK1NR w KQkq - 0 8",
    "1rbqk1nr/pp2ppbp/2np2p1/2p5/P3P3/2NP2P1/1PP1NPBP/R1BQK2R b KQk - 0 7",
)


def main() -> int:
    nodes = int(sys.argv[1]) if len(sys.argv) > 1 else 400_000
    bb = np.zeros(15, dtype=np.uint64)
    mb = np.zeros(64, dtype=np.int8)
    st = np.zeros(4, dtype=np.int64)
    key = np.zeros(1, dtype=np.uint64)
    undo = np.zeros((MAX_PLY + 8, 4), dtype=np.int64)
    moves = np.zeros((2 * (MAX_PLY + 8), MAX_MOVES), dtype=np.int32)
    scores = np.zeros((2 * (MAX_PLY + 8), MAX_MOVES), dtype=np.int32)
    tt_key = np.zeros(TT_SIZE, dtype=np.uint64)
    tt_val = np.zeros(TT_SIZE, dtype=np.uint64)
    killers = np.zeros((MAX_PLY + 8, 2), dtype=np.int32)
    history = np.zeros((2, 64, 64), dtype=np.int32)
    counters = np.zeros((12, 64), dtype=np.int32)
    conthist = np.zeros((2, 768, 768), dtype=np.int32)
    stack = np.zeros(MAX_PLY + 8, dtype=np.int32)
    evals = np.zeros(MAX_PLY + 8, dtype=np.int32)
    repetition = np.zeros(2000, dtype=np.uint64)
    escratch = np.zeros(16, dtype=np.uint64)
    info = np.zeros(16, dtype=np.int64)

    total_nodes = 0
    total_time = 0.0
    depths = 0
    for fen in POSITIONS:
        parsed = parse(fen)
        bb[:] = parsed[0]
        mb[:] = parsed[1]
        st[:] = parsed[2]
        key[0] = parsed[3][0]
        tt_key[:] = 0
        tt_val[:] = 0
        killers[:] = 0
        history[:] = 0
        counters[:] = 0
        conthist[:] = 0
        info[4] = 0
        repetition[0] = key[0]
        info[10] = nodes
        started = time.perf_counter()
        search_root(
            bb, mb, st, key, undo, moves, scores, tt_key, tt_val, killers, history, counters,
            conthist, stack, evals, repetition, escratch, info, FAR, FAR, MAX_PLY - 16, 0,
        )
        total_time += time.perf_counter() - started
        info[10] = 0
        total_nodes += int(info[0])
        depths += int(info[7])

    print(
        f"{total_nodes:,} nodes in {total_time:.2f}s = "
        f"{total_nodes / total_time / 1000:.0f}k nodes/s, depth sum {depths}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
