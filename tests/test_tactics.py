"""An absolute check on tactical strength, not a relative one.

Self-play says whether a change beat the last version. It cannot say whether the engine is any
good. These are standard tactical positions with one known winning move; the count solved is a
number that means the same thing from one week to the next.
"""

import sys
import time
from pathlib import Path

import chess

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import agent

# Position, then the move in standard algebraic notation that wins.
CASES = (
    ("2rr3k/pp3pp1/1nnqbN1p/3pN3/2pP4/2P3Q1/PPB4P/R4RK1 w - - 0 1", "Qg6"),
    ("8/7p/5k2/5p2/p1p2P2/Pr1pPK2/1P1R3P/8 b - - 0 1", "Rxb2"),
    ("5rk1/1ppb3p/p1pb4/6q1/3P1p1r/2P1R2P/PP1BQ1P1/5RKN w - - 0 1", "Rg3"),
    ("r1bq2rk/pp3pbp/2p1p1pQ/7P/3P4/2PB1N2/PP3PPR/2KR4 w - - 0 1", "Qxh7+"),
    ("5k2/6pp/p1qN4/1p1p4/3P4/2PKP2Q/PP3r2/3R4 b - - 0 1", "Qc4+"),
    ("7k/p7/1R5K/6r1/6p1/6P1/8/8 w - - 0 1", "Rb7"),
    ("rnbqkb1r/pppp1ppp/8/4P3/6n1/7P/PPPNPnP1/R1BQKBNR b KQkq - 0 1", "Nfg3"),
    ("r4q1k/p2bR1rp/2p2Q1N/5p2/5p2/2P5/PP3PPP/R5K1 w - - 0 1", "Rf7"),
    ("2br2k1/2q3rn/p2NppQ1/2p1P3/Pp5R/4P3/1P3PPP/3R2K1 w - - 0 1", "Rxh7"),
    ("r1b1kb1r/3q1ppp/pBp1pn2/8/Np3P2/5B2/PPP3PP/R2Q1RK1 w kq - 0 1", "Bxc6"),
    ("4k1r1/2p3r1/1pR1p3/3pP2p/3P2qP/P4N2/1PQ4P/5R1K b - - 0 1", "Qxf3+"),
    ("5rk1/pp4p1/2n1p2p/2Npq3/2p5/6P1/P3P1BP/R4Q1K w - - 0 1", "Qxf8+"),
    ("r2rb1k1/pp1q1p1p/2n1p1p1/2bp4/5P2/PP1BPR1Q/1BPN2PP/R5K1 w - - 0 1", "Qxh7+"),
    ("1r3r2/4q1kp/b1pp2p1/5p2/pPn1N3/6P1/P3PPBP/2QRR1K1 w - - 0 1", "Nxd6"),
    ("6k1/p4p2/6p1/1p1bN3/1P1P4/P4nPq/2Q2P2/3R2K1 b - - 0 1", "Qxg3+"),
    ("r3r1k1/ppqb1ppp/8/4p1NQ/8/2P5/PP3PPP/R3R1K1 b - - 0 1", "Bf5"),
    ("2r3k1/pppR1pp1/4p3/4P1P1/5P2/1P4K1/P1P5/8 w - - 0 1", "Rxf7"),
    ("3r1r1k/1p4pp/p4p2/8/1PQR4/6Pq/P3PP2/2R3K1 b - - 0 1", "Rc8"),
    ("2b1r1k1/r4ppp/p3pn2/2Pp4/8/1P1BPN2/P4PPP/2R2RK1 w - - 0 1", "Rxc8"),
    ("6k1/pp1R1nr1/2p1p3/8/4P2p/1P1n1P2/P4KPP/3RB3 w - - 0 1", "Rxf7"),
    ("r1bqk2r/pp3ppp/5n2/8/1b1npB2/2N5/PP1Q1PPP/1K1R1B1R w kq - 0 1", "Nd5"),
    ("r3k2r/pbpn2q1/1p2pp2/8/2PP1p2/1P4P1/PB1N1PBP/R3QRK1 w kq - 0 1", "Nc4"),
    ("1r2r1k1/2p2ppp/p7/1p2P3/3B1P2/3Pq1P1/PPQ4P/2R2R1K b - - 0 1", "Rbd8"),
    ("2r5/2rk2pp/1pn1pb2/pN1p4/P2P4/1N2B3/nPR1KPPP/3R4 b - - 0 1", "Nxd4+"),
    ("8/6pp/3q1p2/3n1k2/1P6/3NQ2P/5PP1/6K1 w - - 0 1", "g4+"),
    ("r1b2rk1/pp2bppp/2n1pn2/q5B1/2BP4/2N2N2/PP2QPPP/2R2RK1 w - - 0 1", "Bxf6"),
    ("2kr3r/pppq1ppp/2n1b3/1B2P3/8/2P2N2/PP3PPP/R1BQ1RK1 w - - 0 1", "Bxc6"),
    ("r4rk1/pp1n1pbp/1qp1p1p1/3pP3/3P1P2/2NB4/PPPQ2PP/2KR3R w - - 0 1", "f5"),
    ("3q1rk1/p4pp1/2pb3p/3p4/6Pr/1PNQ4/P1PB1PP1/4RRK1 b - - 0 1", "Bh2+"),
    ("2r2rk1/pbq2pp1/1p2pn1p/n2p4/2PP4/PP1BPN2/1B2QPPP/R2R2K1 w - - 0 1", "cxd5"),
)


def main() -> int:
    budget = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
    solved = 0
    started = time.perf_counter()
    for fen, expected in CASES:
        board = chess.Board(fen)
        try:
            want = board.parse_san(expected)
        except ValueError:
            print(f"  SKIP unparseable {expected!r} in {fen}")
            continue
        agent._game_length = 0
        got = chess.Move.from_uci(agent.get_move(fen, budget * 28))
        mark = "ok " if got == want else "MISS"
        if got == want:
            solved += 1
        else:
            print(f"  {mark} wanted {expected}, played {board.san(got)}   {fen}")
    elapsed = time.perf_counter() - started
    print(f"\n{solved}/{len(CASES)} solved in {elapsed:.0f}s at {budget}ms a move")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
