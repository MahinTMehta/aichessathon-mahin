"""Check the root move list against python-chess, with en passant specifically in mind.

Perft starts from a handful of fixed positions and exercises en passant only through moves the
engine itself makes, where make_move sets the square. It never checks the other path: a FEN
handed to us with an en passant square already on it, which is every position the platform ever
asks about. A bug there is invisible to perft and loses a game the first time en passant is the
only legal move.
"""

import random
import sys
from pathlib import Path

import chess
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bitboards import KING, MAX_MOVES, MAX_PLY, ST_SIDE, lsb
from fen import parse, to_uci
from position import generate, is_attacked, make_move, unmake_move


def legal_moves(fen: str) -> set[str]:
    bb, mb, st, key, _ = parse(fen)
    undo = np.zeros((MAX_PLY + 8, 4), dtype=np.int64)
    buffer = np.zeros(MAX_MOVES, dtype=np.int32)
    count = generate(bb, st, buffer, 0)
    side = int(st[ST_SIDE])
    found = set()
    for i in range(count):
        move = buffer[i]
        make_move(bb, mb, st, key, undo, 0, move)
        if not is_attacked(bb, lsb(bb[side * 6 + KING]), 1 - side):
            found.add(to_uci(int(move)))
        unmake_move(bb, mb, st, key, undo, 0, move)
    return found


def main() -> int:
    rng = random.Random(31337)
    failures = 0
    checked = 0
    with_ep = 0

    # Positions where en passant is the only legal move, which is the case that turns a missed
    # move into a forfeit.
    forced = (
        "8/8/8/8/k1p4R/8/3P4/3K4 w - - 0 1",
        "8/5k2/8/2Pp4/2B5/1K6/8/8 w - d6 0 1",
        "8/8/8/1Ppp3r/RK3p1k/8/4P1P1/8 w - c6 0 1",
    )

    boards = [chess.Board(fen) for fen in forced]
    for _ in range(4000):
        board = chess.Board()
        for _ in range(rng.randint(0, 60)):
            legal = list(board.legal_moves)
            if not legal:
                break
            board.push(rng.choice(legal))
            if board.ep_square is not None:
                boards.append(board.copy())
                break

    for board in boards:
        if board.is_game_over(claim_draw=False):
            continue
        checked += 1
        if board.ep_square is not None:
            with_ep += 1
        expected = {move.uci() for move in board.legal_moves}
        got = legal_moves(board.fen())
        if expected != got:
            failures += 1
            if failures <= 5:
                print(f"MISMATCH {board.fen()}")
                print(f"  missing {sorted(expected - got)}  extra {sorted(got - expected)}")

    print(f"{checked} positions ({with_ep} with an en passant square), {failures} mismatches")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
