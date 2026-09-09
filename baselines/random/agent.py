import os
import random

import chess

RNG = random.Random(os.environ.get("HARNESS_SEED", "0"))


def get_move(fen: str, time_left_ms: int) -> str:
    board = chess.Board(fen)
    return RNG.choice(list(board.legal_moves)).uci()
