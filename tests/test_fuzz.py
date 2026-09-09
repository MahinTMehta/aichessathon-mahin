"""Play a lot of random games and check the engine never lies about a move.

The failures that cost whole games on the platform are rare-path failures: a promotion, an en
passant capture, a stalemate, a position with one legal move. Random play reaches all of them
quickly, and python-chess is the referee, so a disagreement is the engine's fault by definition.
"""

import random
import sys
import time
from pathlib import Path

import chess
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import agent
from evaluate import evaluate
from fen import parse
from position import compute_key

# The evaluation writes its attack maps into a caller-owned buffer.
SCRATCH = np.zeros(16, dtype=np.uint64)


def check_position(board: chess.Board, budget_ms: int) -> tuple[str | None, str]:
    """Return (error or None, the move the engine played)."""
    fen = board.fen()

    bb, _, st, key, _ = parse(fen)
    if key[0] != compute_key(bb, st):
        return f"zobrist mismatch on {fen}", ""

    mirror_bb, _, mirror_st, _, _ = parse(board.mirror().fen())
    left = evaluate(bb, st, SCRATCH)
    right = evaluate(mirror_bb, mirror_st, SCRATCH)
    if left != right:
        return f"evaluation is not colour-symmetric: {left} vs {right} on {fen}", ""

    move = agent.get_move(fen, budget_ms)
    try:
        parsed = chess.Move.from_uci(move)
    except chess.InvalidMoveError:
        return f"unparseable move {move!r} in {fen}", move
    if parsed not in board.legal_moves:
        return f"illegal move {move} in {fen}", move
    return None, move


def main() -> int:
    rng = random.Random(20260908)
    games = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    failures = 0
    positions = 0
    slowest = 0.0
    started = time.perf_counter()

    for game in range(games):
        board = chess.Board()
        # A random opening depth so the sample is not all early middlegames.
        for _ in range(rng.randint(0, 40)):
            legal = list(board.legal_moves)
            if not legal:
                break
            board.push(rng.choice(legal))

        # The engine keeps state between calls, so each game starts from an empty history.
        agent._game_length = 0
        while not board.is_game_over(claim_draw=False) and board.ply() < 220:
            budget = rng.choice((60, 120, 400))
            call_started = time.perf_counter()
            error, move = check_position(board, budget)
            spent = time.perf_counter() - call_started
            slowest = max(slowest, spent)
            positions += 1
            if error is not None:
                print(f"FAIL game {game}: {error}")
                failures += 1
                break
            # Overspending the clock is the other way to lose a game for nothing.
            if spent * 1000.0 > budget + 150:
                print(f"OVERSPENT game {game}: {spent * 1000:.0f}ms of a {budget}ms budget")
                failures += 1
            board.push(chess.Move.from_uci(move))

    elapsed = time.perf_counter() - started
    print(
        f"{games} games, {positions} positions, {elapsed:.1f}s, "
        f"slowest call {slowest * 1000:.0f}ms, {failures} failures"
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
