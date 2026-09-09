"""Generate labelled positions for the evaluation tuner.

The engine plays itself from the generated openings at a low node count. Every quiet position
along the way is written out with the result the game eventually reached. That pairing —
"this position, and the side to move went on to score this" — is the whole training signal: a
weight is better if it makes the evaluation agree with outcomes more often.

Positions where the side to move is in check, or where the move played was a capture, are
skipped: a static evaluation has nothing useful to say about a position in the middle of an
exchange, and including them teaches the tuner to fit noise.
"""

import random
import sys
from pathlib import Path

import chess
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bitboards import MAX_MOVES, MAX_PLY
from fen import parse, to_uci
from search import TT_SIZE, search_root

FAR = 1e18
PLY_CAP = 300
OPENINGS = Path(__file__).resolve().parent / "openings_big.txt"


class Player:
    def __init__(self) -> None:
        self.bb = np.zeros(15, dtype=np.uint64)
        self.mb = np.zeros(64, dtype=np.int8)
        self.st = np.zeros(4, dtype=np.int64)
        self.key = np.zeros(1, dtype=np.uint64)
        self.undo = np.zeros((MAX_PLY + 8, 4), dtype=np.int64)
        self.moves = np.zeros((2 * (MAX_PLY + 8), MAX_MOVES), dtype=np.int32)
        self.scores = np.zeros((2 * (MAX_PLY + 8), MAX_MOVES), dtype=np.int32)
        self.tt_key = np.zeros(TT_SIZE, dtype=np.uint64)
        self.tt_val = np.zeros(TT_SIZE, dtype=np.uint64)
        self.killers = np.zeros((MAX_PLY + 8, 2), dtype=np.int32)
        self.history = np.zeros((2, 64, 64), dtype=np.int32)
        self.counters = np.zeros((12, 64), dtype=np.int32)
        self.conthist = np.zeros((2, 768, 768), dtype=np.int32)
        self.stack = np.zeros(MAX_PLY + 8, dtype=np.int32)
        self.evals = np.zeros(MAX_PLY + 8, dtype=np.int32)
        self.repetition = np.zeros(2000, dtype=np.uint64)
        self.escratch = np.zeros(16, dtype=np.uint64)
        self.info = np.zeros(16, dtype=np.int64)
        self.length = 0

    def new_game(self) -> None:
        self.tt_key[:] = 0
        self.tt_val[:] = 0
        self.killers[:] = 0
        self.history[:] = 0
        self.counters[:] = 0
        self.conthist[:] = 0
        self.length = 0

    def best(self, fen: str, nodes: int) -> tuple[str, int]:
        bb, mb, st, key, _ = parse(fen)
        self.bb[:] = bb
        self.mb[:] = mb
        self.st[:] = st
        self.key[0] = key[0]
        self.repetition[self.length] = self.key[0]
        self.info[4] = self.length
        self.length += 1
        self.info[10] = nodes
        move = int(
            search_root(
                self.bb, self.mb, self.st, self.key, self.undo, self.moves, self.scores,
                self.tt_key, self.tt_val, self.killers, self.history, self.counters,
                self.conthist, self.stack, self.evals, self.repetition, self.escratch,
                self.info, FAR, FAR, MAX_PLY - 16, 0,
            )
        )
        self.info[10] = 0
        return to_uci(move), int(self.info[6])

    def observe(self, fen: str) -> None:
        _, _, _, key, _ = parse(fen)
        self.repetition[self.length] = key[0]
        self.length += 1


def main() -> None:
    games = int(sys.argv[1]) if len(sys.argv) > 1 else 800
    nodes = int(sys.argv[2]) if len(sys.argv) > 2 else 5000
    destination = Path(sys.argv[3]) if len(sys.argv) > 3 else Path("tools/positions.txt")
    seed = int(sys.argv[4]) if len(sys.argv) > 4 else 1

    openings = [line.strip() for line in OPENINGS.read_text().splitlines() if line.strip()]
    rng = random.Random(seed)
    player = Player()
    written = 0

    with destination.open("w") as handle:
        for game in range(games):
            board = chess.Board(openings[rng.randrange(len(openings))])
            # A couple of random moves so the same opening does not give the same game twice.
            for _ in range(rng.randint(0, 3)):
                legal = list(board.legal_moves)
                if not legal:
                    break
                board.push(rng.choice(legal))
            if board.is_game_over(claim_draw=False):
                continue

            player.new_game()
            player.observe(board.fen())
            player.length -= 1
            samples: list[tuple[str, int]] = []

            while not board.is_game_over(claim_draw=False) and board.ply() < PLY_CAP:
                if board.is_repetition(3) or board.is_fifty_moves():
                    break
                fen = board.fen()
                uci, score = player.best(fen, nodes)
                if uci == "a1a1":
                    print(
                        f"NULLMOVE game={game} ply={board.ply()} length={player.length} "
                        f"info={list(int(x) for x in player.info)} fen={fen}",
                        flush=True,
                    )
                    break
                move = chess.Move.from_uci(uci)
                if move not in board.legal_moves:
                    print(f"ILLEGAL {uci} in {fen}", flush=True)
                    break
                quiet = not board.is_capture(move) and move.promotion is None
                # A position the search already calls decided teaches the evaluation nothing
                # except to shout, so leave those out.
                if (
                    quiet
                    and not board.is_check()
                    and board.ply() >= 6
                    and abs(score) < 900
                ):
                    white_score = score if board.turn == chess.WHITE else -score
                    samples.append((fen, white_score))
                board.push(move)
                player.observe(board.fen())

            outcome = board.outcome(claim_draw=False)
            if outcome is None or outcome.winner is None:
                result = 0.5
            else:
                result = 1.0 if outcome.winner == chess.WHITE else 0.0

            for fen, white_score in samples:
                # Result, the search's own verdict, then the position. The tuner blends the
                # first two: the result is what matters and the score is far less noisy.
                handle.write(f"{result}|{white_score}|{fen}\n")
                written += 1
            if (game + 1) % 25 == 0:
                print(f"{game + 1} games, {written} positions", flush=True)

    print(f"{written} positions written to {destination}")


if __name__ == "__main__":
    main()
