"""Build a varied set of balanced opening positions to start test games from.

Sixteen games over eight openings stop being independent almost immediately, which is what
makes a 16-game result look decisive and mean nothing. These positions are generated instead:
from the initial position, play a dozen plies choosing randomly among moves a shallow search
thinks are reasonable, then keep the position only if it is roughly level and not already a
tactical mess. Diverse starts, neither side favoured.
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


class Probe:
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
        self.repetition = np.zeros(MAX_PLY + 1400, dtype=np.uint64)
        self.info = np.zeros(16, dtype=np.int64)
        self.escratch = np.zeros(16, dtype=np.uint64)

    def evaluate(self, fen: str, nodes: int) -> tuple[str, int]:
        bb, mb, st, key, _ = parse(fen)
        self.bb[:] = bb
        self.mb[:] = mb
        self.st[:] = st
        self.key[0] = key[0]
        self.tt_key[:] = 0
        self.tt_val[:] = 0
        self.info[4] = 0
        self.repetition[0] = self.key[0]
        self.info[10] = nodes
        move = int(
            search_root(
                self.bb, self.mb, self.st, self.key, self.undo, self.moves, self.scores,
                self.tt_key, self.tt_val, self.killers, self.history, self.counters,
                self.conthist, self.stack, self.evals, self.repetition, self.escratch,
                self.info, FAR, FAR, 40, 0,
            )
        )
        self.info[10] = 0
        return to_uci(move), int(self.info[6])


def main() -> None:
    wanted = int(sys.argv[1]) if len(sys.argv) > 1 else 64
    destination = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("tools/openings.txt")
    rng = random.Random(4242)
    probe = Probe()
    kept: list[str] = []
    seen: set[str] = set()

    while len(kept) < wanted:
        board = chess.Board()
        plies = rng.choice((10, 12, 14))
        for _ in range(plies):
            legal = list(board.legal_moves)
            if not legal:
                break
            # Mostly sensible, sometimes not, which is where the variety comes from.
            if rng.random() < 0.65:
                best, _ = probe.evaluate(board.fen(), 12_000)
                move = chess.Move.from_uci(best)
                if move not in board.legal_moves:
                    move = rng.choice(legal)
            else:
                quiet = [m for m in legal if not board.is_capture(m)] or legal
                move = rng.choice(quiet)
            board.push(move)

        if board.is_game_over(claim_draw=False) or board.is_check():
            continue
        fen = board.fen()
        if fen in seen:
            continue
        _, score = probe.evaluate(fen, 60_000)
        if abs(score) > 55:
            continue
        seen.add(fen)
        kept.append(fen)
        print(f"{len(kept):3d}  {score:+4d}  {fen}", flush=True)

    destination.write_text("\n".join(kept) + "\n")
    print(f"\n{len(kept)} positions written to {destination}")


if __name__ == "__main__":
    main()
