"""Widen the training set to the positions a search actually evaluates.

Every position we have been training on came from a game: our engine chose both sides' moves, so
every one of them is a position a reasonable player would allow. A search does not spend its time
there. It spends it in the branches it is about to refute — a queen left hanging for a ply, a king
walked into the open, a piece sacrificed for nothing — and those are the positions the evaluation
is being asked about a million times a move.

Trained only on game positions, the network was excellent where it had seen things and an
extrapolation where it had not: on held-out game positions it cut our evaluation's error by a
third, and in actual games it lost 81 elo. The corrections it produced reached 786 centipawns,
which is not a correction, it is a guess.

So: take each position we have, play a handful of uniformly random legal moves from it, and keep
where that lands. Random play is not what a search does either, but it covers the same ground —
material imbalances, exposed kings, pieces on squares no plan would put them — and it costs
nothing to produce. What matters is that the network stops meeting those positions for the first
time in the middle of a game.

    python3 tools/make_forks.py <positions.txt> <out.txt> [per-position] [seed]

Output is one FEN per line, ready for tools/label.py.
"""

import random
import sys
from pathlib import Path

import chess

# Deep enough to leave the neighbourhood of a sane game, shallow enough that the position is still
# recognisably chess rather than a random arrangement of pieces.
MIN_PLIES = 1
MAX_PLIES = 5


def main() -> int:
    source = Path(sys.argv[1])
    destination = Path(sys.argv[2])
    per_position = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    rng = random.Random(int(sys.argv[4]) if len(sys.argv) > 4 else 7)

    seen: set[str] = set()
    written = 0
    read = 0
    with source.open() as handle, destination.open("w") as out:
        for line in handle:
            fen = line.rsplit("|", 1)[-1].strip()
            if not fen:
                continue
            read += 1
            try:
                start = chess.Board(fen)
            except ValueError:
                continue
            for _ in range(per_position):
                board = start.copy(stack=False)
                for _ in range(rng.randint(MIN_PLIES, MAX_PLIES)):
                    legal = list(board.legal_moves)
                    if not legal:
                        break
                    board.push(rng.choice(legal))
                if board.is_game_over(claim_draw=False) or board.is_check():
                    continue
                # A static evaluation is asked about quiet positions, and the reference engine's
                # opinion of a position in the middle of an exchange is about the exchange.
                key = board.board_fen() + " " + ("w" if board.turn else "b")
                if key in seen:
                    continue
                seen.add(key)
                out.write(board.fen() + "\n")
                written += 1
            if read % 100_000 == 0:
                print(f"{read:,} read, {written:,} written", flush=True)

    print(f"{written:,} positions from {read:,} starting points -> {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
