"""One engine variant, held open across a whole match.

The competition harness starts a fresh process per game, which is right for a rated game and
useless for tuning: 30 seconds of numba compilation per game means a few hundred games take a
day. This worker imports once and plays until it is told to stop, so a 400-game A/B run costs
one compile per side.

It also searches to a node count rather than a clock, which removes timing noise from the
comparison entirely. Two variants given the same nodes are being compared on the quality of
what they do with them.

Protocol, one JSON object per line on stdin, one per line on stdout:

    {"cmd": "newgame"}                          -> {"ok": true}
    {"cmd": "move", "fen": ..., "nodes": ...}   -> {"move": "e2e4", "score": .., "depth": ..}
    {"cmd": "quit"}
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from bitboards import MAX_MOVES, MAX_PLY
from fen import parse, to_uci
from search import TT_SIZE, search_root

MAX_GAME_PLIES = 1200


class Engine:
    """All the state one side of a match needs, allocated once."""

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
        self.repetition = np.zeros(MAX_GAME_PLIES + MAX_PLY + 8, dtype=np.uint64)
        self.info = np.zeros(16, dtype=np.int64)
        self.escratch = np.zeros(16, dtype=np.uint64)
        self.length = 0

    def new_game(self) -> None:
        self.tt_key[:] = 0
        self.tt_val[:] = 0
        self.killers[:] = 0
        self.history[:] = 0
        self.counters[:] = 0
        self.conthist[:] = 0
        self.repetition[:] = 0
        self.length = 0

    def search(self, fen: str, nodes: int) -> tuple[str, int, int]:
        bb, mb, st, key, _ = parse(fen)
        self.bb[:] = bb
        self.mb[:] = mb
        self.st[:] = st
        self.key[0] = key[0]

        if self.length >= MAX_GAME_PLIES:
            self.length = 0
        self.repetition[self.length] = self.key[0]
        self.info[4] = self.length
        self.length += 1

        self.info[10] = nodes
        far = 1e18
        move = int(
            search_root(
                self.bb, self.mb, self.st, self.key, self.undo, self.moves, self.scores,
                self.tt_key, self.tt_val, self.killers, self.history, self.counters,
                self.conthist, self.stack, self.evals, self.repetition, self.escratch,
                self.info, far, far, MAX_PLY - 16, 0,
            )
        )
        self.info[10] = 0
        return to_uci(move), int(self.info[6]), int(self.info[7])

    def observe(self, fen: str) -> None:
        """Record a position the game passed through without searching it."""
        _, _, _, key, _ = parse(fen)
        if self.length < MAX_GAME_PLIES:
            self.repetition[self.length] = key[0]
            self.length += 1


def main() -> None:
    engine = Engine()
    # One tiny search so numba has compiled everything before the match starts.
    engine.search("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", 20000)
    engine.new_game()
    sys.stdout.write(json.dumps({"ready": True}) + "\n")
    sys.stdout.flush()

    for line in sys.stdin:
        request = json.loads(line)
        command = request["cmd"]
        if command == "quit":
            return
        if command == "newgame":
            engine.new_game()
            reply: dict[str, object] = {"ok": True}
        elif command == "observe":
            engine.observe(request["fen"])
            reply = {"ok": True}
        else:
            move, score, depth = engine.search(request["fen"], request["nodes"])
            reply = {"move": move, "score": score, "depth": depth}
        sys.stdout.write(json.dumps(reply) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
