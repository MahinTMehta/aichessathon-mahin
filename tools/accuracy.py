"""How much a version of the engine gives away, in centipawns, per move it plays.

Elo from self-play says which of two versions wins more games against the other. It does not say
whether either of them hangs pieces, and it cannot: both sides make the same kind of mistake, and
the mistakes cancel. A version can be 60 elo better and still throw a game away in a way the
opponent it was tested against never punished.

This measures the thing directly. For each position: what does the engine play, what does a
reference engine at high depth think the position was worth before, and what does it think it is
worth after. The difference is what the move cost. Averaged, that is the accuracy figure human
chess uses; counted at thresholds, it is a blunder rate.

    python3 tools/accuracy.py <engine dir> <positions file> [count] [nodes] [ref depth]

Two numbers matter and they say different things: the mean tells you how well it plays, and the
tail — how often it drops 200 or 300 centipawns in one move — tells you how often it loses a game
it should not have. A change that improves the first and worsens the second is not an improvement
for a tournament that is decided by a handful of losses.
"""

import subprocess
import sys
import time
from pathlib import Path

import chess

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.reference import MATE, Reference, find  # noqa: E402

class Engine:
    """The engine under test, one move at a time from unrelated positions.

    It runs through `tools/worker.py`, the same process the match harness drives, and gets a
    `newgame` before every position: the engine keeps a table and a repetition history between
    moves of a game, and feeding it unrelated positions in sequence would let one position's
    history claim a repetition in the next. The budget is in nodes rather than milliseconds so
    the measurement does not move when the machine is busy.
    """

    def __init__(self, root: Path) -> None:
        self.process = subprocess.Popen(
            [sys.executable, str(root / "tools" / "worker.py")],
            cwd=str(root),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )
        assert self.process.stdout is not None
        self.process.stdout.readline()

    def move(self, fen: str, nodes: int) -> str:
        import json

        assert self.process.stdin is not None and self.process.stdout is not None
        self.process.stdin.write(json.dumps({"cmd": "newgame"}) + "\n")
        self.process.stdin.flush()
        self.process.stdout.readline()
        self.process.stdin.write(
            json.dumps({"cmd": "move", "fen": fen, "nodes": nodes}) + "\n"
        )
        self.process.stdin.flush()
        return json.loads(self.process.stdout.readline())["move"]

    def close(self) -> None:
        try:
            assert self.process.stdin is not None
            self.process.stdin.write('{"cmd": "quit"}\n')
            self.process.stdin.flush()
            self.process.wait(timeout=5)
        except Exception:
            self.process.kill()


def load_positions(path: Path, count: int) -> list[str]:
    """One FEN per line, or `something|FEN`, taking every nth so the sample is spread out."""
    lines = [line.rsplit("|", 1)[-1].strip() for line in path.read_text().splitlines()]
    lines = [line for line in lines if line]
    step = max(1, len(lines) // count)
    return lines[::step][:count]


def main() -> int:
    root = Path(sys.argv[1]).resolve()
    positions = load_positions(Path(sys.argv[2]), int(sys.argv[3]) if len(sys.argv) > 3 else 300)
    nodes = int(sys.argv[4]) if len(sys.argv) > 4 else 1_500_000
    depth = int(sys.argv[5]) if len(sys.argv) > 5 else 16

    engine = Engine(root)
    reference = Reference(find())
    losses: list[tuple[float, str, str, str]] = []
    started = time.perf_counter()

    for index, fen in enumerate(positions):
        board = chess.Board(fen)
        if board.is_game_over(claim_draw=False):
            continue
        # What the position is worth to the side to move, before it plays.
        best_uci, before = reference.go(fen, depth=depth)
        played = engine.move(fen, nodes)
        try:
            move = chess.Move.from_uci(played)
        except chess.InvalidMoveError:
            losses.append((9999.0, fen, played, best_uci))
            continue
        if move not in board.legal_moves:
            losses.append((9999.0, fen, played, best_uci))
            continue
        board.push(move)
        if board.is_checkmate():
            after = MATE
        elif board.is_stalemate() or board.is_insufficient_material():
            after = 0
        else:
            # The reference now answers from the opponent's point of view, so flip it back.
            _, opponent = reference.go(board.fen(), depth=depth)
            after = -opponent
        # Mate scores would swamp the average; what matters there is only that we did not miss it.
        cost = max(0.0, min(1000.0, float(before - after)))
        losses.append((cost, fen, played, best_uci))

        if (index + 1) % 25 == 0:
            done = [loss for loss, _, _, _ in losses]
            rate = (index + 1) / (time.perf_counter() - started)
            print(
                f"{index + 1:>5}/{len(positions)}  mean {sum(done) / len(done):6.1f}cp  "
                f"{rate:4.1f} pos/s",
                flush=True,
            )

    engine.close()
    reference.close()

    costs = [loss for loss, _, _, _ in losses]
    if not costs:
        print("no positions measured")
        return 1
    illegal = sum(1 for loss in costs if loss >= 9999.0)
    clean = [loss for loss in costs if loss < 9999.0]
    print()
    print(f"{root.name}: {len(clean)} moves at {nodes:,} nodes, reference depth {depth}")
    print(f"  mean centipawn loss     {sum(clean) / len(clean):8.1f}")
    print(f"  median                  {sorted(clean)[len(clean) // 2]:8.1f}")
    for threshold in (50, 100, 200, 300, 500):
        share = 100.0 * sum(1 for loss in clean if loss >= threshold) / len(clean)
        print(f"  moves losing {threshold:>3}cp+      {share:7.2f}%")
    if illegal:
        print(f"  ILLEGAL OR NO MOVE      {illegal}")

    worst = sorted(losses, reverse=True)[:8]
    print("\n  worst moves")
    for cost, fen, played, best in worst:
        print(f"    -{cost:4.0f}cp  played {played:6s} best {best:6s}  {fen}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
