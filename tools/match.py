"""Play two engine variants against each other and say whether the difference is real.

Usage:
    python3 tools/match.py <baseline-dir> <candidate-dir> --games 400 --nodes 120000

Both sides get the same node budget per move, so the result is not a measurement of who
compiled faster. Colours alternate and every opening is played twice, once from each side, so
an opening that favours White cannot flatter either variant.

The output is a score, an elo estimate and a 95% interval. A change is worth keeping when the
interval clears zero, and not before.
"""

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

import chess

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OPENINGS = ROOT / "tools" / "openings_big.txt"
PLY_CAP = 400


class Worker:
    def __init__(self, directory: Path, name: str) -> None:
        self.name = name
        self.process = subprocess.Popen(
            [sys.executable, str(directory / "tools" / "worker.py")],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            bufsize=1,
            cwd=str(directory),
        )
        ready = json.loads(self._readline())
        if not ready.get("ready"):
            raise RuntimeError(f"{name} did not come up")

    def _readline(self) -> str:
        assert self.process.stdout is not None
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError(f"{self.name} died")
        return line

    def send(self, request: dict[str, object]) -> dict[str, object]:
        assert self.process.stdin is not None
        self.process.stdin.write(json.dumps(request) + "\n")
        self.process.stdin.flush()
        return json.loads(self._readline())

    def close(self) -> None:
        try:
            assert self.process.stdin is not None
            self.process.stdin.write(json.dumps({"cmd": "quit"}) + "\n")
            self.process.stdin.flush()
            self.process.wait(timeout=5)
        except Exception:
            self.process.kill()


def play_game(
    white: Worker, black: Worker, fen: str, nodes: int
) -> tuple[str, str]:
    board = chess.Board(fen)
    for worker in (white, black):
        worker.send({"cmd": "newgame"})
        worker.send({"cmd": "observe", "fen": fen})

    while True:
        outcome = board.outcome(claim_draw=False)
        if outcome is not None:
            if outcome.winner is None:
                return "draw", outcome.termination.name.lower()
            return ("white" if outcome.winner else "black"), outcome.termination.name.lower()
        if board.is_repetition(3):
            return "draw", "threefold"
        if board.is_fifty_moves():
            return "draw", "fifty"
        if board.ply() >= PLY_CAP:
            return "draw", "ply_cap"

        mover = white if board.turn == chess.WHITE else black
        waiter = black if board.turn == chess.WHITE else white
        reply = mover.send({"cmd": "move", "fen": board.fen(), "nodes": nodes})
        uci = str(reply["move"])
        try:
            move = chess.Move.from_uci(uci)
        except chess.InvalidMoveError:
            return ("black" if board.turn == chess.WHITE else "white"), f"illegal {uci}"
        if move not in board.legal_moves:
            return ("black" if board.turn == chess.WHITE else "white"), f"illegal {uci}"
        board.push(move)
        # The side that did not move still has to see the position for its repetition history.
        waiter.send({"cmd": "observe", "fen": board.fen()})
        mover.send({"cmd": "observe", "fen": board.fen()})


def elo(score: float) -> float:
    if score <= 0.0:
        return -800.0
    if score >= 1.0:
        return 800.0
    return -400.0 * math.log10(1.0 / score - 1.0)


def report(wins: int, draws: int, losses: int) -> str:
    games = wins + draws + losses
    if games == 0:
        return "no games"
    score = (wins + 0.5 * draws) / games
    # Variance of the per-game score, which is what the interval should be built from.
    mean_square = (wins + 0.25 * draws) / games
    variance = max(mean_square - score * score, 1e-12)
    margin = 1.96 * math.sqrt(variance / games)
    low, high = max(score - margin, 0.0), min(score + margin, 1.0)
    verdict = "REAL" if low > 0.5 else ("WORSE" if high < 0.5 else "not separated")
    return (
        f"+{wins} ={draws} -{losses}  score {score * 100:.1f}% "
        f"[{low * 100:.1f}, {high * 100:.1f}]  "
        f"elo {elo(score):+.0f} [{elo(low):+.0f}, {elo(high):+.0f}]  {verdict}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--games", type=int, default=200)
    parser.add_argument("--nodes", type=int, default=120_000)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--openings", type=Path, default=DEFAULT_OPENINGS)
    arguments = parser.parse_args()

    openings = [
        line.strip() for line in arguments.openings.read_text().splitlines() if line.strip()
    ]
    base = Worker(arguments.baseline.resolve(), "baseline")
    cand = Worker(arguments.candidate.resolve(), "candidate")

    wins = draws = losses = 0
    try:
        for index in range(arguments.start, arguments.start + arguments.games):
            fen = openings[(index // 2) % len(openings)]
            candidate_is_white = index % 2 == 0
            white, black = (cand, base) if candidate_is_white else (base, cand)
            result, termination = play_game(white, black, fen, arguments.nodes)
            if result == "draw":
                draws += 1
            elif (result == "white") == candidate_is_white:
                wins += 1
            else:
                losses += 1
            if "illegal" in termination:
                print(f"  !! {termination} in game {index}: {fen}")
            if (index - arguments.start + 1) % 10 == 0:
                played = index - arguments.start + 1
                print(f"{played:4d}  {report(wins, draws, losses)}", flush=True)
    finally:
        base.close()
        cand.close()

    print("\nFINAL " + report(wins, draws, losses))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
