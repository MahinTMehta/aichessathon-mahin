"""Play two versions against each other on a real clock, not a node count.

Every measurement in this engine so far has been node-limited, because that is what makes an A/B
clean: both sides get the same budget, so the result is the change and nothing else. It also
makes the rig blind to the one thing it cannot see — how the engine spends its clock. A change
to time management scores exactly zero there, by construction.

This uses the competition's own referee, at the competition's own control: 120 seconds plus 0.5
a move, both agents started through the platform's runner, the waiting one suspended so it never
steals the mover's core. It is slow — a game is minutes, not seconds — and it is the only test
that sees the clock.

    python3 tools/clockmatch.py <baseline dir> <candidate dir> [--games N] [--start K]

Reports the score, and for both sides the worst clock either ever got down to. A change that
wins on score while dropping the floor toward zero is not a change worth having: flagging is a
whole point, and the reason the reserve exists.
"""

import argparse
import math
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from harness.referee import FAILED_TERMINATIONS, play_match  # noqa: E402
from harness.rules import BASE_MS, INCREMENT_MS  # noqa: E402
from harness.sandbox import local  # noqa: E402

OPENINGS = Path(__file__).resolve().parent / "openings_big.txt"


def interval(wins: int, draws: int, losses: int) -> tuple[float, float, float]:
    games = wins + draws + losses
    score = (wins + 0.5 * draws) / games
    spread = statistics.NormalDist().inv_cdf(0.975) * (
        (wins * (1 - score) ** 2 + draws * (0.5 - score) ** 2 + losses * score**2)
        / max(1, games - 1)
        / games
    ) ** 0.5
    return score, max(0.0, score - spread), min(1.0, score + spread)


def elo(score: float) -> float:
    score = min(max(score, 1e-4), 1.0 - 1e-4)
    return -400.0 * math.log10(1.0 / score - 1.0)


def clock_floor(pgn: str) -> float:
    """The lowest clock either side reached, in seconds, from the referee's own annotations."""
    lowest = 1e9
    for chunk in pgn.split("[%clk ")[1:]:
        value = chunk.split("]")[0]
        try:
            hours, minutes, seconds = value.split(":")
            lowest = min(lowest, int(hours) * 3600 + int(minutes) * 60 + float(seconds))
        except ValueError:
            continue
    return lowest if lowest < 1e9 else float("nan")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--games", type=int, default=40)
    parser.add_argument("--start", type=int, default=0)
    arguments = parser.parse_args()

    openings = [line.strip() for line in OPENINGS.read_text().splitlines() if line.strip()]
    wins = draws = losses = 0
    floors: list[float] = []
    failures: list[str] = []

    for index in range(arguments.start, arguments.start + arguments.games):
        fen = openings[(index // 2) % len(openings)]
        candidate_is_white = index % 2 == 0
        base = local(arguments.baseline.resolve(), index)
        cand = local(arguments.candidate.resolve(), index + 100000)
        white, black = (cand, base) if candidate_is_white else (base, cand)
        outcome = play_match(white, black, BASE_MS, INCREMENT_MS, start_fen=fen)

        if outcome.termination in FAILED_TERMINATIONS:
            failures.append(f"{outcome.termination} as {'white' if candidate_is_white else 'black'}")
        floor = clock_floor(outcome.pgn)
        if floor == floor:  # not NaN
            floors.append(floor)

        if outcome.result == "draw" or outcome.result == "void":
            draws += 1
        elif (outcome.result == "white") == candidate_is_white:
            wins += 1
        else:
            losses += 1

        played = wins + draws + losses
        if played % 2 == 0:
            score, low, high = interval(wins, draws, losses)
            print(
                f"{played:>4}  +{wins} ={draws} -{losses}  score {100 * score:.1f}% "
                f"[{100 * low:.1f}, {100 * high:.1f}]  elo {elo(score):+.0f}  "
                f"worst clock {min(floors) if floors else float('nan'):.1f}s",
                flush=True,
            )

    print()
    score, low, high = interval(wins, draws, losses)
    print(f"{wins + draws + losses} games at {BASE_MS // 1000}s + {INCREMENT_MS / 1000}s")
    print(f"  score {100 * score:.1f}%  [{100 * low:.1f}, {100 * high:.1f}]  elo {elo(score):+.0f}")
    if floors:
        print(f"  clock floor: worst {min(floors):.1f}s, median {statistics.median(floors):.1f}s")
    for failure in failures:
        print(f"  FAILURE {failure}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
