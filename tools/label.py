"""Label positions with a reference engine's evaluation.

Why this exists. Every tuning run so far has fitted our evaluation to labels that came from our
own games: the result, or our own search score. Both are weak teachers. The result is extremely
noisy — one position from a game lost for unrelated reasons is told it was bad — and our own
score is worse than noisy, it is *circular*: it teaches the evaluation to agree with itself, so
the fit gets better at predicting our own opinions without ever learning that any of them were
wrong. That is why fitting piece-square tables to it lost 147 elo.

A strong reference engine's score is a real teacher. It carries information our evaluation does
not have, which is the whole point.

The competition rules are explicit that this is allowed: the ban covers what ships inside the
zip, not what you learn from, and labelling positions with an existing engine is called out by
name as a normal way to train. Nothing from the reference engine ends up in the submission — it
never touches the agent, only the weights that are fitted offline and then written down as
ordinary numbers.

    python3 tools/label.py <engine> <positions.txt> <out.txt> [depth] [limit]

Output is `score|fen`, the score in centipawns from White's point of view.
"""

import subprocess
import sys
import time
from pathlib import Path

MATE_SCORE = 2000


def main() -> int:
    engine_path = sys.argv[1]
    source = Path(sys.argv[2])
    destination = Path(sys.argv[3])
    depth = int(sys.argv[4]) if len(sys.argv) > 4 else 8
    limit = int(sys.argv[5]) if len(sys.argv) > 5 else 400_000

    fens = []
    seen = set()
    with source.open() as handle:
        for line in handle:
            parts = line.rsplit("|", 1)
            fen = parts[-1].strip()
            if not fen or fen in seen:
                continue
            seen.add(fen)
            fens.append(fen)
            if len(fens) >= limit:
                break
    print(f"{len(fens):,} distinct positions to label at depth {depth}", flush=True)

    engine = subprocess.Popen(
        [engine_path],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    assert engine.stdin is not None and engine.stdout is not None
    engine.stdin.write("uci\n")
    engine.stdin.flush()
    for line in engine.stdout:
        if line.startswith("uciok"):
            break
    engine.stdin.write("setoption name Threads value 1\nsetoption name Hash value 64\n")
    engine.stdin.flush()

    started = time.perf_counter()
    written = 0
    with destination.open("w") as out:
        for index, fen in enumerate(fens):
            engine.stdin.write(f"position fen {fen}\ngo depth {depth}\n")
            engine.stdin.flush()
            score = None
            for line in engine.stdout:
                if line.startswith("info ") and " score " in line:
                    parts = line.split()
                    at = parts.index("score")
                    if parts[at + 1] == "cp":
                        score = int(parts[at + 2])
                    elif parts[at + 1] == "mate":
                        plies = int(parts[at + 2])
                        score = MATE_SCORE if plies > 0 else -MATE_SCORE
                elif line.startswith("bestmove"):
                    break
            if score is None:
                continue
            # The engine reports from the side to move; store from White's point of view.
            if fen.split()[1] == "b":
                score = -score
            out.write(f"{score}|{fen}\n")
            written += 1
            if (index + 1) % 20000 == 0:
                elapsed = time.perf_counter() - started
                rate = (index + 1) / elapsed
                remaining = (len(fens) - index - 1) / rate / 60
                print(
                    f"{index + 1:>8,}  {rate:6.0f}/s  {remaining:5.1f} min left",
                    flush=True,
                )

    engine.stdin.write("quit\n")
    engine.stdin.flush()
    engine.wait(timeout=10)
    print(f"{written:,} labelled positions written to {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
