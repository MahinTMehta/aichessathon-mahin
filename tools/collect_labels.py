"""Label positions with Stockfish, using every core on this machine.

Why this is the thing to run. The evaluation is now a handcrafted one plus a small network that
corrects it, and what the network is worth is decided almost entirely by how many positions it
was fitted to. It has seen 600,000. On held-out positions it ranks pairs the way Stockfish does
92% of the time against the handcrafted evaluation's 87%, and that difference is worth about 70
elo in games — the largest single gain in the engine after king safety. More positions is the
straightforward way to push it further, and labelling is a Stockfish search per position, so it
is the one job that scales with cores.

One thing was tried and did not work, so it is not in here: making extra positions by playing
random moves from the ones we have. It doubled the data and cost 21 elo, because it moved the
fit away from the positions that actually occur. Positions from real games are what counts.

    uv run python tools/collect_labels.py

It runs two stages: self-play to make positions (skipped if `tools/training_data.txt.gz` already
has some, unless you pass `--more`), then Stockfish over all of them. Stop it whenever — what is
finished is kept — and send back `tools/labelled_data.txt.gz`.

Stockfish has to be findable: on PATH, at `$SF`, or as `tools/stockfish.exe`. It is not
downloaded for you; https://stockfishchess.org/download/ has the Windows AVX2 build.
"""

import gzip
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SHARDS = ROOT / "tools" / "label_shards"
POSITIONS = ROOT / "tools" / "training_data.txt.gz"
FLAT = ROOT / "tools" / "all_positions.txt"
OUTPUT = ROOT / "tools" / "labelled_data.txt.gz"
DEPTH = 8


def find_engine() -> str | None:
    candidate = os.environ.get("SF")
    if candidate and Path(candidate).exists():
        return candidate
    for name in ("stockfish", "stockfish.exe"):
        found = shutil.which(name)
        if found:
            return found
    for path in (ROOT / "tools" / "stockfish.exe", ROOT / "tools" / "stockfish"):
        if path.exists():
            return str(path)
    return None


def count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as handle:
        return sum(1 for _ in handle)


def main() -> int:
    engine = find_engine()
    if engine is None:
        print("No Stockfish found.")
        print()
        print("Get the Windows AVX2 build from https://stockfishchess.org/download/, unzip it,")
        print(f"and put stockfish.exe in {ROOT / 'tools'}. Then run this again.")
        return 1
    print(f"reference engine: {engine}")

    have = count_lines(POSITIONS)
    if have == 0 or "--more" in sys.argv:
        print(f"{have:,} positions on disk - playing more (Ctrl-C when you have had enough)")
        subprocess.run([sys.executable, str(ROOT / "tools" / "collect.py")], check=False)
        have = count_lines(POSITIONS)

    seen: set[str] = set()
    with FLAT.open("w") as out:
        with gzip.open(POSITIONS, "rt") as handle:
            for line in handle:
                fen = line.rsplit("|", 1)[-1].strip()
                if fen and fen not in seen:
                    seen.add(fen)
                    out.write(fen + "\n")
    total = len(seen)
    print(f"{total:,} distinct positions to label at depth {DEPTH}")

    count = max(1, (os.cpu_count() or 2) - 1)
    SHARDS.mkdir(parents=True, exist_ok=True)
    lines = FLAT.read_text().splitlines(keepends=True)
    per_worker = (total + count - 1) // count
    processes = []
    for index in range(count):
        shard = SHARDS / f"in_{index}.txt"
        shard.write_text("".join(lines[index * per_worker : (index + 1) * per_worker]))
        processes.append(
            subprocess.Popen(
                [
                    sys.executable,
                    str(ROOT / "tools" / "label.py"),
                    engine,
                    str(shard),
                    str(SHARDS / f"out_{index}.txt"),
                    str(DEPTH),
                    str(per_worker),
                ],
                stdout=subprocess.DEVNULL,
            )
        )
    print(f"{count} workers started, about 300 positions a second each")

    started = time.perf_counter()
    try:
        while any(process.poll() is None for process in processes):
            time.sleep(20)
            done = sum(count_lines(SHARDS / f"out_{i}.txt") for i in range(count))
            elapsed = time.perf_counter() - started
            rate = done / elapsed if elapsed else 0.0
            left = (total - done) / rate / 60 if rate else 0.0
            print(f"{done:,} of {total:,}   {rate:.0f}/s   {left:.0f} min left", flush=True)
    except KeyboardInterrupt:
        for process in processes:
            process.terminate()

    written = 0
    with gzip.open(OUTPUT, "wt") as out:
        for index in range(count):
            shard = SHARDS / f"out_{index}.txt"
            if not shard.exists():
                continue
            with shard.open() as handle:
                for line in handle:
                    out.write(line)
                    written += 1
    print(f"\n{written:,} labelled positions in {OUTPUT}")
    print("That is the file to send back.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
