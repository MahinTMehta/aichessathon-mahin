"""Generate evaluation training data using every core on this machine.

One command. It works out how many cores you have, starts that many self-play workers, and
writes a compressed file of labelled positions. Leave it running as long as you like and stop
it whenever — everything produced up to that point is kept.

    uv run python tools/collect.py

Optional: `uv run python tools/collect.py 400000` to stop at a position count, or add a second
number to cap the worker count.

Each position is written as `result|search score|FEN`: what the game finished as, what a
25,000-node search thought at the time, and the position itself. Those two labels together are
what the weight tuner fits the evaluation to, and better labels are the single thing that
separated a tuning run worth +55 elo from one worth nothing.
"""

import gzip
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SHARDS = ROOT / "tools" / "shards"
OUTPUT = ROOT / "tools" / "training_data.txt.gz"

GAMES_PER_WORKER = 100_000
NODES = 25_000
# Data collection runs a dozen engines at once, so each gets a small table rather than the
# quarter-gigabyte one a rated game uses. It makes no difference to a 25,000-node search.
TT_BITS = "20"


def workers_to_start(requested: int | None) -> int:
    available = os.cpu_count() or 2
    default = max(1, available - 1)
    return max(1, min(requested or default, available))


def count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("rb") as handle:
        return sum(chunk.count(b"\n") for chunk in iter(lambda: handle.read(1 << 20), b""))


def main() -> int:
    target = int(sys.argv[1]) if len(sys.argv) > 1 else 600_000
    count = workers_to_start(int(sys.argv[2]) if len(sys.argv) > 2 else None)

    SHARDS.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ)
    environment["ENGINE_TT_BITS"] = TT_BITS
    environment["OMP_NUM_THREADS"] = "1"

    print(f"{os.cpu_count()} cores detected, starting {count} workers")
    print("each spends about 40 seconds compiling before it produces anything")
    print(f"target {target:,} positions - stop with Ctrl-C whenever, nothing is lost\n")

    processes = []
    for index in range(count):
        shard = SHARDS / f"shard{index}.txt"
        processes.append(
            subprocess.Popen(
                [
                    sys.executable,
                    str(ROOT / "tools" / "make_positions.py"),
                    str(GAMES_PER_WORKER),
                    str(NODES),
                    str(shard),
                    str(1000 + index * 7919),
                ],
                cwd=str(ROOT),
                env=environment,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        )

    started = time.perf_counter()
    total = 0
    try:
        while True:
            time.sleep(20)
            total = sum(count_lines(SHARDS / f"shard{i}.txt") for i in range(count))
            elapsed = time.perf_counter() - started
            rate = total / elapsed * 3600 if elapsed > 0 else 0
            print(
                f"{total:>9,} positions  {elapsed / 60:6.1f} min  {rate:,.0f}/hour",
                flush=True,
            )
            if total >= target:
                print("\ntarget reached")
                break
            if all(process.poll() is not None for process in processes):
                print("\nworkers finished")
                break
    except KeyboardInterrupt:
        print("\nstopping")
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
        for process in processes:
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                process.kill()

    with gzip.open(OUTPUT, "wt", compresslevel=6) as out:
        for index in range(count):
            shard = SHARDS / f"shard{index}.txt"
            if shard.exists():
                with shard.open() as handle:
                    shutil.copyfileobj(handle, out)

    total = sum(count_lines(SHARDS / f"shard{i}.txt") for i in range(count))
    size = OUTPUT.stat().st_size / 1e6
    print(f"\n{total:,} positions written to {OUTPUT} ({size:.1f} MB)")
    print("Send that one file back. The shards folder can be deleted.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
