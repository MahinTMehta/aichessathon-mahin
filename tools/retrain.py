"""Labelled positions in, a candidate engine out.

Everything between a file of `score|fen` lines and a directory you can point `tools/match.py` at:
extract the features, fit the network, quantise it to int16, prove the integer arithmetic agrees
with the fit, and write the weights into a copy of the engine.

    python3 tools/retrain.py <labelled.txt> <candidate-dir> [L1] [bound]

The extraction step runs with the network switched off on purpose. The network is fitted to the
*difference* between the reference engine's score and what the handcrafted evaluation says, so
the handcrafted evaluation is what has to be recorded — with the current network still applied,
the next one would be fitted on top of itself.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORK = Path(os.environ.get("RETRAIN_WORK", "/home/claude/work/data"))


def run(command: list[str], environment: dict[str, str] | None = None) -> None:
    print(f"\n$ {' '.join(str(part) for part in command)}", flush=True)
    result = subprocess.run(command, env=environment, check=False)
    if result.returncode != 0:
        raise SystemExit(f"failed: {' '.join(str(part) for part in command)}")


def main() -> int:
    labels = Path(sys.argv[1]).resolve()
    destination = Path(sys.argv[2]).resolve()
    width = sys.argv[3] if len(sys.argv) > 3 else "64"
    bound = sys.argv[4] if len(sys.argv) > 4 else "0"

    stem = destination.name
    features = WORK / f"{stem}_features.npz"
    fitted = WORK / f"{stem}_net.npz"

    base = dict(os.environ)
    base["PYTHONPATH"] = str(ROOT)
    base["OMP_NUM_THREADS"] = "1"

    without_network = dict(base)
    without_network["ENGINE_NO_NNUE"] = "1"
    run(
        [sys.executable, str(ROOT / "tools" / "nnue_data.py"), str(labels), str(features)],
        without_network,
    )

    fitting = dict(base)
    fitting["NNUE_BOUND"] = bound
    run(
        [
            sys.executable,
            str(ROOT / "tools" / "nnue_train.py"),
            str(features),
            str(fitted),
            width,
            "200",
        ],
        fitting,
    )

    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(ROOT, destination, ignore=shutil.ignore_patterns("__pycache__", "*.zip"))
    (destination / "weights").mkdir(exist_ok=True)
    run(
        [
            sys.executable,
            str(ROOT / "tools" / "nnue_quantise.py"),
            str(fitted),
            str(destination / "weights" / "net.npz"),
            str(features),
        ],
        base,
    )

    print(f"\n{destination} is ready. Measure it:")
    print(
        f"  python3 tools/match.py {ROOT} {destination} "
        f"--games 300 --nodes 60000 --start 0"
    )
    print("  python3 tests/test_accumulator.py   # if the engine keeps one")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
