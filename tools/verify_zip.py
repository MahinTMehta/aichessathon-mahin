"""Prove that the zip plays the same chess as the working tree.

The packaging harness plays two smoke games out of the archive, which catches an agent that
crashes or cannot find a module. It does not catch the failure that matters most here: a data
file left out. The network loads if `net.npz` is present and silently does not if it is absent,
and an engine missing its network still starts, still plays legal chess, and is simply weaker —
by rather more than everything else measured in this file put together.

So this compares evaluations. Same positions, same numbers, or the zip is not the engine.

    python3 tools/verify_zip.py <submission.zip>
"""

import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

POSITIONS = (
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
    "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 b - - 0 1",
    "6k1/5ppp/8/8/8/8/5PPP/3R2K1 w - - 0 1",
    "8/8/8/4k3/8/4K3/4P3/8 w - - 0 1",
    "4k3/8/8/8/8/8/8/4K2R w K - 0 1",
)

PROBE = """
import sys, numpy as np
from fen import parse
from evaluate import ESCRATCH_SIZE, evaluate
scratch = np.zeros(ESCRATCH_SIZE, dtype=np.uint64)
import nnue
print("net", int(nnue.ENABLED), nnue.L1)
for line in sys.argv[1:]:
    bb, mb, st, key, full = parse(line)
    print(int(evaluate(bb, st, scratch)))
"""


def probe(directory: Path) -> list[str]:
    result = subprocess.run(
        [sys.executable, "-c", PROBE, *POSITIONS],
        cwd=directory,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise SystemExit(f"probe failed in {directory}:\n{result.stderr}")
    return result.stdout.split()


def main() -> int:
    archive_path = Path(sys.argv[1])
    with zipfile.ZipFile(archive_path) as archive:
        names = archive.namelist()
        print(f"{archive_path} contains {len(names)} files")
        for name in names:
            print(f"  {name}")
        if not any(name.endswith("net.npz") for name in names):
            print("\nNo net.npz in the archive - the network would not load")
            return 1
        with tempfile.TemporaryDirectory() as workspace:
            extracted = Path(workspace) / "agent"
            archive.extractall(extracted)
            from_zip = probe(extracted)
    from_tree = probe(ROOT)

    if from_zip != from_tree:
        print("\nThe zip does not evaluate the same as the working tree:")
        print(f"  zip:  {from_zip}")
        print(f"  tree: {from_tree}")
        return 1
    if from_zip[1] != "1":
        print("\nThe archive imports, but the network did not load inside it")
        return 1
    print(
        f"\n{len(POSITIONS)} positions, identical evaluations, network loaded, "
        f"{from_zip[2]} wide"
    )
    print("  " + " ".join(from_tree[3:]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
