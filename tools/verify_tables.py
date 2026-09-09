"""Check that a baked evaluation agrees with the parameter array it came from.

Baking rewrites literals and splits each table into a piece value plus offsets. That is a
mechanical transformation, and mechanical transformations are exactly where a silent off-by-one
lives. This evaluates the same positions both ways and insists on identical numbers.

    python3 tools/verify_tables.py <tuned_params.npy> <tuning-checkout>
"""

import random
import subprocess
import sys
from pathlib import Path

import chess

ROOT = Path(__file__).resolve().parents[1]

PROBE = '''
import json, sys
import numpy as np
sys.path.insert(0, sys.argv[1])
import evaluate as ev
from evaluate import evaluate
from fen import parse
scratch = np.zeros(16, dtype=np.uint64)
params = np.load(sys.argv[3]) if len(sys.argv) > 3 else None
out = []
for fen in json.load(open(sys.argv[2])):
    bb, _, st, _, _ = parse(fen)
    if params is None:
        out.append(int(evaluate(bb, st, scratch)))
    else:
        out.append(int(evaluate(bb, st, scratch, params)))
print(json.dumps(out))
'''


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    tuned, checkout = sys.argv[1], sys.argv[2]

    rng = random.Random(20260910)
    positions = []
    while len(positions) < 600:
        board = chess.Board()
        for _ in range(rng.randint(0, 70)):
            legal = list(board.legal_moves)
            if not legal:
                break
            board.push(rng.choice(legal))
        if not board.is_game_over(claim_draw=False):
            positions.append(board.fen())

    fens = Path("/tmp/verify_fens.json")
    fens.write_text(str(positions).replace("'", '"'))
    script = Path("/tmp/verify_probe.py")
    script.write_text(PROBE)

    baked = subprocess.run(
        [sys.executable, str(script), str(ROOT), str(fens)],
        capture_output=True, text=True, check=True,
    ).stdout.strip().splitlines()[-1]
    reference = subprocess.run(
        [sys.executable, str(script), checkout, str(fens), tuned],
        capture_output=True, text=True, check=True,
    ).stdout.strip().splitlines()[-1]

    import json

    left, right = json.loads(baked), json.loads(reference)
    mismatches = [(f, a, b) for f, a, b in zip(positions, left, right, strict=True) if a != b]
    for fen, a, b in mismatches[:5]:
        print(f"MISMATCH baked {a} vs tuned {b}  {fen}")
    print(f"{len(positions)} positions, {len(mismatches)} mismatches")
    return 1 if mismatches else 0


if __name__ == "__main__":
    raise SystemExit(main())
