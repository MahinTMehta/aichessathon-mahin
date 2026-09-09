"""Write tuned piece-square tables back into evaluate.py as readable literals.

The tuner works against a flat parameter array, because numba folds a global array's contents
into the compiled code and an in-place change to one is invisible. That indirection costs about
fourteen percent of the search speed, which is worth paying while looking for better numbers and
not worth paying afterwards.

This takes a tuned parameter file and rewrites the tables in evaluate.py: a piece value plus a
per-square offset, laid out the way a board looks, so the result still reads as a chess
evaluation rather than a dump of 768 numbers.
"""

import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
NAMES = ("PAWN", "KNIGHT", "BISHOP", "ROOK", "QUEEN", "KING")


def table_literal(values: list[int]) -> str:
    rows = []
    for rank in range(8):
        row = values[rank * 8 : rank * 8 + 8]
        rows.append("    " + ", ".join(f"{v:4d}" for v in row) + ",")
    return "(\n" + "\n".join(rows) + "\n)"


def main() -> int:
    if len(sys.argv) < 3:
        print("usage: bake_tables.py <tuned_params.npy> <P_MG_TABLE offset>")
        return 2
    params = np.load(sys.argv[1])
    mg_base = int(sys.argv[2])
    eg_base = mg_base + 12 * 64

    text = (ROOT / "evaluate.py").read_text()
    mg_values, eg_values = [], []

    for kind, name in enumerate(NAMES):
        # Black reads the tables directly, so those entries are the table as written.
        mg = [int(params[mg_base + (6 + kind) * 64 + square]) for square in range(64)]
        eg = [int(params[eg_base + (6 + kind) * 64 + square]) for square in range(64)]
        # Split each table into a piece value plus offsets, so the file still reads as chess.
        mg_value = round(sum(mg) / 64.0)
        eg_value = round(sum(eg) / 64.0)
        mg_values.append(mg_value)
        eg_values.append(eg_value)
        text = re.sub(
            r"_MG_" + name + r" = \((?:[^()]*)\)",
            "_MG_" + name + " = " + table_literal([v - mg_value for v in mg]),
            text,
            count=1,
        )
        text = re.sub(
            r"_EG_" + name + r" = \((?:[^()]*)\)",
            "_EG_" + name + " = " + table_literal([v - eg_value for v in eg]),
            text,
            count=1,
        )

    text = re.sub(r"MG_VALUE = \([^)]*\)", f"MG_VALUE = {tuple(mg_values)}", text, count=1)
    text = re.sub(r"EG_VALUE = \([^)]*\)", f"EG_VALUE = {tuple(eg_values)}", text, count=1)
    (ROOT / "evaluate.py").write_text(text)

    print(f"MG_VALUE = {tuple(mg_values)}")
    print(f"EG_VALUE = {tuple(eg_values)}")
    print("tables written; verify with tools/verify_tables.py before trusting them")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
