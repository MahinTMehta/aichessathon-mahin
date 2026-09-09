"""Fit the evaluation's weights to labelled positions. Run from the tuning checkout, not here.

This copy is included so the file that produced the shipped numbers travels with them. It needs
the parameter-array build of evaluate.py to run, because numba folds a global array's contents
into the compiled code and an in-place change to one is invisible.


The measure is the standard one: push each position's evaluation through a sigmoid to turn it
into a predicted score, and take the mean squared error against what the game actually
finished as. A weight is better if it lowers that error.

The search is plain coordinate descent — try each weight up and down by a step, keep whatever
helps, shrink the step, go round again. It is not clever, but with a hundred thousand positions
and an evaluation that runs in a microsecond, a full pass over every weight costs seconds, and
unlike a hand-picked number the result is answerable to evidence.

This only works because the weights live in an array the compiled code reads at runtime rather
than as constants baked in at compile time. Changing one is a store, not a 40-second rebuild.
"""

import gzip
import sys
import time
from pathlib import Path

import numpy as np
from numba import float64, int64, njit, uint64

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import evaluate as ev
from bitboards import PAWN, ST_SIDE, WHITE
from evaluate import evaluate

PAWN_KIND = PAWN
from fen import parse  # noqa: E402


@njit(
    float64(uint64[:, ::1], int64[:, ::1], float64[::1], float64, uint64[::1], int64[::1]),
    cache=False,
)
def total_loss(
    boards: np.ndarray,
    states: np.ndarray,
    results: np.ndarray,
    k: float,
    scratch: np.ndarray,
    params: np.ndarray,
) -> float:
    """Mean squared error between the sigmoid of the evaluation and the game result."""
    total = 0.0
    for i in range(boards.shape[0]):
        score = evaluate(boards[i], states[i], scratch, params)
        if states[i, ST_SIDE] != WHITE:
            score = -score
        predicted = 1.0 / (1.0 + np.exp(-k * score))
        error = results[i] - predicted
        total += error * error
    return total / boards.shape[0]


def load(
    path: Path, limit: int, blend: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read positions and build the target for each.

    Two labels are available: what the game finished as, and what a search thought at the time.
    The result is the thing that matters and is very noisy; the search score is much quieter and
    only as good as the evaluation that produced it. `blend` mixes them.
    """
    boards = []
    states = []
    results = []
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as handle:
        for line in handle:
            if len(boards) >= limit:
                break
            parts = line.split("|")
            if len(parts) == 2:
                outcome, fen = float(parts[0]), parts[1].strip()
                target = outcome
            elif len(parts) == 3:
                outcome, score, fen = float(parts[0]), float(parts[1]), parts[2].strip()
                searched = 1.0 / (1.0 + np.exp(-score / 220.0))
                target = (1.0 - blend) * outcome + blend * searched
            else:
                continue
            if not fen:
                continue
            bb, _, st, _, _ = parse(fen)
            boards.append(bb)
            states.append(st)
            results.append(target)
    return (
        np.ascontiguousarray(np.array(boards, dtype=np.uint64)),
        np.ascontiguousarray(np.array(states, dtype=np.int64)),
        np.array(results, dtype=np.float64),
    )


def fit_k(boards, states, results, scratch, params) -> float:
    """The sigmoid's scale is itself a parameter; fit it before touching the weights."""
    best_k, best = 0.004, 1e9
    for candidate in np.arange(0.0015, 0.0140, 0.00025):
        value = total_loss(boards, states, results, float(candidate), scratch, params)
        if value < best:
            best, best_k = value, float(candidate)
    return best_k


def main() -> int:
    source = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("tools/positions.txt")
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else 200_000
    blend = float(sys.argv[3]) if len(sys.argv) > 3 else 0.5
    start_from = Path(sys.argv[4]) if len(sys.argv) > 4 else None
    include_tables = len(sys.argv) > 6 and sys.argv[6] == "tables"
    scratch = np.zeros(16, dtype=np.uint64)

    started = time.perf_counter()
    boards, states, results = load(source, limit, blend)
    print(f"{boards.shape[0]:,} positions loaded in {time.perf_counter() - started:.0f}s")

    # Hold out a slice the tuner never sees. With hundreds of parameters and a finite sample,
    # a falling training loss stops meaning anything long before the weights stop moving; the
    # held-out loss is the one that says whether the fit is learning chess or learning noise.
    split = (boards.shape[0] * 6) // 7
    order = np.random.default_rng(20260909).permutation(boards.shape[0])
    boards, states, results = boards[order], states[order], results[order]
    boards = np.ascontiguousarray(boards)
    states = np.ascontiguousarray(states)
    holdout_boards = np.ascontiguousarray(boards[split:])
    holdout_states = np.ascontiguousarray(states[split:])
    holdout_results = np.ascontiguousarray(results[split:])
    boards = np.ascontiguousarray(boards[:split])
    states = np.ascontiguousarray(states[:split])
    results = np.ascontiguousarray(results[:split])
    print(f"tuning on {boards.shape[0]:,}, holding out {holdout_boards.shape[0]:,}")

    params = ev.EVAL_PARAMS.copy()
    if start_from is not None and start_from.exists():
        # Continue from an earlier fit rather than from the shipped defaults.
        params[:] = np.load(start_from)
        print(f"starting from {start_from}")
    k = fit_k(boards, states, results, scratch, params)
    best = total_loss(boards, states, results, k, scratch, params)
    print(f"k = {k:.5f}, starting loss {best:.6f}")

    # What the tuner is allowed to move: the scalar weights, the passed-pawn tables, the
    # king-zone attack weights, and the piece values.
    piece_mg = np.array(ev.MG_VALUE, dtype=np.int64)
    piece_eg = np.array(ev.EG_VALUE, dtype=np.int64)

    # The piece-square tables as one number per piece and square, written the way a board looks.
    # White reads them mirrored and Black reads them directly, so a tuned entry has to be
    # written into both colours at once or the evaluation stops being colour-symmetric.
    base_mg = np.zeros((6, 64), dtype=np.int64)
    base_eg = np.zeros((6, 64), dtype=np.int64)
    for kind in range(6):
        for square in range(64):
            base_mg[kind, square] = params[ev.P_MG_TABLE + (6 + kind) * 64 + square]
            base_eg[kind, square] = params[ev.P_EG_TABLE + (6 + kind) * 64 + square]

    def write_table(table: int, kind: int, square: int, value: int) -> None:
        offset = ev.P_MG_TABLE if table == 0 else ev.P_EG_TABLE
        store = base_mg if table == 0 else base_eg
        store[kind, square] = value
        params[offset + (6 + kind) * 64 + square] = value
        params[offset + kind * 64 + (square ^ 56)] = value

    # What the tuner may move: every scalar weight, the passed-pawn bonus per rank, the
    # king-zone attack weights, and the piece values (which are folded into the tables).
    targets: list[tuple[str, object, object]] = []
    for index, name in enumerate(ev.WEIGHT_NAMES):
        targets.append((name, index, False))
    for rank in range(1, 7):
        targets.append((f"PASSED_MG[{rank}]", ev.P_PASSED_MG + rank, False))
        targets.append((f"PASSED_EG[{rank}]", ev.P_PASSED_EG + rank, False))
    for kind in range(1, 5):
        targets.append((f"ATTACK_WEIGHT[{kind}]", ev.P_ATTACK + kind, False))
    if include_tables:
        # Tuning the tables makes the piece values redundant: a value is just a constant added
        # to every square of its table, and the tuner can find that itself.
        for kind in range(6):
            for square in range(64):
                if kind == PAWN_KIND and (square < 8 or square >= 56):
                    continue
                targets.append((f"MG_TABLE[{kind}][{square}]", (0, kind, square), "table"))
                targets.append((f"EG_TABLE[{kind}][{square}]", (1, kind, square), "table"))
    else:
        for kind in range(5):
            targets.append((f"PIECE_VALUE_MG[{kind}]", kind, True))
            targets.append((f"PIECE_VALUE_EG[{kind}]", 6 + kind, True))

    def limits(name: str, start: int) -> tuple[int, int]:
        """Keep the search inside sane ground.

        Coordinate descent on a finite sample will happily push a queen to 3000 centipawns if
        that shaves the loss, because scaling every weight up is nearly the same as raising the
        sigmoid's slope. Bounds, plus refitting that slope between passes, keep the numbers
        meaning what they are supposed to mean.
        """
        if name.endswith("]") and "TABLE" in name:
            # A square is allowed to move by a bishop's worth in either direction, no further.
            return start - 120, start + 120
        if name.startswith("PIECE_VALUE"):
            return int(abs(start) * 0.55), int(abs(start) * 1.7) + 10
        span = int(abs(start) * 0.9) + 45
        return start - span, start + span

    starts = {name: 0 for name, _, _ in targets}

    def read(index, piece) -> int:
        if piece == "table":
            table, kind, square = index
            store = base_mg if table == 0 else base_eg
            return int(store[kind, square])
        if not piece:
            return int(params[index])
        return int(piece_mg[index]) if index < 6 else int(piece_eg[index - 6])

    def write(index, piece, value: int) -> None:
        if piece == "table":
            table, kind, square = index
            write_table(table, kind, square, value)
        elif not piece:
            params[index] = value
        elif index < 6:
            piece_mg[index] = value
            ev.apply_piece_values(params, piece_mg, piece_eg)
        else:
            piece_eg[index - 6] = value
            ev.apply_piece_values(params, piece_mg, piece_eg)

    for name, index, piece in targets:
        starts[name] = read(index, piece)

    best_params = params.copy()
    best_held = total_loss(
        holdout_boards, holdout_states, holdout_results, k, scratch, params
    )
    stale = 0
    step_exhausted = False
    print(f"held-out loss at the start: {best_held:.6f}")

    for step in (32, 16, 8, 4, 2, 1):
        # Refit the sigmoid's slope to the weights as they now stand, so the next pass is not
        # rewarded for simply making every number bigger.
        k = fit_k(boards, states, results, scratch, params)
        best = total_loss(boards, states, results, k, scratch, params)
        improved = True
        while improved:
            improved = False
            for name, index, piece in targets:
                original = read(index, piece)
                for delta in (step, -step):
                    candidate = original + delta
                    # A divisor of zero would be a different kind of bug entirely.
                    low, high = limits(name, starts[name])
                    if candidate < low or candidate > high:
                        continue
                    if "DIVISOR" in name and candidate < 4:
                        continue
                    write(index, piece, candidate)
                    value = total_loss(boards, states, results, k, scratch, params)
                    if value < best - 1e-9:
                        best = value
                        original = candidate
                        improved = True
                    else:
                        write(index, piece, original)
            held = total_loss(
                holdout_boards, holdout_states, holdout_results, k, scratch, params
            )
            print(
                f"step {step:3d}  train {best:.6f}  held-out {held:.6f}  "
                f"({time.perf_counter() - started:.0f}s)",
                flush=True,
            )
            if held < best_held - 1e-9:
                best_held = held
                best_params[:] = params
                stale = 0
            else:
                stale += 1
                if stale >= 3:
                    print("held-out loss stopped improving; keeping the best fit so far")
                    improved = False
                    step_exhausted = True
                    break
        if step_exhausted:
            break

    params[:] = best_params
    print(f"\nbest held-out loss {best_held:.6f}")
    print("\n# Tuned weights\n")
    for index, name in enumerate(ev.WEIGHT_NAMES):
        print(f"{name} = {int(params[index])}")
    print(f"PASSED_MG = {[int(params[ev.P_PASSED_MG + r]) for r in range(8)]}")
    print(f"PASSED_EG = {[int(params[ev.P_PASSED_EG + r]) for r in range(8)]}")
    print(f"ATTACK_WEIGHT = {[int(params[ev.P_ATTACK + k]) for k in range(6)]}")
    if include_tables:
        for kind in range(6):
            print(f"MG_TABLE[{kind}] = {[int(x) for x in base_mg[kind]]}")
            print(f"EG_TABLE[{kind}] = {[int(x) for x in base_eg[kind]]}")
    else:
        print(f"MG_VALUE = {[int(x) for x in piece_mg]}")
        print(f"EG_VALUE = {[int(x) for x in piece_eg]}")
    destination = sys.argv[5] if len(sys.argv) > 5 else "tools/tuned_params.npy"
    np.save(destination, params)
    print(f"saved to {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
