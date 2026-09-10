"""The submission entrypoint. The platform imports this file and calls get_move.

Everything expensive happens at import: numba compiles the whole engine, and the 64 MB
transposition table is allocated once and kept for the rest of the game. By the time the clock
starts, get_move is a FEN parse, a time budget and a search.

The engine itself lives beside this file:

    bitboards.py   attack tables, magic bitboards, zobrist keys
    position.py    board state, move encoding, move generation, make/unmake
    evaluate.py    tapered hand-crafted evaluation
    search.py      negamax with alpha-beta, and the pruning that makes it worth running
    fen.py         FEN parsing and UCI formatting
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMBA_NUM_THREADS", "1")

import time

_IMPORT_STARTED = time.perf_counter()

import numpy as np  # noqa: E402

from bitboards import KING, MAX_MOVES, MAX_PLY, ST_SIDE, lsb  # noqa: E402
from evaluate import ESCRATCH_SIZE  # noqa: E402
from fen import parse, to_uci  # noqa: E402
from position import generate, is_attacked, make_move  # noqa: E402
from search import NO_MOVE, TT_SIZE, search_root  # noqa: E402

# Time control is 120 s plus 0.5 s per move. The increment means the budget below converges on
# roughly the increment as the clock runs down, instead of running it to zero.
INCREMENT_MS = 500
# Moves the budget assumes are still to come. Games here start from curated middlegames and can
# run to 300 moves a side, so this is deliberately generous rather than tuned to a short game.
EXPECTED_MOVES = 28
# Held back from the budget entirely. Without it the per-move spend converges on slightly more
# than the increment, so a long game bleeds the clock toward zero: a real 70-move game finished
# with 2.8 seconds left. Reserving this makes the steady-state spend land just under the
# increment instead, so a long game slowly refills the clock rather than draining it.
RESERVE_MS = 6000
# Held back for the FEN parse, the JSON round trip and the referee's own measurement.
OVERHEAD_MS = 55
# Never spend more than this share of the clock on one move, whatever the budget says.
MAX_SHARE = 0.28

MAX_GAME_PLIES = 1200
VERBOSE = os.environ.get("ENGINE_VERBOSE", "") == "1"

# All engine state, allocated once. None of it is reallocated between moves.
_bb = np.zeros(15, dtype=np.uint64)
_mb = np.zeros(64, dtype=np.int8)
_st = np.zeros(4, dtype=np.int64)
_key = np.zeros(1, dtype=np.uint64)
_undo = np.zeros((MAX_PLY + 8, 4), dtype=np.int64)
_moves = np.zeros((2 * (MAX_PLY + 8), MAX_MOVES), dtype=np.int32)
_scores = np.zeros((2 * (MAX_PLY + 8), MAX_MOVES), dtype=np.int32)
_tt_key = np.zeros(TT_SIZE, dtype=np.uint64)
_tt_val = np.zeros(TT_SIZE, dtype=np.uint64)
_killers = np.zeros((MAX_PLY + 8, 2), dtype=np.int32)
_history = np.zeros((2, 64, 64), dtype=np.int32)
_counters = np.zeros((12, 64), dtype=np.int32)
# Continuation history: how a move has done as a reply to each of the last two moves.
_conthist = np.zeros((2, 768, 768), dtype=np.int32)
_stack = np.zeros(MAX_PLY + 8, dtype=np.int32)
_evals = np.zeros(MAX_PLY + 8, dtype=np.int32)
_repetition = np.zeros(MAX_GAME_PLIES + MAX_PLY + 8, dtype=np.uint64)
_info = np.zeros(16, dtype=np.int64)
# Attack maps the evaluation fills in, kept here so evaluating costs no allocation.
_escratch = np.zeros(ESCRATCH_SIZE, dtype=np.uint64)

# Positions the game has actually passed through, so a search never walks into a repetition
# that the referee would claim as a draw. Index 0 is the first position we were asked about.
_game_length = 0
_move_number = 0


def _budget(time_left_ms: int) -> tuple[float, float]:
    """Return (soft, hard) budgets in seconds.

    Soft is the point past which a new iteration is not worth starting; hard is the point the
    search abandons whatever it is doing. The gap between them absorbs one iteration that turns
    out to cost more than the last.
    """
    available = time_left_ms - OVERHEAD_MS
    if available <= 0:
        return 0.0, 0.0

    usable = float(available - RESERVE_MS)
    if usable < 0.0:
        usable = 0.0
    soft_ms = usable / EXPECTED_MOVES + INCREMENT_MS * 0.70
    hard_ms = soft_ms * 3.0

    ceiling = available * MAX_SHARE
    if soft_ms > ceiling:
        soft_ms = ceiling
    if hard_ms > ceiling:
        hard_ms = ceiling

    # Down to the last few seconds, stop trying to think and just stay on the board.
    if available < 3000:
        emergency = available * 0.20
        if soft_ms > emergency:
            soft_ms = emergency
        if hard_ms > emergency:
            hard_ms = emergency

    return soft_ms / 1000.0, hard_ms / 1000.0


def _first_legal() -> int:
    side = _st[ST_SIDE]
    count = generate(_bb, _st, _moves[0], 0)
    for i in range(count):
        move = _moves[0, i]
        make_move(_bb, _mb, _st, _key, _undo, 0, np.int32(move))
        legal = not is_attacked(_bb, lsb(_bb[side * 6 + KING]), 1 - side)
        _unmake(move)
        if legal:
            return int(move)
    return int(NO_MOVE)


def _unmake(move: int) -> None:
    from position import unmake_move

    unmake_move(_bb, _mb, _st, _key, _undo, 0, np.int32(move))


def _only_legal_move() -> int:
    """The move to play when there is exactly one, or 0. Saves the clock in forced positions."""
    side = _st[ST_SIDE]
    count = generate(_bb, _st, _moves[0], 0)
    found = int(NO_MOVE)
    legal_count = 0
    for i in range(count):
        move = _moves[0, i]
        make_move(_bb, _mb, _st, _key, _undo, 0, np.int32(move))
        legal = not is_attacked(_bb, lsb(_bb[side * 6 + KING]), 1 - side)
        _unmake(move)
        if legal:
            legal_count += 1
            if legal_count > 1:
                return int(NO_MOVE)
            found = int(move)
    return found


def _record_history(fen_key: np.uint64) -> None:
    """Append the position we are about to search, keeping the ply-by-ply sequence intact."""
    global _game_length
    if _game_length >= MAX_GAME_PLIES:
        # Shift the window rather than grow it; the fifty-move clock bounds what matters anyway.
        keep = MAX_GAME_PLIES // 2
        _repetition[:keep] = _repetition[_game_length - keep : _game_length]
        _game_length = keep
    _repetition[_game_length] = fen_key
    _info[4] = _game_length
    _game_length += 1


def _second_opinion(fen: str) -> str:
    """A legal move from python-chess, used only if our own generator produced none."""
    try:
        import chess

        board = chess.Board(fen)
        for move in board.legal_moves:
            print(f"fallback: own movegen found nothing in {fen}", flush=True)
            return str(move.uci())
    except Exception as failure:
        print(f"fallback failed: {failure!r}", flush=True)
    return "0000"


def get_move(fen: str, time_left_ms: int) -> str:
    """Return a legal move in UCI notation for the side to move in `fen`."""
    global _game_length, _move_number
    started = time.perf_counter()
    _move_number += 1

    bb, mb, st, key, _ = parse(fen)
    _bb[:] = bb
    _mb[:] = mb
    _st[:] = st
    _key[0] = key[0]

    forced = _only_legal_move()
    if forced != int(NO_MOVE):
        _record_history(_key[0])
        _advance(forced)
        return to_uci(forced)

    _record_history(_key[0])

    soft, hard = _budget(time_left_ms)
    soft_deadline = started + soft
    hard_deadline = started + hard

    move = int(
        search_root(
            _bb, _mb, _st, _key, _undo, _moves, _scores, _tt_key, _tt_val, _killers,
            _history, _counters, _conthist, _stack, _evals, _repetition, _escratch, _info,
            soft_deadline, hard_deadline,
            MAX_PLY - 16, 1 if VERBOSE else 0,
        )
    )
    if move == int(NO_MOVE):
        move = _first_legal()
        if move == int(NO_MOVE):
            # Our own move generator says there is nothing to play. Either the game is already
            # over, in which case anything is fine, or the generator is wrong, in which case a
            # second opinion is the difference between a bad move and a forfeit. This has never
            # fired since the en passant parsing bug was fixed, and it costs nothing to keep.
            return _second_opinion(fen)

    elapsed = (time.perf_counter() - started) * 1000.0
    print(
        f"move {_move_number} depth {_info[7]}/{_info[3]} score {_info[6]} "
        f"nodes {_info[0]} {elapsed:.0f}ms clock {time_left_ms}ms "
        f"({_info[0] / max(elapsed, 1e-9) * 1000.0:.0f} nps)",
        flush=True,
    )

    _advance(move)
    return to_uci(move)


def _advance(move: int) -> None:
    """Play our own move on the tracked position so the next call sees an unbroken history."""
    global _game_length
    make_move(_bb, _mb, _st, _key, _undo, 0, np.int32(move))
    if _game_length < MAX_GAME_PLIES:
        _repetition[_game_length] = _key[0]
        _game_length += 1


def _warm_up() -> None:
    """Force every compiled path to run once, inside the 90 s import budget."""
    for fen, budget in (
        ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", 0.25),
        ("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1", 0.25),
        ("8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 b - - 0 1", 0.15),
    ):
        bb, mb, st, key, _ = parse(fen)
        _bb[:] = bb
        _mb[:] = mb
        _st[:] = st
        _key[0] = key[0]
        _info[4] = 0
        _repetition[0] = _key[0]
        now = time.perf_counter()
        search_root(
            _bb, _mb, _st, _key, _undo, _moves, _scores, _tt_key, _tt_val, _killers,
            _history, _counters, _conthist, _stack, _evals, _repetition, _escratch, _info,
            now + budget, now + budget, 12, 0,
        )
        # The helpers get_move calls from Python, so their entry points exist before the clock.
        _only_legal_move()
        _first_legal()
    _tt_key[:] = 0
    _tt_val[:] = 0
    _history[:] = 0
    _counters[:] = 0
    _conthist[:] = 0
    _killers[:] = 0
    _repetition[:] = 0


_warm_up()
print(f"init {time.perf_counter() - _IMPORT_STARTED:.1f}s", flush=True)
