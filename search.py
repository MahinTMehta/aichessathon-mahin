"""Negamax with alpha-beta, and the pruning and ordering that make it worth running.

The shape is conventional: iterative deepening with aspiration windows around the root,
principal variation search inside, a transposition table shared across the whole game, and a
quiescence search at the leaves so the evaluation is only ever asked about quiet positions.

Everything is compiled by numba, which is the only reason the node counts here are in the
millions rather than the thousands. That constrains the style: state travels as arrays rather
than objects, and the functions take long argument lists instead of closing over a context.

`info` carries the mutable scalars the search needs to share with its caller:

    0 nodes searched          3 seldepth
    1 stop flag               4 number of game positions in `repetition` before the root
    2 countdown to next clock read   10 node cap, 0 for none   12 table generation
"""

import os
import time

import numpy as np
from numba import float64, int8, int32, int64, njit, objmode, uint64, void

from bitboards import (
    BISHOP,
    KING,
    KNIGHT,
    MAX_PLY,
    OCC_ALL,
    OCC_W,
    PAWN,
    QUEEN,
    ROOK,
    ST_HALF,
    ST_SIDE,
    bishop_attacks,
    lsb,
    rook_attacks,
)
from evaluate import evaluate, insufficient_material
from position import (
    EP_CAPTURE,
    NORMAL,
    attackers_to,
    generate,
    in_check,
    is_attacked,
    make_move,
    make_null,
    move_from,
    move_promotion,
    move_special,
    move_to,
    unmake_move,
    unmake_null,
)

U64 = np.uint64
ONE = U64(1)
ZERO = U64(0)

MATE = 32000
MATE_IN_MAX = MATE - MAX_PLY
INFINITE = 32001
NO_MOVE = np.int32(0)

# Transposition table. Two flat arrays of 2^24 slots, about 256 MB, kept for the whole game.
# Sized for the search a rated move actually gets: several million nodes, which writes more
# entries than a 4-million-slot table holds, so it is overwritten inside a single move. Self-play
# at 300,000 nodes cannot see the difference and scored exactly 50%; the bench shows no speed
# cost and the process sits at 612 MB against a 2 GB limit, so this is sized by the mechanism
# rather than by a measurement the test rig is able to make.
# Slots are grouped into buckets of four: a probe looks at all four, and a store replaces the
# least valuable of them rather than whatever happened to land there. Without that, one deep
# result is thrown away by the next shallow one that hashes to the same index.
# The size is fixed for a rated game. ENGINE_TT_BITS exists only so the data-collection tools
# can run a dozen engines side by side on one machine without asking for a dozen quarter-gigabyte
# tables; the platform sets no such variable, so a rated game always gets the full size.
TT_BITS = int(os.environ.get("ENGINE_TT_BITS", "24"))
TT_SIZE = 1 << TT_BITS
TT_BUCKET = 4
TT_CLUSTERS = TT_SIZE // TT_BUCKET
TT_CLUSTER_MASK = U64(TT_CLUSTERS - 1)
TT_AGE_CYCLE = 8

TT_EXACT, TT_LOWER, TT_UPPER = 1, 2, 3

# Sentinel for "no static evaluation stored", chosen outside any score the evaluation can return.
NO_EVAL = -65536

# Values used by the static exchange evaluation. Flat, because an exchange is about counting
# material and a tapered value would only add noise.
SEE_VALUES = np.array([100, 320, 330, 500, 900, 20000, 0], dtype=np.int64)

# Ordering bands, kept far apart so a band can never leak into the one below it.
ORDER_TT = 1 << 24
ORDER_GOOD_CAPTURE = 1 << 22
ORDER_KILLER_1 = (1 << 21) + 2
ORDER_KILLER_2 = (1 << 21) + 1
ORDER_COUNTER = 1 << 21
ORDER_BAD_CAPTURE = -(1 << 22)

# Pruning margins, in centipawns.
FUTILITY_MARGIN = 110
RAZOR_MARGIN = 300
DELTA_MARGIN = 950

_LMR = np.zeros((64, 64), dtype=np.int64)
for _d in range(1, 64):
    for _m in range(1, 64):
        _LMR[_d, _m] = int(0.80 + np.log(_d) * np.log(_m) / 2.30)
LMR_TABLE = _LMR

_LMP = np.zeros(12, dtype=np.int64)
for _d in range(12):
    _LMP[_d] = 3 + _d * _d
LMP_TABLE = _LMP

# A singular verification search re-enters the same node, so it would generate over the move
# list the node is still walking. The move buffers carry a second block of rows, one per ply,
# purely to park that list across the verification.
SCRATCH = MAX_PLY + 8

# Singular extensions measured worse than they cost here: the verification search is a second
# full search of the node, and at these node counts the plies it buys are worth less than the
# plies it spends. Kept behind a flag rather than deleted, because that trade moves with depth.
USE_SINGULAR = False


@njit(float64(), cache=False)
def now() -> float:
    """Wall clock. objmode costs about half a microsecond, so this is read once per 1024 nodes."""
    with objmode(stamp="float64"):
        stamp = time.perf_counter()
    return stamp


@njit(int64(int64[::1], float64), cache=False)
def out_of_time(info: np.ndarray, deadline: float) -> int:
    info[2] -= 1
    if info[2] > 0:
        return int(info[1])
    info[2] = 1024
    if (info[10] > 0 and info[0] >= info[10]) or now() >= deadline:
        info[1] = 1
    return int(info[1])


@njit(uint64(uint64[::1], int64, uint64), inline="always", cache=False)
def all_attackers_to(bb: np.ndarray, square: int, occupancy: np.uint64) -> np.uint64:
    return attackers_to(bb, square, 0, occupancy) | attackers_to(bb, square, 1, occupancy)


@njit(int64(uint64[::1], int8[::1], int32, int64), cache=False)
def see_ge(bb: np.ndarray, mb: np.ndarray, move: np.int32, threshold: int) -> int:
    """True when the exchange on the destination square wins at least `threshold` centipawns.

    The swap-list formulation: keep taking with the cheapest attacker and see whether the side
    to move can ever stop at a point it likes.
    """
    origin = move_from(move)
    target = move_to(move)
    if move_special(move) != NORMAL or move_promotion(move):
        # En passant, castling and promotions distort the material count; let the search have them.
        return 1 if threshold <= 0 else 0

    captured = mb[target]
    swap = SEE_VALUES[6 if captured == 12 else captured % 6] - threshold
    if swap < 0:
        return 0
    mover = mb[origin]
    swap = SEE_VALUES[mover % 6] - swap
    if swap <= 0:
        return 1

    occupancy = bb[OCC_ALL] ^ (ONE << U64(origin)) ^ (ONE << U64(target))
    side = mover // 6
    attackers = all_attackers_to(bb, target, occupancy)
    result = 1

    while True:
        side = 1 - side
        attackers &= occupancy
        mine = attackers & bb[OCC_W + side]
        if not mine:
            break
        result ^= 1
        base = side * 6

        candidates = mine & bb[base + PAWN]
        if candidates:
            swap = SEE_VALUES[PAWN] - swap
            if swap < result:
                break
            occupancy ^= candidates & (~candidates + ONE)
            attackers |= bishop_attacks(U64(target), occupancy) & (
                bb[BISHOP] | bb[QUEEN] | bb[6 + BISHOP] | bb[6 + QUEEN]
            )
            continue

        candidates = mine & bb[base + KNIGHT]
        if candidates:
            swap = SEE_VALUES[KNIGHT] - swap
            if swap < result:
                break
            occupancy ^= candidates & (~candidates + ONE)
            continue

        candidates = mine & bb[base + BISHOP]
        if candidates:
            swap = SEE_VALUES[BISHOP] - swap
            if swap < result:
                break
            occupancy ^= candidates & (~candidates + ONE)
            attackers |= bishop_attacks(U64(target), occupancy) & (
                bb[BISHOP] | bb[QUEEN] | bb[6 + BISHOP] | bb[6 + QUEEN]
            )
            continue

        candidates = mine & bb[base + ROOK]
        if candidates:
            swap = SEE_VALUES[ROOK] - swap
            if swap < result:
                break
            occupancy ^= candidates & (~candidates + ONE)
            attackers |= rook_attacks(U64(target), occupancy) & (
                bb[ROOK] | bb[QUEEN] | bb[6 + ROOK] | bb[6 + QUEEN]
            )
            continue

        candidates = mine & bb[base + QUEEN]
        if candidates:
            swap = SEE_VALUES[QUEEN] - swap
            if swap < result:
                break
            occupancy ^= candidates & (~candidates + ONE)
            attackers |= (
                bishop_attacks(U64(target), occupancy)
                & (bb[BISHOP] | bb[QUEEN] | bb[6 + BISHOP] | bb[6 + QUEEN])
            ) | (
                rook_attacks(U64(target), occupancy)
                & (bb[ROOK] | bb[QUEEN] | bb[6 + ROOK] | bb[6 + QUEEN])
            )
            continue

        # Only the king is left. Taking with it is illegal if the other side still defends.
        if attackers & occupancy & bb[OCC_W + (1 - side)]:
            return result ^ 1
        return result

    return result


@njit(uint64(int32, int64, int64, int64, int64, int64), inline="always", cache=False)
def tt_pack(
    move: np.int32, flag: int, depth: int, score: int, static: int, age: int
) -> np.uint64:
    """One entry in 64 bits: move, bound type, depth, score, static evaluation and generation."""
    return (
        U64(np.int64(move) & 0x1FFFF)
        | (U64(flag) << U64(17))
        | (U64(depth & 0xFF) << U64(19))
        | (U64(score + 65536) << U64(27))
        | (U64(static + 65536) << U64(44))
        | (U64(age & 7) << U64(61))
    )


@njit(int64(uint64[::1], uint64), inline="always", cache=False)
def tt_cluster(tt_key: np.ndarray, key: np.uint64) -> int:
    return int(np.int64(key & TT_CLUSTER_MASK) * TT_BUCKET)


@njit(int64(uint64[::1], uint64[::1], int64, uint64, int64), cache=False)
def tt_slot_to_replace(
    tt_key: np.ndarray, tt_val: np.ndarray, cluster: int, key: np.uint64, age: int
) -> int:
    """Where to write. An entry for this exact position wins; otherwise the least useful goes.

    Usefulness is depth minus a penalty for being from an older search, so a deep result from
    three moves ago still outranks a shallow one from this move, but not forever.
    """
    best = cluster
    best_value = 1 << 30
    for i in range(TT_BUCKET):
        slot = cluster + i
        if tt_key[slot] == key or tt_val[slot] == ZERO:
            return slot
        entry_depth = np.int64((tt_val[slot] >> U64(19)) & U64(0xFF))
        entry_age = np.int64((tt_val[slot] >> U64(61)) & U64(7))
        drift = (age - entry_age) & (TT_AGE_CYCLE - 1)
        value = entry_depth - 6 * drift
        if value < best_value:
            best_value = value
            best = slot
    return best


@njit(int64(int64, int64), inline="always", cache=False)
def score_to_tt(score: int, ply: int) -> int:
    """Mate scores are stored as distance-to-mate from the node, not from the root."""
    if score >= MATE_IN_MAX:
        return score + ply
    if score <= -MATE_IN_MAX:
        return score - ply
    return score


@njit(int64(int64, int64), inline="always", cache=False)
def score_from_tt(score: int, ply: int) -> int:
    if score >= MATE_IN_MAX:
        return score - ply
    if score <= -MATE_IN_MAX:
        return score + ply
    return score


@njit(int64(uint64[::1], int64[::1], int64, uint64), cache=False)
def is_repetition(repetition: np.ndarray, st: np.ndarray, ply: int, key: np.uint64) -> int:
    """A single earlier occurrence is treated as a draw inside the search.

    The referee claims the third, so a line that returns to a position already on the board is
    worth exactly nothing to search further.
    """
    index = ply - 2
    stop = ply - st[ST_HALF]
    if stop < 0:
        stop = 0
    while index >= stop:
        if repetition[index] == key:
            return 1
        index -= 2
    return 0


@njit(
    void(
        uint64[::1], int8[::1], int32[::1], int32[::1], int64, int32, int32[:, ::1],
        int32[:, :, ::1], int32[:, ::1], int32[:, :, ::1], int64, int64, int64, int64,
    ),
    cache=False,
)
def order_moves(
    bb: np.ndarray,
    mb: np.ndarray,
    moves: np.ndarray,
    scores: np.ndarray,
    count: int,
    tt_move: np.int32,
    killers: np.ndarray,
    history: np.ndarray,
    counters: np.ndarray,
    conthist: np.ndarray,
    side: int,
    ply: int,
    previous: int,
    grandparent: int,
) -> None:
    """Score every move so the good ones come first. Alpha-beta is only worth what this is.

    `previous` and `grandparent` are piece-square indices (piece * 64 + destination) for the
    last two moves played, or -1 where there was none.
    """
    killer_1 = killers[ply, 0]
    killer_2 = killers[ply, 1]
    counter = NO_MOVE
    if previous >= 0:
        counter = counters[previous // 64, previous & 63]

    for i in range(count):
        move = moves[i]
        if move == tt_move:
            scores[i] = ORDER_TT
            continue
        target = move_to(move)
        victim = mb[target]
        promotion = move_promotion(move)
        if victim != 12 or move_special(move) == EP_CAPTURE or promotion:
            attacker = mb[move_from(move)] % 6
            gain = SEE_VALUES[6 if victim == 12 else victim % 6]
            if promotion:
                gain += SEE_VALUES[promotion] - SEE_VALUES[PAWN]
            elif move_special(move) == EP_CAPTURE:
                gain = SEE_VALUES[PAWN]
            base = gain * 16 - attacker
            # A capture that already wins material on the face of it cannot be a losing
            # exchange worth demoting, so the static exchange search is only run on the ones
            # where it can change the answer. That is most of the SEE calls saved, and about
            # four percent more nodes per second for an identical search tree.
            if gain >= SEE_VALUES[attacker] or see_ge(bb, mb, move, -20):
                scores[i] = ORDER_GOOD_CAPTURE + base
            else:
                scores[i] = ORDER_BAD_CAPTURE + base
        elif move == killer_1:
            scores[i] = ORDER_KILLER_1
        elif move == killer_2:
            scores[i] = ORDER_KILLER_2
        elif move == counter:
            scores[i] = ORDER_COUNTER
        else:
            # The move's own record, plus how it has done as a reply to these exact last moves.
            value = history[side, move_from(move), target]
            piece_square = (mb[move_from(move)] if promotion == 0 else side * 6 + promotion) * 64
            piece_square += target
            if previous >= 0:
                value += conthist[0, previous, piece_square]
            if grandparent >= 0:
                value += conthist[1, grandparent, piece_square]
            scores[i] = value


@njit(int64(int32[::1], int32[::1], int64, int64), cache=False)
def pick_best(moves: np.ndarray, scores: np.ndarray, count: int, index: int) -> int:
    """Selection sort, one step at a time, so a cutoff never pays for sorting the tail."""
    best = index
    for i in range(index + 1, count):
        if scores[i] > scores[best]:
            best = i
    if best != index:
        moves[index], moves[best] = moves[best], moves[index]
        scores[index], scores[best] = scores[best], scores[index]
    return int(scores[index])


@njit(void(uint64[::1], int8[::1], int32[::1], int32[::1], int64), cache=False)
def order_captures(
    bb: np.ndarray, mb: np.ndarray, moves: np.ndarray, scores: np.ndarray, count: int
) -> None:
    """Most valuable victim first, cheapest attacker as the tie-break."""
    for i in range(count):
        move = moves[i]
        victim = mb[move_to(move)]
        gain = SEE_VALUES[6 if victim == 12 else victim % 6]
        promotion = move_promotion(move)
        if promotion:
            gain += SEE_VALUES[promotion] - SEE_VALUES[PAWN]
        elif move_special(move) == EP_CAPTURE:
            gain = SEE_VALUES[PAWN]
        scores[i] = gain * 16 - (mb[move_from(move)] % 6)


@njit(int64(uint64[::1], int64), cache=False)
def has_non_pawn(bb: np.ndarray, side: int) -> int:
    base = side * 6
    return 1 if (bb[base + KNIGHT] | bb[base + BISHOP] | bb[base + ROOK] | bb[base + QUEEN]) else 0


@njit(
    int64(
        uint64[::1], int8[::1], int64[::1], uint64[::1], int64[:, ::1], int32[:, ::1],
        int32[:, ::1],
        uint64[::1], int64[::1], float64, int64, int64, int64,
    ),
    cache=False,
)
def quiescence(
    bb: np.ndarray,
    mb: np.ndarray,
    st: np.ndarray,
    key: np.ndarray,
    undo: np.ndarray,
    moves: np.ndarray,
    scores: np.ndarray,
    escratch: np.ndarray,
    info: np.ndarray,
    deadline: float,
    ply: int,
    alpha: int,
    beta: int,
) -> int:
    """Search on past the horizon until nothing is hanging, so evaluate() sees a quiet board."""
    info[0] += 1
    if out_of_time(info, deadline):
        return 0
    if ply > info[3]:
        info[3] = ply
    if ply >= MAX_PLY - 2:
        return int(evaluate(bb, st, escratch))
    if st[ST_HALF] >= 100:
        return 0

    checked = in_check(bb, st)
    best = -INFINITE
    if not checked:
        best = int(evaluate(bb, st, escratch))
        if best >= beta:
            return best
        if best > alpha:
            alpha = best

    count = generate(bb, st, moves[ply], 0 if checked else 1)
    order_captures(bb, mb, moves[ply], scores[ply], count)
    side = st[ST_SIDE]
    king_index = side * 6 + KING
    legal = 0

    for index in range(count):
        pick_best(moves[ply], scores[ply], count, index)
        move = moves[ply, index]
        if not checked:
            # Nothing this capture could win brings the score near alpha, so skip it.
            victim = mb[move_to(move)]
            gain = SEE_VALUES[6 if victim == 12 else victim % 6]
            promotion = move_promotion(move)
            if promotion:
                gain += SEE_VALUES[promotion] - SEE_VALUES[PAWN]
            elif move_special(move) == EP_CAPTURE:
                gain = SEE_VALUES[PAWN]
            if best + gain + 180 < alpha:
                continue
            if not see_ge(bb, mb, move, 0):
                continue

        make_move(bb, mb, st, key, undo, ply, move)
        if is_attacked(bb, lsb(bb[king_index]), 1 - side):
            unmake_move(bb, mb, st, key, undo, ply, move)
            continue
        legal += 1
        score = -quiescence(
            bb, mb, st, key, undo, moves, scores, escratch, info, deadline, ply + 1, -beta, -alpha
        )
        unmake_move(bb, mb, st, key, undo, ply, move)
        if info[1]:
            return 0
        if score > best:
            best = score
            if score > alpha:
                alpha = score
                if score >= beta:
                    break

    if checked and legal == 0:
        return -MATE + ply
    return best


@njit(int32(int8[::1], int32, int64), inline="always", cache=False)
def piece_square_of(mb: np.ndarray, move: np.int32, side: int) -> np.int32:
    """The (piece, destination) pair a move produces, which is how continuation history is keyed."""
    promotion = move_promotion(move)
    piece = mb[move_from(move)] if promotion == 0 else side * 6 + promotion
    return np.int32(piece * 64 + move_to(move))


@njit(int64(int32, int64), inline="always", cache=False)
def nudge(value: int, bonus: int) -> int:
    """Move a history score toward its ceiling, so old entries decay instead of saturating."""
    return value + bonus - (value * (bonus if bonus > 0 else -bonus) // 8192)


@njit(
    void(
        int8[::1], int32[::1], int32[:, ::1], int32[:, :, ::1], int32[:, ::1], int32[:, :, ::1],
        int64, int64, int64, int32, int64, int64, int64,
    ),
    cache=False,
)
def reward_quiet(
    mb: np.ndarray,
    tried: np.ndarray,
    killers: np.ndarray,
    history: np.ndarray,
    counters: np.ndarray,
    conthist: np.ndarray,
    side: int,
    ply: int,
    index: int,
    move: np.int32,
    previous: int,
    grandparent: int,
    depth: int,
) -> None:
    """A quiet move caused a cutoff, so remember it and demote everything tried before it."""
    if killers[ply, 0] != move:
        killers[ply, 1] = killers[ply, 0]
        killers[ply, 0] = move
    if previous >= 0:
        counters[previous // 64, previous & 63] = move

    bonus = depth * depth * 5
    if bonus > 1400:
        bonus = 1400

    origin = move_from(move)
    target = move_to(move)
    history[side, origin, target] = nudge(history[side, origin, target], bonus)
    piece_square = piece_square_of(mb, move, side)
    if previous >= 0:
        conthist[0, previous, piece_square] = nudge(conthist[0, previous, piece_square], bonus)
    if grandparent >= 0:
        conthist[1, grandparent, piece_square] = nudge(
            conthist[1, grandparent, piece_square], bonus
        )

    for j in range(index):
        other = tried[j]
        if mb[move_to(other)] == 12 and move_promotion(other) == 0:
            other_from = move_from(other)
            other_to = move_to(other)
            history[side, other_from, other_to] = nudge(
                history[side, other_from, other_to], -bonus
            )
            other_square = piece_square_of(mb, other, side)
            if previous >= 0:
                conthist[0, previous, other_square] = nudge(
                    conthist[0, previous, other_square], -bonus
                )
            if grandparent >= 0:
                conthist[1, grandparent, other_square] = nudge(
                    conthist[1, grandparent, other_square], -bonus
                )


@njit(
    int64(
        uint64[::1], int8[::1], int64[::1], uint64[::1], int64[:, ::1], int32[:, ::1],
        int32[:, ::1],
        uint64[::1], uint64[::1], int32[:, ::1], int32[:, :, ::1], int32[:, ::1],
        int32[:, :, ::1], int32[::1], int32[::1], uint64[::1], uint64[::1], int64[::1],
        float64, int64, int64, int64, int64, int64, int32,
    ),
    cache=False,
)
def negamax(
    bb: np.ndarray,
    mb: np.ndarray,
    st: np.ndarray,
    key: np.ndarray,
    undo: np.ndarray,
    moves: np.ndarray,
    scores: np.ndarray,
    tt_key: np.ndarray,
    tt_val: np.ndarray,
    killers: np.ndarray,
    history: np.ndarray,
    counters: np.ndarray,
    conthist: np.ndarray,
    stack: np.ndarray,
    evals: np.ndarray,
    repetition: np.ndarray,
    escratch: np.ndarray,
    info: np.ndarray,
    deadline: float,
    ply: int,
    depth: int,
    alpha: int,
    beta: int,
    can_null: int,
    excluded: np.int32,
) -> int:
    info[0] += 1
    if out_of_time(info, deadline):
        return 0

    root = ply == 0
    pv_node = beta - alpha > 1
    side = st[ST_SIDE]
    previous = stack[ply - 1] if ply > 0 else -1
    grandparent = stack[ply - 2] if ply > 1 else -1

    position_index = info[4] + ply
    repetition[position_index] = key[0]

    if not root and excluded == NO_MOVE:
        if st[ST_HALF] >= 100 or insufficient_material(bb):
            return 0
        if is_repetition(repetition, st, position_index, key[0]):
            return 0
        # Mate distance pruning: a shorter mate already found elsewhere caps what is worth finding.
        if alpha < -MATE + ply:
            alpha = -MATE + ply
        if beta > MATE - ply - 1:
            beta = MATE - ply - 1
        if alpha >= beta:
            return alpha

    checked = in_check(bb, st)
    if checked and depth < MAX_PLY - 8:
        depth += 1

    if depth <= 0:
        return int(quiescence(
            bb, mb, st, key, undo, moves, scores, escratch, info, deadline, ply, alpha, beta
        ))
    if ply >= MAX_PLY - 8:
        return int(evaluate(bb, st, escratch))

    cluster = tt_cluster(tt_key, key[0])
    tt_move = NO_MOVE
    tt_static = NO_EVAL
    tt_depth = -1
    tt_score = 0
    tt_flag = 0
    hit = -1
    for i in range(TT_BUCKET):
        if tt_key[cluster + i] == key[0] and tt_val[cluster + i] != ZERO:
            hit = cluster + i
            break
    if hit >= 0:
        data = tt_val[hit]
        tt_move = np.int32(data & U64(0x1FFFF))
        tt_flag = np.int64((data >> U64(17)) & U64(3))
        tt_depth = np.int64((data >> U64(19)) & U64(0xFF))
        tt_score = int(score_from_tt(np.int64((data >> U64(27)) & U64(0x1FFFF)) - 65536, ply))
        tt_static = np.int64((data >> U64(44)) & U64(0x1FFFF)) - 65536
        if not pv_node and not root and excluded == NO_MOVE and tt_depth >= depth:
            if tt_flag == TT_EXACT:
                return tt_score
            if tt_flag == TT_LOWER and tt_score >= beta:
                return tt_score
            if tt_flag == TT_UPPER and tt_score <= alpha:
                return tt_score

    if checked:
        static = NO_EVAL
    elif tt_static != NO_EVAL:
        static = tt_static
    else:
        static = int(evaluate(bb, st, escratch))
    evals[ply] = static

    # Whether our position has got better since our last turn. When it has not, the position is
    # worth less benefit of the doubt and every pruning margin below tightens.
    improving = 0
    if not checked and ply >= 2 and evals[ply - 2] != NO_EVAL and static > evals[ply - 2]:
        improving = 1

    quiet_window = beta > -MATE_IN_MAX and beta < MATE_IN_MAX
    if not pv_node and not checked and excluded == NO_MOVE and quiet_window:
        # Reverse futility: so far ahead that giving back a piece per ply still clears beta.
        if depth <= 8 and static - (82 - 22 * improving) * depth >= beta:
            return static
        # Razoring: so far behind that only a tactic saves it, and quiescence would find one.
        if depth <= 3 and static + RAZOR_MARGIN * depth < alpha:
            razor = int(quiescence(
                bb, mb, st, key, undo, moves, scores, escratch, info, deadline, ply,
                alpha - 1, alpha,
            ))
            if razor < alpha:
                return razor
        # Null move: pass, and if the position is still winning the real move will be too.
        if can_null and depth >= 3 and static >= beta and has_non_pawn(bb, side):
            margin = (static - beta) // 190
            if margin > 3:
                margin = 3
            reduction = 3 + depth // 4 + margin
            stack[ply] = -1
            make_null(st, key, undo, ply)
            null_score = -negamax(
                bb, mb, st, key, undo, moves, scores, tt_key, tt_val, killers, history,
                counters, conthist, stack, evals, repetition, escratch, info, deadline, ply + 1,
                depth - reduction, -beta, -beta + 1, 0, NO_MOVE,
            )
            unmake_null(st, key, undo, ply)
            if info[1]:
                return 0
            if null_score >= beta:
                return beta if null_score >= MATE_IN_MAX else null_score

    # With no table move to try first, a shallower pass is cheaper than searching in the dark.
    if tt_move == NO_MOVE and depth >= 5 and pv_node:
        depth -= 1

    count = generate(bb, st, moves[ply], 0)
    if tt_move != NO_MOVE:
        found = 0
        for i in range(count):
            if moves[ply, i] == tt_move:
                found = 1
                break
        if not found:
            tt_move = NO_MOVE

    order_moves(
        bb, mb, moves[ply], scores[ply], count, tt_move, killers, history, counters,
        conthist, side, ply, previous, grandparent,
    )

    killers[ply + 1, 0] = NO_MOVE
    killers[ply + 1, 1] = NO_MOVE

    king_index = side * 6 + KING
    best_score = -INFINITE
    best_move = NO_MOVE
    flag = TT_UPPER
    legal = 0

    for index in range(count):
        move_score = pick_best(moves[ply], scores[ply], count, index)
        move = moves[ply, index]
        if move == excluded:
            continue
        origin = move_from(move)
        target = move_to(move)
        promotion = move_promotion(move)
        capture = mb[target] != 12 or move_special(move) == EP_CAPTURE
        quiet = (not capture) and promotion == 0

        if legal > 0 and best_score > -MATE_IN_MAX and not root:
            if quiet:
                limit = LMP_TABLE[depth if depth < 12 else 11]
                if not improving:
                    limit = limit // 2 + 1
                if depth <= 10 and legal >= limit:
                    continue
                if (
                    depth <= 6
                    and not checked
                    and static != NO_EVAL
                    and static + FUTILITY_MARGIN * depth + 90 <= alpha
                ):
                    continue
                if depth <= 7 and not see_ge(bb, mb, move, -28 * depth * depth):
                    continue
            elif depth <= 6 and not see_ge(bb, mb, move, -92 * depth):
                continue

        extension = 0
        if (
            USE_SINGULAR
            and not root
            and excluded == NO_MOVE
            and move == tt_move
            and depth >= 7
            and tt_depth >= depth - 3
            and tt_flag != TT_UPPER
            and tt_score > -MATE_IN_MAX
            and tt_score < MATE_IN_MAX
        ):
            # Is this move the only one that holds? Search the rest with a window just below the
            # table score; if they all fail low, the move is singular and deserves an extra ply.
            border = tt_score - 2 * depth
            for i in range(count):
                moves[SCRATCH + ply, i] = moves[ply, i]
                scores[SCRATCH + ply, i] = scores[ply, i]
            saved_static = evals[ply]
            verify = negamax(
                bb, mb, st, key, undo, moves, scores, tt_key, tt_val, killers, history,
                counters, conthist, stack, evals, repetition, escratch, info, deadline, ply,
                (depth - 1) // 2, border - 1, border, 0, move,
            )
            evals[ply] = saved_static
            for i in range(count):
                moves[ply, i] = moves[SCRATCH + ply, i]
                scores[ply, i] = scores[SCRATCH + ply, i]
            if info[1]:
                return 0
            if verify < border:
                extension = 1
            elif border >= beta:
                # Everything else already beats beta, so this node is not worth finishing.
                return border

        stack[ply] = piece_square_of(mb, move, side)
        make_move(bb, mb, st, key, undo, ply, move)
        if is_attacked(bb, lsb(bb[king_index]), 1 - side):
            unmake_move(bb, mb, st, key, undo, ply, move)
            continue
        legal += 1
        gives_check = in_check(bb, st)
        new_depth = depth - 1 + extension

        # One recursive call site, walked through up to three stages: a reduced null-window
        # probe, the same window at full depth, and only then the full window. Collapsing the
        # usual four call sites into one keeps the compiled function small enough that numba
        # finishes inside the import budget.
        reduction = 0
        if legal == 1:
            stage = 0
        else:
            if depth >= 3 and legal >= 3 and quiet and not gives_check and not checked:
                reduction = LMR_TABLE[
                    depth if depth < 63 else 63, legal if legal < 63 else 63
                ]
                if pv_node:
                    reduction -= 1
                if move_score >= ORDER_COUNTER:
                    reduction -= 1
                if not improving:
                    reduction += 1
                total = history[side, origin, target]
                square = piece_square_of(mb, move, side)
                if previous >= 0:
                    total += conthist[0, previous, square]
                if grandparent >= 0:
                    total += conthist[1, grandparent, square]
                reduction -= total // 7000
                if reduction < 0:
                    reduction = 0
                elif reduction > new_depth - 1:
                    reduction = new_depth - 1
            stage = 1 if reduction > 0 else 2

        while True:
            if stage == 1:
                child_depth = new_depth - reduction
                child_alpha = -alpha - 1
                child_beta = -alpha
            elif stage == 2:
                child_depth = new_depth
                child_alpha = -alpha - 1
                child_beta = -alpha
            else:
                child_depth = new_depth
                child_alpha = -beta
                child_beta = -alpha
            score = -negamax(
                bb, mb, st, key, undo, moves, scores, tt_key, tt_val, killers, history,
                counters, conthist, stack, evals, repetition, escratch, info, deadline, ply + 1,
                child_depth, child_alpha, child_beta, 1, NO_MOVE,
            )
            if info[1] or stage == 0 or stage == 3 or score <= alpha:
                break
            if stage == 1:
                stage = 2
            elif pv_node and score < beta:
                stage = 3
            else:
                break

        unmake_move(bb, mb, st, key, undo, ply, move)

        if info[1]:
            return 0

        if score > best_score:
            best_score = score
            best_move = move
            if root:
                info[8] = move
                info[9] = score
            if score > alpha:
                alpha = score
                flag = TT_EXACT
                if score >= beta:
                    flag = TT_LOWER
                    if quiet:
                        reward_quiet(
                            mb, moves[ply], killers, history, counters, conthist, side, ply,
                            index, move, previous, grandparent, depth,
                        )
                    break

    if legal == 0:
        return -MATE + ply if checked else 0

    if excluded == NO_MOVE:
        slot = tt_slot_to_replace(tt_key, tt_val, cluster, key[0], info[12])
        tt_key[slot] = key[0]
        tt_val[slot] = tt_pack(
            best_move, flag, depth, score_to_tt(best_score, ply), static, info[12]
        )
    return best_score


@njit(
    int32(
        uint64[::1], int8[::1], int64[::1], uint64[::1], int64[:, ::1], int32[:, ::1],
        int32[:, ::1],
        uint64[::1], uint64[::1], int32[:, ::1], int32[:, :, ::1], int32[:, ::1],
        int32[:, :, ::1], int32[::1], int32[::1], uint64[::1], uint64[::1], int64[::1],
        float64, float64, int64, int64,
    ),
    cache=False,
)
def search_root(
    bb: np.ndarray,
    mb: np.ndarray,
    st: np.ndarray,
    key: np.ndarray,
    undo: np.ndarray,
    moves: np.ndarray,
    scores: np.ndarray,
    tt_key: np.ndarray,
    tt_val: np.ndarray,
    killers: np.ndarray,
    history: np.ndarray,
    counters: np.ndarray,
    conthist: np.ndarray,
    stack: np.ndarray,
    evals: np.ndarray,
    repetition: np.ndarray,
    escratch: np.ndarray,
    info: np.ndarray,
    soft_deadline: float,
    hard_deadline: float,
    max_depth: int,
    verbose: int,
) -> np.int32:
    """Iterative deepening. Every pass leaves a better move behind than the one before it."""
    info[0] = 0
    info[1] = 0
    info[2] = 1
    info[3] = 0
    info[8] = NO_MOVE
    info[9] = 0
    # A new generation for the table, so entries from earlier moves age out rather than
    # competing on depth alone with what this search is finding.
    info[12] = (info[12] + 1) & (TT_AGE_CYCLE - 1)
    for i in range(stack.shape[0]):
        stack[i] = -1
        evals[i] = NO_EVAL

    # A legal move in hand before any searching, so there is always something to return.
    side = st[ST_SIDE]
    king_index = side * 6 + KING
    fallback = NO_MOVE
    count = generate(bb, st, moves[0], 0)
    for i in range(count):
        candidate = moves[0, i]
        make_move(bb, mb, st, key, undo, 0, candidate)
        legal = not is_attacked(bb, lsb(bb[king_index]), 1 - side)
        unmake_move(bb, mb, st, key, undo, 0, candidate)
        if legal:
            fallback = candidate
            break
    info[5] = fallback
    info[6] = 0
    info[7] = 0
    if fallback == NO_MOVE:
        return NO_MOVE

    # Time management. The soft deadline is a budget, not a fixed point: a position whose best
    # move keeps changing, or whose score is falling, deserves more of the clock than one where
    # the same move has survived five iterations.
    started = now()
    budget = soft_deadline - started
    stability = 0
    previous_move = NO_MOVE
    previous_score = 0

    score = 0
    for depth in range(1, max_depth + 1):
        if depth > 1:
            factor = 1.0
            if stability >= 5:
                factor = 0.55
            elif stability >= 3:
                factor = 0.72
            elif stability >= 1:
                factor = 0.88
            else:
                factor = 1.35
            if depth > 5 and score < previous_score - 35:
                # We are losing the thread. Buy another iteration to find out why.
                factor *= 1.6
            effective = started + budget * factor
            if effective > hard_deadline:
                effective = hard_deadline
            if now() >= effective:
                break
        delta = 18
        if depth >= 5:
            alpha = score - delta
            beta = score + delta
        else:
            alpha = -INFINITE
            beta = INFINITE

        while True:
            value = negamax(
                bb, mb, st, key, undo, moves, scores, tt_key, tt_val, killers, history,
                counters, conthist, stack, evals, repetition, escratch, info, hard_deadline, 0,
                depth, alpha, beta, 0, NO_MOVE,
            )
            if info[1]:
                break
            if value <= alpha:
                beta = (alpha + beta) // 2
                alpha = value - delta
                if alpha < -INFINITE:
                    alpha = -INFINITE
            elif value >= beta:
                beta = value + delta
                if beta > INFINITE:
                    beta = INFINITE
            else:
                break
            delta += delta // 2 + 6
            if now() >= hard_deadline:
                info[1] = 1
                break

        if info[1]:
            break
        previous_score = score
        score = value
        if np.int32(info[8]) == previous_move:
            stability += 1
        else:
            stability = 0
        previous_move = np.int32(info[8])
        info[5] = info[8]
        info[6] = score
        info[7] = depth
        if verbose:
            print("depth", depth, "score", score, "nodes", info[0], "seldepth", info[3])
        if score >= MATE_IN_MAX or score <= -MATE_IN_MAX:
            break

    return np.int32(info[5])
