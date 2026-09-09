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
    2 countdown to next clock read
"""

import time

import numpy as np
from numba import float64, int8, int32, int64, njit, objmode, uint64, void

from bitboards import (
    BISHOP,
    KING,
    KNIGHT,
    MAX_MOVES,
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
    popcount,
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

# Transposition table. Two flat arrays of 2^22 slots, about 64 MB, kept for the whole game.
TT_BITS = 22
TT_SIZE = 1 << TT_BITS
TT_MASK = U64(TT_SIZE - 1)

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
        return info[1]
    info[2] = 1024
    if now() >= deadline:
        info[1] = 1
    return info[1]


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


@njit(uint64(int32, int64, int64, int64, int64), inline="always", cache=False)
def tt_pack(move: np.int32, flag: int, depth: int, score: int, static: int) -> np.uint64:
    return (
        U64(np.int64(move) & 0x1FFFF)
        | (U64(flag) << U64(17))
        | (U64(depth & 0xFF) << U64(19))
        | (U64(score + 65536) << U64(27))
        | (U64(static + 65536) << U64(44))
    )


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


@njit(cache=False)
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
    side: int,
    ply: int,
    previous: np.int32,
) -> None:
    killer_1 = killers[ply, 0]
    killer_2 = killers[ply, 1]
    counter = NO_MOVE
    if previous != NO_MOVE:
        counter = counters[mb[move_to(previous)] % 12, move_to(previous)]

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
            if see_ge(bb, mb, move, -20):
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
            scores[i] = history[side, move_from(move), target]


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
    return scores[index]


@njit(cache=False)
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
        uint64[::1], int8[::1], int64[::1], uint64[::1], int64[:, ::1], int32[:, ::1], int32[:, ::1],
        int64[::1], float64, int64, int64, int64,
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
        return evaluate(bb, st)
    if st[ST_HALF] >= 100:
        return 0

    checked = in_check(bb, st)
    best = -INFINITE
    if not checked:
        best = evaluate(bb, st)
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
            bb, mb, st, key, undo, moves, scores, info, deadline, ply + 1, -beta, -alpha
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


@njit(
    void(
        int8[::1], int32[::1], int32[:, ::1], int32[:, :, ::1], int32[:, ::1], int64, int64,
        int64, int32, int32, int64,
    ),
    cache=False,
)
def reward_quiet(
    mb: np.ndarray,
    tried: np.ndarray,
    killers: np.ndarray,
    history: np.ndarray,
    counters: np.ndarray,
    side: int,
    ply: int,
    index: int,
    move: np.int32,
    previous: np.int32,
    depth: int,
) -> None:
    """A quiet move caused a cutoff, so remember it and demote everything tried before it."""
    if killers[ply, 0] != move:
        killers[ply, 1] = killers[ply, 0]
        killers[ply, 0] = move
    if previous != NO_MOVE:
        previous_to = move_to(previous)
        counters[mb[previous_to] % 12, previous_to] = move

    bonus = depth * depth * 5
    if bonus > 1400:
        bonus = 1400

    origin = move_from(move)
    target = move_to(move)
    current = history[side, origin, target]
    history[side, origin, target] = current + bonus - (current * bonus // 8192)

    for j in range(index):
        other = tried[j]
        if mb[move_to(other)] == 12 and move_promotion(other) == 0:
            other_from = move_from(other)
            other_to = move_to(other)
            value = history[side, other_from, other_to]
            history[side, other_from, other_to] = value - bonus - (value * bonus // 8192)


@njit(
    int64(
        uint64[::1], int8[::1], int64[::1], uint64[::1], int64[:, ::1], int32[:, ::1], int32[:, ::1],
        uint64[::1], uint64[::1], int32[:, ::1], int32[:, :, ::1], int32[:, ::1], uint64[::1], int64[::1],
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
    repetition: np.ndarray,
    info: np.ndarray,
    deadline: float,
    ply: int,
    depth: int,
    alpha: int,
    beta: int,
    can_null: int,
    previous: np.int32,
) -> int:
    info[0] += 1
    if out_of_time(info, deadline):
        return 0

    root = ply == 0
    pv_node = beta - alpha > 1
    side = st[ST_SIDE]

    position_index = info[4] + ply
    repetition[position_index] = key[0]

    if not root:
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
        return quiescence(
            bb, mb, st, key, undo, moves, scores, info, deadline, ply, alpha, beta
        )
    if ply >= MAX_PLY - 8:
        return evaluate(bb, st)

    slot = np.int64(key[0] & TT_MASK)
    tt_move = NO_MOVE
    tt_static = NO_EVAL
    if tt_key[slot] == key[0]:
        data = tt_val[slot]
        tt_move = np.int32(data & U64(0x1FFFF))
        tt_flag = np.int64((data >> U64(17)) & U64(3))
        tt_depth = np.int64((data >> U64(19)) & U64(0xFF))
        tt_score = score_from_tt(np.int64((data >> U64(27)) & U64(0x1FFFF)) - 65536, ply)
        tt_static = np.int64((data >> U64(44)) & U64(0x1FFFF)) - 65536
        if not pv_node and not root and tt_depth >= depth:
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
        static = evaluate(bb, st)

    if not pv_node and not checked and beta > -MATE_IN_MAX and beta < MATE_IN_MAX:
        # Reverse futility: so far ahead that giving back a piece per ply still clears beta.
        if depth <= 8 and static - 82 * depth >= beta:
            return static
        # Razoring: so far behind that only a tactic saves it, and quiescence would find one.
        if depth <= 3 and static + RAZOR_MARGIN * depth < alpha:
            razor = quiescence(
                bb, mb, st, key, undo, moves, scores, info, deadline, ply, alpha - 1, alpha
            )
            if razor < alpha:
                return razor
        # Null move: pass, and if the position is still winning the real move will be too.
        if can_null and depth >= 3 and static >= beta and has_non_pawn(bb, side):
            margin = (static - beta) // 190
            if margin > 3:
                margin = 3
            reduction = 3 + depth // 4 + margin
            make_null(st, key, undo, ply)
            null_score = -negamax(
                bb, mb, st, key, undo, moves, scores, tt_key, tt_val, killers, history,
                counters, repetition, info, deadline, ply + 1, depth - reduction,
                -beta, -beta + 1, 0, NO_MOVE,
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
        side, ply, previous,
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
        origin = move_from(move)
        target = move_to(move)
        promotion = move_promotion(move)
        capture = mb[target] != 12 or move_special(move) == EP_CAPTURE
        quiet = (not capture) and promotion == 0

        if legal > 0 and best_score > -MATE_IN_MAX and not root:
            if quiet:
                if depth <= 10 and legal >= LMP_TABLE[depth if depth < 12 else 11]:
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

        make_move(bb, mb, st, key, undo, ply, move)
        if is_attacked(bb, lsb(bb[king_index]), 1 - side):
            unmake_move(bb, mb, st, key, undo, ply, move)
            continue
        legal += 1
        gives_check = in_check(bb, st)
        new_depth = depth - 1

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
                reduction -= history[side, origin, target] // 6000
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
                counters, repetition, info, deadline, ply + 1, child_depth,
                child_alpha, child_beta, 1, move,
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
                            mb, moves[ply], killers, history, counters, side, ply, index,
                            move, previous, depth,
                        )
                    break

    if legal == 0:
        return -MATE + ply if checked else 0

    tt_key[slot] = key[0]
    tt_val[slot] = tt_pack(
        best_move, flag, depth, score_to_tt(best_score, ply), static
    )
    return best_score


@njit(
    int32(
        uint64[::1], int8[::1], int64[::1], uint64[::1], int64[:, ::1], int32[:, ::1], int32[:, ::1],
        uint64[::1], uint64[::1], int32[:, ::1], int32[:, :, ::1], int32[:, ::1], uint64[::1], int64[::1],
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
    repetition: np.ndarray,
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

    score = 0
    for depth in range(1, max_depth + 1):
        if depth > 1 and now() >= soft_deadline:
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
                counters, repetition, info, hard_deadline, 0, depth, alpha, beta, 0, NO_MOVE,
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
        score = value
        info[5] = info[8]
        info[6] = score
        info[7] = depth
        if verbose:
            print("depth", depth, "score", score, "nodes", info[0], "seldepth", info[3])
        if score >= MATE_IN_MAX or score <= -MATE_IN_MAX:
            break

    return np.int32(info[5])
