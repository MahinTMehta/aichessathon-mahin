"""Position state, move encoding, move generation and make/unmake.

A position is four plain arrays rather than an object, because numba compiles free functions
over arrays far better than it compiles classes:

    bb    uint64[15]  one bitboard per colour-tagged piece, then the two occupancies and their union
    mb    int8[64]    the same information as a mailbox, so "what is on this square" is one read
    st    int64[4]    side to move, castling rights, en passant square, halfmove clock
    key   uint64[1]   the zobrist hash of everything above

make_move writes an undo record into a per-ply stack so unmake_move can restore the position
without copying it.

A move is packed into one int32:

    bits 0-5    origin square
    bits 6-11   destination square
    bits 12-14  promotion piece type, 0 when the move is not a promotion
    bits 15-16  0 normal, 1 en passant, 2 castling
"""

import numpy as np
from numba import int8, int32, int64, njit, uint64

from bitboards import (
    BISHOP,
    CR_BK,
    CR_BQ,
    CR_WK,
    CR_WQ,
    EMPTY,
    KING,
    KING_ATTACKS,
    KNIGHT,
    KNIGHT_ATTACKS,
    OCC_ALL,
    OCC_B,
    OCC_W,
    PAWN,
    PAWN_ATTACKS,
    QUEEN,
    ROOK,
    ST_CASTLE,
    ST_EP,
    ST_HALF,
    ST_SIDE,
    WHITE,
    ZOBRIST_CASTLE,
    ZOBRIST_EP,
    ZOBRIST_PIECE,
    ZOBRIST_SIDE,
    bishop_attacks,
    lsb,
    queen_attacks,
    rook_attacks,
)

U64 = np.uint64
ONE = U64(1)
ZERO = U64(0)

RANK_2_BB = U64(0x000000000000FF00)
RANK_7_BB = U64(0x00FF000000000000)

# Losing a right is keyed on a square being vacated or captured on, which covers king moves,
# rook moves and rooks captured on their home square in one table.
_castle_mask = np.full(64, 15, dtype=np.int64)
_castle_mask[4] = 15 ^ (CR_WK | CR_WQ)
_castle_mask[0] = 15 ^ CR_WQ
_castle_mask[7] = 15 ^ CR_WK
_castle_mask[60] = 15 ^ (CR_BK | CR_BQ)
_castle_mask[56] = 15 ^ CR_BQ
_castle_mask[63] = 15 ^ CR_BK
CASTLE_MASK = _castle_mask

NORMAL, EP_CAPTURE, CASTLING = 0, 1, 2


@njit(int32(int64, int64, int64, int64), inline="always", cache=False)
def encode(origin: int, target: int, promotion: int, special: int) -> np.int32:
    return np.int32(origin | (target << 6) | (promotion << 12) | (special << 15))


@njit(int64(int32), inline="always", cache=False)
def move_from(move: np.int32) -> int:
    return np.int64(move) & 63


@njit(int64(int32), inline="always", cache=False)
def move_to(move: np.int32) -> int:
    return (np.int64(move) >> 6) & 63


@njit(int64(int32), inline="always", cache=False)
def move_promotion(move: np.int32) -> int:
    return (np.int64(move) >> 12) & 7


@njit(int64(int32), inline="always", cache=False)
def move_special(move: np.int32) -> int:
    return (np.int64(move) >> 15) & 3


@njit(uint64(uint64[::1], int64, int64, uint64), cache=False)
def attackers_to(bb: np.ndarray, square: int, side: int, occupancy: np.uint64) -> np.uint64:
    """Every piece of `side` that attacks `square` given `occupancy`."""
    base = side * 6
    sq = uint64(square)
    attacks = PAWN_ATTACKS[1 - side, square] & bb[base + PAWN]
    attacks |= KNIGHT_ATTACKS[square] & bb[base + KNIGHT]
    attacks |= KING_ATTACKS[square] & bb[base + KING]
    diagonal = bb[base + BISHOP] | bb[base + QUEEN]
    if diagonal:
        attacks |= bishop_attacks(sq, occupancy) & diagonal
    straight = bb[base + ROOK] | bb[base + QUEEN]
    if straight:
        attacks |= rook_attacks(sq, occupancy) & straight
    return attacks


@njit(int64(uint64[::1], int64, int64), cache=False)
def is_attacked(bb: np.ndarray, square: int, side: int) -> int:
    return 1 if attackers_to(bb, square, side, bb[OCC_ALL]) else 0


@njit(int64(uint64[::1], int64[::1]), cache=False)
def in_check(bb: np.ndarray, st: np.ndarray) -> int:
    side = st[ST_SIDE]
    king = bb[side * 6 + KING]
    if king == ZERO:
        return 0
    return is_attacked(bb, lsb(king), 1 - side)


@njit(cache=False)
def generate(bb: np.ndarray, st: np.ndarray, moves: np.ndarray, captures_only: int) -> int:
    """Pseudo-legal moves into `moves`, returning how many. Legality is checked after make_move.

    With captures_only set, only captures and promotions are produced: that is the quiescence
    move set.
    """
    side = st[ST_SIDE]
    base = side * 6
    own = bb[OCC_W + side]
    enemy = bb[OCC_W + (1 - side)]
    occupancy = bb[OCC_ALL]
    empty = ~occupancy
    targets = enemy if captures_only else ~own
    count = 0

    # Pawns. Directions are written out per colour rather than parameterised, because the
    # shifts have to stay compile-time constants for numba to fold them.
    pawns = bb[base + PAWN]
    if side == WHITE:
        promoting = pawns & RANK_7_BB
        ordinary = pawns ^ promoting
        if promoting:
            pushes = (promoting << U64(8)) & empty
            while pushes:
                target = lsb(pushes)
                pushes &= pushes - ONE
                for piece in (QUEEN, ROOK, BISHOP, KNIGHT):
                    moves[count] = encode(target - 8, target, piece, NORMAL)
                    count += 1
            left = ((promoting & ~U64(0x0101010101010101)) << U64(7)) & enemy
            while left:
                target = lsb(left)
                left &= left - ONE
                for piece in (QUEEN, ROOK, BISHOP, KNIGHT):
                    moves[count] = encode(target - 7, target, piece, NORMAL)
                    count += 1
            right = ((promoting & ~U64(0x8080808080808080)) << U64(9)) & enemy
            while right:
                target = lsb(right)
                right &= right - ONE
                for piece in (QUEEN, ROOK, BISHOP, KNIGHT):
                    moves[count] = encode(target - 9, target, piece, NORMAL)
                    count += 1
        left = ((ordinary & ~U64(0x0101010101010101)) << U64(7)) & enemy
        while left:
            target = lsb(left)
            left &= left - ONE
            moves[count] = encode(target - 7, target, 0, NORMAL)
            count += 1
        right = ((ordinary & ~U64(0x8080808080808080)) << U64(9)) & enemy
        while right:
            target = lsb(right)
            right &= right - ONE
            moves[count] = encode(target - 9, target, 0, NORMAL)
            count += 1
        if not captures_only:
            single = (ordinary << U64(8)) & empty
            double = ((single & RANK_2_BB << U64(8)) << U64(8)) & empty
            while single:
                target = lsb(single)
                single &= single - ONE
                moves[count] = encode(target - 8, target, 0, NORMAL)
                count += 1
            while double:
                target = lsb(double)
                double &= double - ONE
                moves[count] = encode(target - 16, target, 0, NORMAL)
                count += 1
    else:
        promoting = pawns & RANK_2_BB
        ordinary = pawns ^ promoting
        if promoting:
            pushes = (promoting >> U64(8)) & empty
            while pushes:
                target = lsb(pushes)
                pushes &= pushes - ONE
                for piece in (QUEEN, ROOK, BISHOP, KNIGHT):
                    moves[count] = encode(target + 8, target, piece, NORMAL)
                    count += 1
            left = ((promoting & ~U64(0x8080808080808080)) >> U64(7)) & enemy
            while left:
                target = lsb(left)
                left &= left - ONE
                for piece in (QUEEN, ROOK, BISHOP, KNIGHT):
                    moves[count] = encode(target + 7, target, piece, NORMAL)
                    count += 1
            right = ((promoting & ~U64(0x0101010101010101)) >> U64(9)) & enemy
            while right:
                target = lsb(right)
                right &= right - ONE
                for piece in (QUEEN, ROOK, BISHOP, KNIGHT):
                    moves[count] = encode(target + 9, target, piece, NORMAL)
                    count += 1
        left = ((ordinary & ~U64(0x8080808080808080)) >> U64(7)) & enemy
        while left:
            target = lsb(left)
            left &= left - ONE
            moves[count] = encode(target + 7, target, 0, NORMAL)
            count += 1
        right = ((ordinary & ~U64(0x0101010101010101)) >> U64(9)) & enemy
        while right:
            target = lsb(right)
            right &= right - ONE
            moves[count] = encode(target + 9, target, 0, NORMAL)
            count += 1
        if not captures_only:
            single = (ordinary >> U64(8)) & empty
            double = ((single & RANK_7_BB >> U64(8)) >> U64(8)) & empty
            while single:
                target = lsb(single)
                single &= single - ONE
                moves[count] = encode(target + 8, target, 0, NORMAL)
                count += 1
            while double:
                target = lsb(double)
                double &= double - ONE
                moves[count] = encode(target + 16, target, 0, NORMAL)
                count += 1

    ep = st[ST_EP]
    if ep >= 0:
        capturers = PAWN_ATTACKS[1 - side, ep] & bb[base + PAWN]
        while capturers:
            origin = lsb(capturers)
            capturers &= capturers - ONE
            moves[count] = encode(origin, ep, 0, EP_CAPTURE)
            count += 1

    knights = bb[base + KNIGHT]
    while knights:
        origin = lsb(knights)
        knights &= knights - ONE
        attacks = KNIGHT_ATTACKS[origin] & targets
        while attacks:
            target = lsb(attacks)
            attacks &= attacks - ONE
            moves[count] = encode(origin, target, 0, NORMAL)
            count += 1

    bishops = bb[base + BISHOP]
    while bishops:
        origin = lsb(bishops)
        bishops &= bishops - ONE
        attacks = bishop_attacks(uint64(origin), occupancy) & targets
        while attacks:
            target = lsb(attacks)
            attacks &= attacks - ONE
            moves[count] = encode(origin, target, 0, NORMAL)
            count += 1

    rooks = bb[base + ROOK]
    while rooks:
        origin = lsb(rooks)
        rooks &= rooks - ONE
        attacks = rook_attacks(uint64(origin), occupancy) & targets
        while attacks:
            target = lsb(attacks)
            attacks &= attacks - ONE
            moves[count] = encode(origin, target, 0, NORMAL)
            count += 1

    queens = bb[base + QUEEN]
    while queens:
        origin = lsb(queens)
        queens &= queens - ONE
        attacks = queen_attacks(uint64(origin), occupancy) & targets
        while attacks:
            target = lsb(attacks)
            attacks &= attacks - ONE
            moves[count] = encode(origin, target, 0, NORMAL)
            count += 1

    kings = bb[base + KING]
    if kings:
        origin = lsb(kings)
        attacks = KING_ATTACKS[origin] & targets
        while attacks:
            target = lsb(attacks)
            attacks &= attacks - ONE
            moves[count] = encode(origin, target, 0, NORMAL)
            count += 1

        if not captures_only:
            rights = st[ST_CASTLE]
            enemy_side = 1 - side
            if side == WHITE:
                if (
                    rights & CR_WK
                    and not (occupancy & U64(0x60))
                    and not is_attacked(bb, 4, enemy_side)
                    and not is_attacked(bb, 5, enemy_side)
                ):
                    moves[count] = encode(4, 6, 0, CASTLING)
                    count += 1
                if (
                    rights & CR_WQ
                    and not (occupancy & U64(0xE))
                    and not is_attacked(bb, 4, enemy_side)
                    and not is_attacked(bb, 3, enemy_side)
                ):
                    moves[count] = encode(4, 2, 0, CASTLING)
                    count += 1
            else:
                if (
                    rights & CR_BK
                    and not (occupancy & U64(0x6000000000000000))
                    and not is_attacked(bb, 60, enemy_side)
                    and not is_attacked(bb, 61, enemy_side)
                ):
                    moves[count] = encode(60, 62, 0, CASTLING)
                    count += 1
                if (
                    rights & CR_BQ
                    and not (occupancy & U64(0x0E00000000000000))
                    and not is_attacked(bb, 60, enemy_side)
                    and not is_attacked(bb, 59, enemy_side)
                ):
                    moves[count] = encode(60, 58, 0, CASTLING)
                    count += 1

    return count


@njit(inline="always", cache=False)
def _put(bb: np.ndarray, mb: np.ndarray, key: np.ndarray, piece: int, square: int) -> None:
    bit = ONE << uint64(square)
    bb[piece] |= bit
    bb[OCC_W + piece // 6] |= bit
    bb[OCC_ALL] |= bit
    mb[square] = piece
    key[0] ^= ZOBRIST_PIECE[piece, square]


@njit(inline="always", cache=False)
def _take(bb: np.ndarray, mb: np.ndarray, key: np.ndarray, piece: int, square: int) -> None:
    bit = ONE << uint64(square)
    bb[piece] ^= bit
    bb[OCC_W + piece // 6] ^= bit
    bb[OCC_ALL] ^= bit
    mb[square] = EMPTY
    key[0] ^= ZOBRIST_PIECE[piece, square]


@njit(cache=False)
def make_move(
    bb: np.ndarray,
    mb: np.ndarray,
    st: np.ndarray,
    key: np.ndarray,
    undo: np.ndarray,
    ply: int,
    move: np.int32,
) -> None:
    origin = move_from(move)
    target = move_to(move)
    promotion = move_promotion(move)
    special = move_special(move)
    side = st[ST_SIDE]
    piece = mb[origin]
    captured = mb[target]

    undo[ply, 0] = captured
    undo[ply, 1] = st[ST_CASTLE]
    undo[ply, 2] = st[ST_EP]
    undo[ply, 3] = st[ST_HALF]

    if st[ST_EP] >= 0:
        key[0] ^= ZOBRIST_EP[st[ST_EP] & 7]
    key[0] ^= ZOBRIST_CASTLE[st[ST_CASTLE]]

    st[ST_HALF] += 1
    st[ST_EP] = -1

    if captured != EMPTY:
        _take(bb, mb, key, captured, target)
        st[ST_HALF] = 0

    _take(bb, mb, key, piece, origin)
    if promotion:
        _put(bb, mb, key, side * 6 + promotion, target)
        st[ST_HALF] = 0
    else:
        _put(bb, mb, key, piece, target)

    kind = piece - side * 6
    if kind == PAWN:
        st[ST_HALF] = 0
        if special == EP_CAPTURE:
            victim = target - 8 if side == WHITE else target + 8
            _take(bb, mb, key, (1 - side) * 6 + PAWN, victim)
        elif target - origin == 16 or origin - target == 16:
            middle = (origin + target) // 2
            # Only advertise the square when a pawn can actually take there, so positions that
            # play the same also hash the same.
            if PAWN_ATTACKS[side, middle] & bb[(1 - side) * 6 + PAWN]:
                st[ST_EP] = middle
                key[0] ^= ZOBRIST_EP[middle & 7]
    elif special == CASTLING:
        if target == 6:
            _take(bb, mb, key, ROOK, 7)
            _put(bb, mb, key, ROOK, 5)
        elif target == 2:
            _take(bb, mb, key, ROOK, 0)
            _put(bb, mb, key, ROOK, 3)
        elif target == 62:
            _take(bb, mb, key, 6 + ROOK, 63)
            _put(bb, mb, key, 6 + ROOK, 61)
        else:
            _take(bb, mb, key, 6 + ROOK, 56)
            _put(bb, mb, key, 6 + ROOK, 59)

    st[ST_CASTLE] &= CASTLE_MASK[origin] & CASTLE_MASK[target]
    key[0] ^= ZOBRIST_CASTLE[st[ST_CASTLE]]
    st[ST_SIDE] = 1 - side
    key[0] ^= ZOBRIST_SIDE


@njit(cache=False)
def unmake_move(
    bb: np.ndarray,
    mb: np.ndarray,
    st: np.ndarray,
    key: np.ndarray,
    undo: np.ndarray,
    ply: int,
    move: np.int32,
) -> None:
    origin = move_from(move)
    target = move_to(move)
    promotion = move_promotion(move)
    special = move_special(move)
    side = 1 - st[ST_SIDE]
    captured = undo[ply, 0]

    key[0] ^= ZOBRIST_SIDE
    key[0] ^= ZOBRIST_CASTLE[st[ST_CASTLE]]
    if st[ST_EP] >= 0:
        key[0] ^= ZOBRIST_EP[st[ST_EP] & 7]

    if special == CASTLING:
        if target == 6:
            _take(bb, mb, key, ROOK, 5)
            _put(bb, mb, key, ROOK, 7)
        elif target == 2:
            _take(bb, mb, key, ROOK, 3)
            _put(bb, mb, key, ROOK, 0)
        elif target == 62:
            _take(bb, mb, key, 6 + ROOK, 61)
            _put(bb, mb, key, 6 + ROOK, 63)
        else:
            _take(bb, mb, key, 6 + ROOK, 59)
            _put(bb, mb, key, 6 + ROOK, 56)

    if promotion:
        _take(bb, mb, key, side * 6 + promotion, target)
        _put(bb, mb, key, side * 6 + PAWN, origin)
    else:
        piece = mb[target]
        _take(bb, mb, key, piece, target)
        _put(bb, mb, key, piece, origin)

    if special == EP_CAPTURE:
        victim = target - 8 if side == WHITE else target + 8
        _put(bb, mb, key, (1 - side) * 6 + PAWN, victim)
    elif captured != EMPTY:
        _put(bb, mb, key, captured, target)

    st[ST_SIDE] = side
    st[ST_CASTLE] = undo[ply, 1]
    st[ST_EP] = undo[ply, 2]
    st[ST_HALF] = undo[ply, 3]
    key[0] ^= ZOBRIST_CASTLE[st[ST_CASTLE]]
    if st[ST_EP] >= 0:
        key[0] ^= ZOBRIST_EP[st[ST_EP] & 7]


@njit(cache=False)
def make_null(st: np.ndarray, key: np.ndarray, undo: np.ndarray, ply: int) -> None:
    undo[ply, 0] = EMPTY
    undo[ply, 1] = st[ST_CASTLE]
    undo[ply, 2] = st[ST_EP]
    undo[ply, 3] = st[ST_HALF]
    if st[ST_EP] >= 0:
        key[0] ^= ZOBRIST_EP[st[ST_EP] & 7]
    st[ST_EP] = -1
    st[ST_HALF] += 1
    st[ST_SIDE] = 1 - st[ST_SIDE]
    key[0] ^= ZOBRIST_SIDE


@njit(cache=False)
def unmake_null(st: np.ndarray, key: np.ndarray, undo: np.ndarray, ply: int) -> None:
    st[ST_SIDE] = 1 - st[ST_SIDE]
    key[0] ^= ZOBRIST_SIDE
    st[ST_CASTLE] = undo[ply, 1]
    st[ST_EP] = undo[ply, 2]
    st[ST_HALF] = undo[ply, 3]
    if st[ST_EP] >= 0:
        key[0] ^= ZOBRIST_EP[st[ST_EP] & 7]


@njit(uint64(uint64[::1], int64[::1]), cache=False)
def compute_key(bb: np.ndarray, st: np.ndarray) -> np.uint64:
    key = ZERO
    for piece in range(12):
        pieces = bb[piece]
        while pieces:
            square = lsb(pieces)
            pieces &= pieces - ONE
            key ^= ZOBRIST_PIECE[piece, square]
    key ^= ZOBRIST_CASTLE[st[ST_CASTLE]]
    if st[ST_EP] >= 0:
        key ^= ZOBRIST_EP[st[ST_EP] & 7]
    if st[ST_SIDE] != WHITE:
        key ^= ZOBRIST_SIDE
    return key
