"""A tapered hand-crafted evaluation.

Every term is computed twice, once with midgame weights and once with endgame weights, and the
two are blended by a phase counted from the non-pawn material left on the board. That is what
lets one function say "keep the king behind pawns" in the opening and "walk the king to the
centre" in a rook ending without contradicting itself.

Scores are in centipawns and always returned from the side to move's point of view.
"""

import numpy as np
from numba import int64, njit, uint64

from bitboards import (
    ADJACENT_FILES,
    BISHOP,
    FILES,
    FRONT_SPAN,
    KING,
    KING_ZONE,
    KNIGHT,
    KNIGHT_ATTACKS,
    OCC_ALL,
    OCC_W,
    PASSED_MASK,
    PAWN,
    PAWN_ATTACKS,
    QUEEN,
    ROOK,
    ST_SIDE,
    WHITE,
    bishop_attacks,
    lsb,
    popcount,
    queen_attacks,
    rook_attacks,
)

U64 = np.uint64
ONE = U64(1)
ZERO = U64(0)

# Piece values, midgame and endgame. The king is scored only by its table.
MG_VALUE = (82, 337, 365, 477, 1025, 0)
EG_VALUE = (94, 281, 297, 512, 936, 0)

# How much of the phase each piece contributes. Pawns contribute nothing, so trading pawns does
# not push the game toward the endgame weights on its own.
PHASE_INC = (0, 1, 1, 2, 4, 0)
PHASE_MAX = 24

# fmt: off
# Piece-square tables, written the way a board looks: the first row is rank 8. White reads them
# through sq ^ 56 and black reads them directly.
_MG_PAWN = (
      0,   0,   0,   0,   0,   0,   0,   0,
     98, 134,  61,  95,  68, 126,  34, -11,
     -6,   7,  26,  31,  65,  56,  25, -20,
    -14,  13,   6,  21,  23,  12,  17, -23,
    -27,  -2,  -5,  12,  17,   6,  10, -25,
    -26,  -4,  -4, -10,   3,   3,  33, -12,
    -35,  -1, -20, -23, -15,  24,  38, -22,
      0,   0,   0,   0,   0,   0,   0,   0,
)
_EG_PAWN = (
      0,   0,   0,   0,   0,   0,   0,   0,
    178, 173, 158, 134, 147, 132, 165, 187,
     94, 100,  85,  67,  56,  53,  82,  84,
     32,  24,  13,   5,  -2,   4,  17,  17,
     13,   9,  -3,  -7,  -7,  -8,   3,  -1,
      4,   7,  -6,   1,   0,  -5,  -1,  -8,
     13,   8,   8,  10,  13,   0,   2,  -7,
      0,   0,   0,   0,   0,   0,   0,   0,
)
_MG_KNIGHT = (
   -167, -89, -34, -49,  61, -97, -15, -107,
    -73, -41,  72,  36,  23,  62,   7,  -17,
    -47,  60,  37,  65,  84, 129,  73,   44,
     -9,  17,  19,  53,  37,  69,  18,   22,
    -13,   4,  16,  13,  28,  19,  21,   -8,
    -23,  -9,  12,  10,  19,  17,  25,  -16,
    -29, -53, -12,  -3,  -1,  18, -14,  -19,
   -105, -21, -58, -33, -17, -28, -19,  -23,
)
_EG_KNIGHT = (
    -58, -38, -13, -28, -31, -27, -63, -99,
    -25,  -8, -25,  -2,  -9, -25, -24, -52,
    -24, -20,  10,   9,  -1,  -9, -19, -41,
    -17,   3,  22,  22,  22,  11,   8, -18,
    -18,  -6,  16,  25,  16,  17,   4, -18,
    -23,  -3,  -1,  15,  10,  -3, -20, -22,
    -42, -20, -10,  -5,  -2, -20, -23, -44,
    -29, -51, -23, -15, -22, -18, -50, -64,
)
_MG_BISHOP = (
    -29,   4, -82, -37, -25, -42,   7,  -8,
    -26,  16, -18, -13,  30,  59,  18, -47,
    -16,  37,  43,  40,  35,  50,  37,  -2,
     -4,   5,  19,  50,  37,  37,   7,  -2,
     -6,  13,  13,  26,  34,  12,  10,   4,
      0,  15,  15,  15,  14,  27,  18,  10,
      4,  15,  16,   0,   7,  21,  33,   1,
    -33,  -3, -14, -21, -13, -12, -39, -21,
)
_EG_BISHOP = (
    -14, -21, -11,  -8,  -7,  -9, -17, -24,
     -8,  -4,   7, -12,  -3, -13,  -4, -14,
      2,  -8,   0,  -1,  -2,   6,   0,   4,
     -3,   9,  12,   9,  14,  10,   3,   2,
     -6,   3,  13,  19,   7,  10,  -3,  -9,
    -12,  -3,   8,  10,  13,   3,  -7, -15,
    -14, -18,  -7,  -1,   4,  -9, -15, -27,
    -23,  -9, -23,  -5,  -9, -16,  -5, -17,
)
_MG_ROOK = (
     32,  42,  32,  51,  63,   9,  31,  43,
     27,  32,  58,  62,  80,  67,  26,  44,
     -5,  19,  26,  36,  17,  45,  61,  16,
    -24, -11,   7,  26,  24,  35,  -8, -20,
    -36, -26, -12,  -1,   9,  -7,   6, -23,
    -45, -25, -16, -17,   3,   0,  -5, -33,
    -44, -16, -20,  -9,  -1,  11,  -6, -71,
    -19, -13,   1,  17,  16,   7, -37, -26,
)
_EG_ROOK = (
     13,  10,  18,  15,  12,  12,   8,   5,
     11,  13,  13,  11,  -3,   3,   8,   3,
      7,   7,   7,   5,   4,  -3,  -5,  -3,
      4,   3,  13,   1,   2,   1,  -1,   2,
      3,   5,   8,   4,  -5,  -6,  -8, -11,
     -4,   0,  -5,  -1,  -7, -12,  -8, -16,
     -6,  -6,   0,   2,  -9,  -9, -11,  -3,
     -9,   2,   3,  -1,  -5, -13,   4, -20,
)
_MG_QUEEN = (
    -28,   0,  29,  12,  59,  44,  43,  45,
    -24, -39,  -5,   1, -16,  57,  28,  54,
    -13, -17,   7,   8,  29,  56,  47,  57,
    -27, -27, -16, -16,  -1,  17,  -2,   1,
     -9, -26,  -9, -10,  -2,  -4,   3,  -3,
    -14,   2, -11,  -2,  -5,   2,  14,   5,
    -35,  -8,  11,   2,   8,  15,  -3,   1,
     -1, -18,  -9,  10, -15, -25, -31, -50,
)
_EG_QUEEN = (
     -9,  22,  22,  27,  27,  19,  10,  20,
    -17,  20,  32,  41,  58,  25,  30,   0,
    -20,   6,   9,  49,  47,  35,  19,   9,
      3,  22,  24,  45,  57,  40,  57,  36,
    -18,  28,  19,  47,  31,  34,  39,  23,
    -16, -27,  15,   6,   9,  17,  10,   5,
    -22, -23, -30, -16, -16, -23, -36, -32,
    -33, -28, -22, -43,  -5, -32, -20, -41,
)
_MG_KING = (
    -65,  23,  16, -15, -56, -34,   2,  13,
     29,  -1, -20,  -7,  -8,  -4, -38, -29,
     -9,  24,   2, -16, -20,   6,  22, -22,
    -17, -20, -12, -27, -30, -25, -14, -36,
    -49,  -1, -27, -39, -46, -44, -33, -51,
    -14, -14, -22, -46, -44, -30, -15, -27,
      1,   7,  -8, -64, -43, -16,   9,   8,
    -15,  36,  12, -54,   8, -28,  24,  14,
)
_EG_KING = (
    -74, -35, -18, -18, -11,  15,   4, -17,
    -12,  17,  14,  17,  17,  38,  23,  11,
     10,  17,  23,  15,  20,  45,  44,  13,
     -8,  22,  24,  27,  26,  33,  26,   3,
    -18,  -4,  21,  24,  27,  23,   9, -11,
    -19,  -3,  11,  21,  23,  16,   7,  -9,
    -27, -11,   4,  13,  14,   4,  -5, -17,
    -53, -34, -21, -11, -28, -14, -24, -43,
)
# fmt: on

_MG_TABLES = (_MG_PAWN, _MG_KNIGHT, _MG_BISHOP, _MG_ROOK, _MG_QUEEN, _MG_KING)
_EG_TABLES = (_EG_PAWN, _EG_KNIGHT, _EG_BISHOP, _EG_ROOK, _EG_QUEEN, _EG_KING)


def _build(tables: tuple[tuple[int, ...], ...], values: tuple[int, ...]) -> np.ndarray:
    """Fold the piece value into the table and produce one array per colour-tagged piece."""
    out = np.zeros((12, 64), dtype=np.int64)
    for kind in range(6):
        for square in range(64):
            out[kind, square] = values[kind] + tables[kind][square ^ 56]
            out[6 + kind, square] = values[kind] + tables[kind][square]
    return out


MG_TABLE = _build(_MG_TABLES, MG_VALUE)
EG_TABLE = _build(_EG_TABLES, EG_VALUE)

PHASE_ARRAY = np.array(PHASE_INC, dtype=np.int64)
MG_VALUE_ARRAY = np.array(MG_VALUE, dtype=np.int64)
EG_VALUE_ARRAY = np.array(EG_VALUE, dtype=np.int64)

# Mobility bonuses indexed by piece type and by how many safe squares the piece reaches. The
# curves are flat-ish at the top because the twentieth square a queen sees is worth little.
def _mobility_curve(scale: float, peak: int, size: int) -> list[int]:
    return [int(round(scale * (min(i, peak) - peak / 2.0))) for i in range(size)]


_MOB_MG = np.zeros((6, 32), dtype=np.int64)
_MOB_EG = np.zeros((6, 32), dtype=np.int64)
for _i, (_s_mg, _s_eg, _peak) in enumerate(
    ((0.0, 0.0, 1), (5.0, 4.0, 8), (4.5, 4.5, 13), (2.5, 4.5, 14), (1.5, 3.0, 27), (0.0, 0.0, 1))
):
    _MOB_MG[_i] = np.array(_mobility_curve(_s_mg, _peak, 32), dtype=np.int64)
    _MOB_EG[_i] = np.array(_mobility_curve(_s_eg, _peak, 32), dtype=np.int64)
MOBILITY_MG = _MOB_MG
MOBILITY_EG = _MOB_EG

# Passed pawn bonus by how far the pawn has travelled, from its own side's point of view.
PASSED_MG = np.array([0, 2, 6, 14, 34, 68, 118, 0], dtype=np.int64)
PASSED_EG = np.array([0, 12, 22, 42, 78, 134, 208, 0], dtype=np.int64)

ISOLATED_MG, ISOLATED_EG = -14, -18
DOUBLED_MG, DOUBLED_EG = -10, -24
BACKWARD_MG, BACKWARD_EG = -8, -12
CONNECTED_MG, CONNECTED_EG = 8, 6
BISHOP_PAIR_MG, BISHOP_PAIR_EG = 28, 48
ROOK_OPEN_MG, ROOK_OPEN_EG = 32, 12
ROOK_SEMI_MG, ROOK_SEMI_EG = 14, 8
ROOK_SEVENTH_MG, ROOK_SEVENTH_EG = 12, 28
KNIGHT_OUTPOST_MG, KNIGHT_OUTPOST_EG = 22, 10
TEMPO = 14

# King safety converts "how much material is aiming at the king zone" into centipawns. The curve
# is quadratic at the bottom and saturates, which is the usual shape.
_KING_DANGER = np.zeros(80, dtype=np.int64)
for _i in range(80):
    _KING_DANGER[_i] = min(int(_i * _i * 0.55), 620)
KING_DANGER = _KING_DANGER
ATTACK_WEIGHT = np.array([0, 20, 20, 40, 80, 0], dtype=np.int64)

SHIELD_MISSING_MG = -18
OPEN_FILE_ON_KING_MG = -24


@njit(int64(uint64[::1], int64), inline="always", cache=False)
def _scale_drawish(bb: np.ndarray, score: int) -> int:
    if score > 0:
        strong = 0
    elif score < 0:
        strong = 1
    else:
        return score
    if bb[strong * 6 + PAWN]:
        return score
    base = strong * 6
    minors = int(popcount(bb[base + KNIGHT] | bb[base + BISHOP]))
    majors = int(popcount(bb[base + ROOK] | bb[base + QUEEN]))
    if majors == 0 and minors <= 2:
        # Two knights, or a lone minor, cannot force mate against a bare king.
        if minors <= 1 or not bb[base + BISHOP]:
            return score // 8
        return score // 2
    return score


@njit(int64(uint64[::1], int64[::1]), cache=False)
def evaluate(bb: np.ndarray, st: np.ndarray) -> int:
    occupancy = bb[OCC_ALL]
    mg = 0
    eg = 0
    phase = 0

    white_pawns = bb[PAWN]
    black_pawns = bb[6 + PAWN]
    white_pawn_attacks = ((white_pawns & ~FILES[0]) << U64(7)) | (
        (white_pawns & ~FILES[7]) << U64(9)
    )
    black_pawn_attacks = ((black_pawns & ~FILES[7]) >> U64(7)) | (
        (black_pawns & ~FILES[0]) >> U64(9)
    )
    pawn_attacks = (white_pawn_attacks, black_pawn_attacks)
    own_pawns_bb = (white_pawns, black_pawns)

    white_king_square = lsb(bb[KING])
    black_king_square = lsb(bb[6 + KING])
    king_squares = (white_king_square, black_king_square)
    king_zones = (KING_ZONE[0, white_king_square], KING_ZONE[1, black_king_square])

    danger = np.zeros(2, dtype=np.int64)
    attacker_count = np.zeros(2, dtype=np.int64)

    for side in range(2):
        sign = 1 if side == WHITE else -1
        base = side * 6
        own = bb[OCC_W + side]
        friendly_pawns = own_pawns_bb[side]
        enemy_pawns = own_pawns_bb[1 - side]
        enemy_pawn_attacks = pawn_attacks[1 - side]
        enemy_zone = king_zones[1 - side]
        safe = ~(own | enemy_pawn_attacks)

        for kind in range(6):
            pieces = bb[base + kind]
            while pieces:
                square = lsb(pieces)
                pieces &= pieces - ONE
                phase += PHASE_ARRAY[kind]
                mg += sign * MG_TABLE[base + kind, square]
                eg += sign * EG_TABLE[base + kind, square]

                if kind == PAWN or kind == KING:
                    continue

                # Bishops and rooks look through their own batteries, so a doubled rook is not
                # scored as if the rook in front of it were a wall.
                attacks = ZERO
                if kind == KNIGHT:
                    attacks = KNIGHT_ATTACKS[square]
                elif kind == BISHOP:
                    attacks = bishop_attacks(U64(square), occupancy ^ bb[base + QUEEN])
                elif kind == ROOK:
                    attacks = rook_attacks(
                        U64(square), occupancy ^ bb[base + ROOK] ^ bb[base + QUEEN]
                    )
                else:
                    attacks = queen_attacks(U64(square), occupancy)

                moves = int(popcount(attacks & safe))
                mg += sign * MOBILITY_MG[kind, moves]
                eg += sign * MOBILITY_EG[kind, moves]

                zone_hits = attacks & enemy_zone
                if zone_hits:
                    attacker_count[side] += 1
                    danger[side] += ATTACK_WEIGHT[kind] * int(popcount(zone_hits))

                if kind == ROOK:
                    file_index = square & 7
                    file_bb = FILES[file_index]
                    if not (file_bb & (friendly_pawns | enemy_pawns)):
                        mg += sign * ROOK_OPEN_MG
                        eg += sign * ROOK_OPEN_EG
                    elif not (file_bb & friendly_pawns):
                        mg += sign * ROOK_SEMI_MG
                        eg += sign * ROOK_SEMI_EG
                    relative_rank = (square >> 3) if side == WHITE else 7 - (square >> 3)
                    if relative_rank == 6:
                        mg += sign * ROOK_SEVENTH_MG
                        eg += sign * ROOK_SEVENTH_EG
                elif kind == KNIGHT:
                    relative_rank = (square >> 3) if side == WHITE else 7 - (square >> 3)
                    if (
                        3 <= relative_rank <= 5
                        and (PAWN_ATTACKS[1 - side, square] & friendly_pawns)
                        and not (PASSED_MASK[side, square] & ~FILES[square & 7] & enemy_pawns)
                    ):
                        mg += sign * KNIGHT_OUTPOST_MG
                        eg += sign * KNIGHT_OUTPOST_EG

        if popcount(bb[base + BISHOP]) >= U64(2):
            mg += sign * BISHOP_PAIR_MG
            eg += sign * BISHOP_PAIR_EG

        # Pawn structure.
        pawns = friendly_pawns
        while pawns:
            square = lsb(pawns)
            pawns &= pawns - ONE
            file_index = square & 7
            file_bb = FILES[file_index]
            neighbours = ADJACENT_FILES[file_index] & friendly_pawns
            if not neighbours:
                mg += sign * ISOLATED_MG
                eg += sign * ISOLATED_EG
            if FRONT_SPAN[side, square] & friendly_pawns:
                mg += sign * DOUBLED_MG
                eg += sign * DOUBLED_EG
            if PAWN_ATTACKS[1 - side, square] & friendly_pawns:
                mg += sign * CONNECTED_MG
                eg += sign * CONNECTED_EG
            if not (PASSED_MASK[side, square] & enemy_pawns) and not (
                FRONT_SPAN[side, square] & friendly_pawns
            ):
                relative_rank = (square >> 3) if side == WHITE else 7 - (square >> 3)
                mg += sign * PASSED_MG[relative_rank]
                eg += sign * PASSED_EG[relative_rank]
            elif neighbours and not (PAWN_ATTACKS[1 - side, square] & friendly_pawns):
                # No friendly pawn can ever defend it from behind on an adjacent file.
                if not (neighbours & ~FRONT_SPAN[side, square] & ~file_bb):
                    mg += sign * BACKWARD_MG
                    eg += sign * BACKWARD_EG

        # Pawn shelter in front of the king, and open files pointing at it.
        king_square = king_squares[side]
        king_file = king_square & 7
        start = king_file - 1 if king_file > 0 else 0
        stop = king_file + 1 if king_file < 7 else 7
        for f in range(start, stop + 1):
            file_bb = FILES[f]
            if not (file_bb & friendly_pawns):
                mg += sign * SHIELD_MISSING_MG
                if not (file_bb & enemy_pawns):
                    mg += sign * OPEN_FILE_ON_KING_MG

    for side in range(2):
        sign = 1 if side == WHITE else -1
        if attacker_count[side] >= 2:
            index = danger[side] // 12
            if index > 79:
                index = 79
            mg += sign * KING_DANGER[index]

    if phase > PHASE_MAX:
        phase = PHASE_MAX
    score = (mg * phase + eg * (PHASE_MAX - phase)) // PHASE_MAX
    score += TEMPO if st[ST_SIDE] == WHITE else -TEMPO

    # A side with no pawns and a small material edge cannot usually convert it.
    score = _scale_drawish(bb, score)

    return score if st[ST_SIDE] == WHITE else -score


@njit(int64(uint64[::1]), cache=False)
def game_phase(bb: np.ndarray) -> int:
    phase = 0
    for side in range(2):
        base = side * 6
        for kind in range(1, 5):
            phase += PHASE_ARRAY[kind] * int(popcount(bb[base + kind]))
    return phase if phase < PHASE_MAX else PHASE_MAX


@njit(int64(uint64[::1]), cache=False)
def insufficient_material(bb: np.ndarray) -> int:
    """True when neither side can deliver mate, which the referee scores as a draw."""
    if bb[PAWN] or bb[6 + PAWN] or bb[ROOK] or bb[6 + ROOK] or bb[QUEEN] or bb[6 + QUEEN]:
        return 0
    white_minors = int(popcount(bb[KNIGHT] | bb[BISHOP]))
    black_minors = int(popcount(bb[6 + KNIGHT] | bb[6 + BISHOP]))
    return 1 if white_minors <= 1 and black_minors <= 1 else 0

