"""Bitboard constants and attack tables.

Everything here is built once at import time and then read as a numba global, which numba
freezes into the compiled code as a read-only array. Nothing in this module is written to
after import.

Square 0 is a1 and square 63 is h8, matching python-chess, so a UCI string maps straight onto
an index without a translation layer.
"""

import numpy as np
from numba import njit, uint64

U64 = np.uint64
ONE = U64(1)
ZERO = U64(0)
FULL = U64(0xFFFFFFFFFFFFFFFF)

# Piece types.
PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING = 0, 1, 2, 3, 4, 5
# Colours.
WHITE, BLACK = 0, 1
# Colour-tagged piece codes: white occupies 0..5, black 6..11, and 12 means an empty square.
EMPTY = 12

# Indices into the bitboard array.
OCC_W, OCC_B, OCC_ALL = 12, 13, 14
NBB = 15

# Castling right bits.
CR_WK, CR_WQ, CR_BK, CR_BQ = 1, 2, 4, 8

# Indices into the state array.
ST_SIDE, ST_CASTLE, ST_EP, ST_HALF = 0, 1, 2, 3
NST = 4

MAX_PLY = 128
MAX_MOVES = 256

FILE_A = U64(0x0101010101010101)
FILE_H = U64(0x8080808080808080)
RANK_1 = U64(0x00000000000000FF)
RANK_8 = U64(0xFF00000000000000)

FILES = np.array([FILE_A << U64(f) for f in range(8)], dtype=np.uint64)
RANKS = np.array([RANK_1 << U64(8 * r) for r in range(8)], dtype=np.uint64)


def _bit(square: int) -> np.uint64:
    return ONE << U64(square)


def _shift(bb: int, dr: int, df: int) -> int:
    """Shift a bitboard by a rank/file delta, dropping anything that falls off the board."""
    out = 0
    for square in range(64):
        if not bb & (1 << square):
            continue
        rank, file = divmod(square, 8)
        rank += dr
        file += df
        if 0 <= rank < 8 and 0 <= file < 8:
            out |= 1 << (rank * 8 + file)
    return out


def _leaper_table(deltas: tuple[tuple[int, int], ...]) -> np.ndarray:
    table = np.zeros(64, dtype=np.uint64)
    for square in range(64):
        rank, file = divmod(square, 8)
        acc = 0
        for dr, df in deltas:
            r, f = rank + dr, file + df
            if 0 <= r < 8 and 0 <= f < 8:
                acc |= 1 << (r * 8 + f)
        table[square] = U64(acc)
    return table


KNIGHT_ATTACKS = _leaper_table(
    ((2, 1), (2, -1), (-2, 1), (-2, -1), (1, 2), (1, -2), (-1, 2), (-1, -2))
)
KING_ATTACKS = _leaper_table(
    ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1))
)

_PAWN = np.zeros((2, 64), dtype=np.uint64)
for _sq in range(64):
    _r, _f = divmod(_sq, 8)
    _w = _b = 0
    for _df in (-1, 1):
        if 0 <= _f + _df < 8:
            if _r + 1 < 8:
                _w |= 1 << ((_r + 1) * 8 + _f + _df)
            if _r - 1 >= 0:
                _b |= 1 << ((_r - 1) * 8 + _f + _df)
    _PAWN[WHITE, _sq] = U64(_w)
    _PAWN[BLACK, _sq] = U64(_b)
PAWN_ATTACKS = _PAWN

ROOK_DELTAS = ((1, 0), (-1, 0), (0, 1), (0, -1))
BISHOP_DELTAS = ((1, 1), (1, -1), (-1, 1), (-1, -1))


def _slider_attacks(square: int, occupancy: int, deltas: tuple[tuple[int, int], ...]) -> int:
    rank, file = divmod(square, 8)
    acc = 0
    for dr, df in deltas:
        r, f = rank + dr, file + df
        while 0 <= r < 8 and 0 <= f < 8:
            target = r * 8 + f
            acc |= 1 << target
            if occupancy & (1 << target):
                break
            r += dr
            f += df
    return acc


def _relevant_mask(square: int, deltas: tuple[tuple[int, int], ...]) -> int:
    """The squares whose occupancy changes the attack set: the ray minus its final square."""
    rank, file = divmod(square, 8)
    acc = 0
    for dr, df in deltas:
        r, f = rank + dr, file + df
        while 0 <= r < 8 and 0 <= f < 8:
            nr, nf = r + dr, f + df
            if 0 <= nr < 8 and 0 <= nf < 8:
                acc |= 1 << (r * 8 + f)
            r, f = nr, nf
    return acc


# Magic multipliers found by docs/gen_magics.py. A magic maps the masked occupancy onto a dense
# index, so a sliding attack set is one multiply, one shift and one lookup.
ROOK_MAGICS = (
    0x0080002080104000, 0x00C0200010004004, 0x0100200009001042, 0x0E00060020084011,
    0x1480040008000280, 0x0B00020824000100, 0x0880010002001080, 0x22000C0100805022,
    0x2202800140008120, 0x0212401000200044, 0x1002004082001020, 0x0001001000240900,
    0x0100808008000400, 0x0020808004000200, 0x2009000100042200, 0x0001000850820900,
    0x0000888004400428, 0x4000808040002000, 0x091D010040200211, 0x0840808010000804,
    0x0008818008000400, 0x0840808004000200, 0x0300440042010890, 0x0000860000E28904,
    0x0022008200204100, 0x4080200240100040, 0x0020008080100020, 0x4084401200082203,
    0x3020050100100800, 0x0288020080800400, 0x0802014400108802, 0x000002420006A114,
    0x0080002002400041, 0x4240081000200020, 0x0C20001021004102, 0x0100080080801000,
    0x0000080005001100, 0x4A04008044800200, 0x0110010204000810, 0x0024210842001184,
    0x0088400028808003, 0x001000402000C008, 0x1106410020050010, 0x0808008010008008,
    0x4000080004008080, 0x1882000468520030, 0x0000100188040002, 0x0004008049020014,
    0x000102A081560200, 0x0000802100400100, 0x0000100020008080, 0x0000801000080080,
    0x0000040048008280, 0x0002000488100E00, 0x0004080102108400, 0x1059000040820100,
    0x04A6410020800991, 0x0401844200122102, 0x0808082001024211, 0x0008040810002101,
    0x0002009024202902, 0x0026000884411022, 0x2110409001080224, 0x2480C40041008832,
)

BISHOP_MAGICS = (
    0x0062080144008600, 0x4128120840410020, 0x00080091020000A0, 0x1234124200000140,
    0x1001104004000040, 0x0042482004102082, 0x000C02A23031121A, 0x2424802401200800,
    0x10000510124A0C00, 0x1040082888208020, 0x2080080801082000, 0x4004885841000801,
    0x40C0020210400400, 0x0880809220200100, 0x2020010450040440, 0x0000C20082211000,
    0x00858020201E4200, 0x4442442144010200, 0x0001080802082200, 0x022100102C008000,
    0x1206020400940902, 0x002A001101010100, 0x0422100888210820, 0x00A024510482106A,
    0x1804400110C20801, 0x0002900060242880, 0x4208040228082164, 0x0200808008020002,
    0x40A0840000802020, 0x40021201C2209000, 0x0B08820041111004, 0x40020082504400A0,
    0x0008044080040800, 0x0110822000100480, 0x0120124808040800, 0x0200020080080080,
    0x0830060080101004, 0x0824081020021000, 0x400A821400804412, 0x4800940100909088,
    0x0004010410A24010, 0x08004210A4411010, 0x0039002504044041, 0x004A02C202212800,
    0x4098085100440400, 0x0420200040402484, 0x0010108200800040, 0x080A04820A803200,
    0x0084020802088002, 0x0050290410040400, 0x00001908A0900000, 0x1040038084040240,
    0x48010030620A0202, 0x0148040810810018, 0x3050030248020040, 0x0020011D42008300,
    0x0400210800A42060, 0x3000020084010904, 0x0084000201008808, 0x4180001000460800,
    0x4101044104050400, 0x3888202411464200, 0x01500802D0340500, 0x1802200444004040,
)


def _build_magic_tables(
    magics: tuple[int, ...], deltas: tuple[tuple[int, int], ...]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    masks = np.zeros(64, dtype=np.uint64)
    shifts = np.zeros(64, dtype=np.uint64)
    offsets = np.zeros(64, dtype=np.int64)
    magic_array = np.array(magics, dtype=np.uint64)
    total = 0
    for square in range(64):
        mask = _relevant_mask(square, deltas)
        masks[square] = U64(mask)
        bits = bin(mask).count("1")
        shifts[square] = U64(64 - bits)
        offsets[square] = total
        total += 1 << bits
    table = np.zeros(total, dtype=np.uint64)
    for square in range(64):
        mask = int(masks[square])
        magic = magics[square]
        shift = int(shifts[square])
        base = int(offsets[square])
        bits = bin(mask).count("1")
        squares = [i for i in range(64) if mask & (1 << i)]
        for index in range(1 << bits):
            occupancy = 0
            for i, sq in enumerate(squares):
                if index & (1 << i):
                    occupancy |= 1 << sq
            slot = ((occupancy * magic) & 0xFFFFFFFFFFFFFFFF) >> shift
            table[base + slot] = U64(_slider_attacks(square, occupancy, deltas))
    return masks, shifts, offsets, magic_array, table


ROOK_MASKS, ROOK_SHIFTS, ROOK_OFFSETS, ROOK_MAGIC, ROOK_TABLE = _build_magic_tables(
    ROOK_MAGICS, ROOK_DELTAS
)
BISHOP_MASKS, BISHOP_SHIFTS, BISHOP_OFFSETS, BISHOP_MAGIC, BISHOP_TABLE = _build_magic_tables(
    BISHOP_MAGICS, BISHOP_DELTAS
)

# BETWEEN[a][b] is the open segment between two aligned squares, empty when they do not align.
_between = np.zeros((64, 64), dtype=np.uint64)
_line = np.zeros((64, 64), dtype=np.uint64)
for _a in range(64):
    for _deltas in (ROOK_DELTAS, BISHOP_DELTAS):
        for _dr, _df in _deltas:
            _r, _f = divmod(_a, 8)
            _path = 0
            _r += _dr
            _f += _df
            while 0 <= _r < 8 and 0 <= _f < 8:
                _b = _r * 8 + _f
                _between[_a, _b] = U64(_path)
                _path |= 1 << _b
                _r += _dr
                _f += _df
    for _b in range(64):
        _ra, _fa = divmod(_a, 8)
        _rb, _fb = divmod(_b, 8)
        if _a == _b:
            continue
        if _ra == _rb:
            _line[_a, _b] = RANKS[_ra]
        elif _fa == _fb:
            _line[_a, _b] = FILES[_fa]
        elif _ra - _fa == _rb - _fb or _ra + _fa == _rb + _fb:
            _diag = 0
            for _dr, _df in BISHOP_DELTAS:
                _r, _f = _ra, _fa
                while 0 <= _r < 8 and 0 <= _f < 8:
                    _diag |= 1 << (_r * 8 + _f)
                    _r += _dr
                    _f += _df
            if _diag & (1 << _b):
                _line[_a, _b] = U64(_diag)
BETWEEN = _between
LINE = _line

# Pawn structure helpers.
_adjacent = np.zeros(8, dtype=np.uint64)
for _f in range(8):
    _acc = 0
    if _f > 0:
        _acc |= int(FILES[_f - 1])
    if _f < 7:
        _acc |= int(FILES[_f + 1])
    _adjacent[_f] = U64(_acc)
ADJACENT_FILES = _adjacent

_front = np.zeros((2, 64), dtype=np.uint64)
_passed = np.zeros((2, 64), dtype=np.uint64)
for _sq in range(64):
    _r, _f = divmod(_sq, 8)
    _up = 0
    for _rr in range(_r + 1, 8):
        _up |= int(RANKS[_rr])
    _down = 0
    for _rr in range(0, _r):
        _down |= int(RANKS[_rr])
    _front[WHITE, _sq] = U64(_up & int(FILES[_f]))
    _front[BLACK, _sq] = U64(_down & int(FILES[_f]))
    _span = int(FILES[_f]) | int(ADJACENT_FILES[_f])
    _passed[WHITE, _sq] = U64(_up & _span)
    _passed[BLACK, _sq] = U64(_down & _span)
FRONT_SPAN = _front
PASSED_MASK = _passed

# The king's own square plus the ring around it, used by the king-safety term.
_king_zone = np.zeros((2, 64), dtype=np.uint64)
for _sq in range(64):
    _ring = int(KING_ATTACKS[_sq]) | (1 << _sq)
    _king_zone[WHITE, _sq] = U64(_ring | _shift(_ring, 1, 0))
    _king_zone[BLACK, _sq] = U64(_ring | _shift(_ring, -1, 0))
KING_ZONE = _king_zone

# Zobrist keys. A fixed seed keeps a game reproducible from a given start position.
_rng = np.random.default_rng(0x5EED_C4E5)


def _keys(*shape: int) -> np.ndarray:
    return _rng.integers(0, 1 << 64, size=shape, dtype=np.uint64)


ZOBRIST_PIECE = _keys(12, 64)
ZOBRIST_CASTLE = _keys(16)
ZOBRIST_EP = _keys(8)
ZOBRIST_SIDE = U64(_keys(1)[0])


@njit(uint64(uint64), inline="always", cache=False)
def popcount(bb: np.uint64) -> np.uint64:
    """Number of set bits, via the SWAR trick that numba compiles to a handful of instructions."""
    x = bb - ((bb >> uint64(1)) & uint64(0x5555555555555555))
    x = (x & uint64(0x3333333333333333)) + ((x >> uint64(2)) & uint64(0x3333333333333333))
    x = (x + (x >> uint64(4))) & uint64(0x0F0F0F0F0F0F0F0F)
    return (x * uint64(0x0101010101010101)) >> uint64(56)


_DEBRUIJN = uint64(0x03F79D71B4CB0A89)
_DEBRUIJN_INDEX = np.zeros(64, dtype=np.int64)
for _i in range(64):
    _DEBRUIJN_INDEX[int((int(_DEBRUIJN) * (1 << _i)) % (1 << 64) >> 58)] = _i
DEBRUIJN_INDEX = _DEBRUIJN_INDEX


@njit(inline="always", cache=False)
def lsb(bb: np.uint64) -> int:
    """Index of the least significant set bit.

    Undefined for an empty board, which is never passed one.
    """
    return int(DEBRUIJN_INDEX[((bb & (uint64(0) - bb)) * _DEBRUIJN) >> uint64(58)])


@njit(inline="always", cache=False)
def pop_lsb(bb: np.uint64) -> tuple[int, np.uint64]:
    square = lsb(bb)
    return square, bb & (bb - uint64(1))


@njit(uint64(uint64, uint64), inline="always", cache=False)
def rook_attacks(square: np.uint64, occupancy: np.uint64) -> np.uint64:
    sq = np.int64(square)
    index = ((occupancy & ROOK_MASKS[sq]) * ROOK_MAGIC[sq]) >> ROOK_SHIFTS[sq]
    return ROOK_TABLE[ROOK_OFFSETS[sq] + np.int64(index)]


@njit(uint64(uint64, uint64), inline="always", cache=False)
def bishop_attacks(square: np.uint64, occupancy: np.uint64) -> np.uint64:
    sq = np.int64(square)
    index = ((occupancy & BISHOP_MASKS[sq]) * BISHOP_MAGIC[sq]) >> BISHOP_SHIFTS[sq]
    return BISHOP_TABLE[BISHOP_OFFSETS[sq] + np.int64(index)]


@njit(uint64(uint64, uint64), inline="always", cache=False)
def queen_attacks(square: np.uint64, occupancy: np.uint64) -> np.uint64:
    return rook_attacks(square, occupancy) | bishop_attacks(square, occupancy)
