"""FEN parsing. Plain Python: it runs once per move, not once per node."""

import numpy as np

from bitboards import (
    CR_BK,
    CR_BQ,
    CR_WK,
    CR_WQ,
    EMPTY,
    NBB,
    NST,
    OCC_ALL,
    OCC_W,
    PAWN_ATTACKS,
    ST_CASTLE,
    ST_EP,
    ST_HALF,
    ST_SIDE,
    WHITE,
)
from position import compute_key

PIECE_CHARS = "PNBRQKpnbrqk"
PROMOTION_CHARS = " nbrq"


def parse(fen: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """Return (bb, mb, st, key, fullmove) for a FEN string."""
    parts = fen.split()
    placement, side, castling, ep = parts[0], parts[1], parts[2], parts[3]
    halfmove = int(parts[4]) if len(parts) > 4 else 0
    fullmove = int(parts[5]) if len(parts) > 5 else 1

    bb = np.zeros(NBB, dtype=np.uint64)
    mb = np.full(64, EMPTY, dtype=np.int8)
    square = 56
    for char in placement:
        if char == "/":
            square -= 16
        elif char.isdigit():
            square += int(char)
        else:
            piece = PIECE_CHARS.index(char)
            bb[piece] |= np.uint64(1) << np.uint64(square)
            mb[square] = piece
            square += 1

    for piece in range(12):
        bb[OCC_W + piece // 6] |= bb[piece]
    bb[OCC_ALL] = bb[OCC_W] | bb[OCC_W + 1]

    st = np.zeros(NST, dtype=np.int64)
    st[ST_SIDE] = WHITE if side == "w" else 1 - WHITE
    rights = 0
    if "K" in castling:
        rights |= CR_WK
    if "Q" in castling:
        rights |= CR_WQ
    if "k" in castling:
        rights |= CR_BK
    if "q" in castling:
        rights |= CR_BQ
    st[ST_CASTLE] = rights
    st[ST_HALF] = halfmove

    st[ST_EP] = -1
    if ep != "-":
        target = (ord(ep[0]) - 97) + 8 * (int(ep[1]) - 1)
        mover = st[ST_SIDE]
        # Record it only when a capture actually exists, matching what make_move stores, so the
        # same playable position always hashes the same.
        #
        # The pawns that can capture on `target` belong to the side to move, and they stand on
        # the squares from which one of their pawns attacks `target` — which is the *opponent's*
        # attack mask read from `target`, because pawn attacks are only symmetric that way
        # round. Getting this backwards made the engine blind to every en passant capture
        # available on the move it was actually being asked about, and in the rare position
        # where en passant is the only legal move it produced no move at all.
        if PAWN_ATTACKS[1 - mover, target] & bb[mover * 6]:
            st[ST_EP] = target

    key = np.zeros(1, dtype=np.uint64)
    key[0] = compute_key(bb, st)
    return bb, mb, st, key, fullmove


def to_uci(move: int) -> str:
    origin = move & 63
    target = (move >> 6) & 63
    promotion = (move >> 12) & 7
    text = _square_name(origin) + _square_name(target)
    if promotion:
        text += PROMOTION_CHARS[promotion]
    return text


def _square_name(square: int) -> str:
    return chr(97 + (square & 7)) + str(1 + (square >> 3))


def from_uci(text: str) -> tuple[int, int, int]:
    origin = (ord(text[0]) - 97) + 8 * (int(text[1]) - 1)
    target = (ord(text[2]) - 97) + 8 * (int(text[3]) - 1)
    promotion = PROMOTION_CHARS.index(text[4]) if len(text) > 4 else 0
    return origin, target, promotion
