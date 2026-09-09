"""Find magic multipliers for rook and bishop sliding attacks and emit them as literals."""

import numpy as np
from numba import njit

M64 = np.uint64(0xFFFFFFFFFFFFFFFF)


@njit(cache=False)
def ray_attacks(sq: int, occ: np.uint64, deltas: np.ndarray) -> np.uint64:
    attacks = np.uint64(0)
    r0 = sq >> 3
    f0 = sq & 7
    for i in range(deltas.shape[0]):
        dr = deltas[i, 0]
        df = deltas[i, 1]
        r = r0 + dr
        f = f0 + df
        while 0 <= r < 8 and 0 <= f < 8:
            s = r * 8 + f
            bit = np.uint64(1) << np.uint64(s)
            attacks |= bit
            if occ & bit:
                break
            r += dr
            f += df
    return attacks


ROOK_DELTAS = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]], dtype=np.int64)
BISHOP_DELTAS = np.array([[1, 1], [1, -1], [-1, 1], [-1, -1]], dtype=np.int64)


def relevant_mask(sq: int, deltas: np.ndarray) -> np.uint64:
    """Attack set ignoring blockers, minus the edges that cannot matter."""
    mask = np.uint64(0)
    r0, f0 = sq >> 3, sq & 7
    for dr, df in deltas:
        r, f = r0 + dr, f0 + df
        while 0 <= r < 8 and 0 <= f < 8:
            nr, nf = r + dr, f + df
            if 0 <= nr < 8 and 0 <= nf < 8:
                mask |= np.uint64(1) << np.uint64(r * 8 + f)
            r, f = nr, nf
    return mask


def subsets(mask: np.uint64) -> list[np.uint64]:
    bits = [np.uint64(1) << np.uint64(i) for i in range(64) if mask & (np.uint64(1) << np.uint64(i))]
    out = []
    for index in range(1 << len(bits)):
        sub = np.uint64(0)
        for i, bit in enumerate(bits):
            if index & (1 << i):
                sub |= bit
        out.append(sub)
    return out


@njit(cache=False)
def try_magic(
    magic: np.uint64, shift: np.uint64, occs: np.ndarray, atts: np.ndarray, table: np.ndarray
) -> bool:
    table[:] = np.uint64(0)
    used = np.zeros(table.shape[0], dtype=np.uint8)
    for i in range(occs.shape[0]):
        index = (occs[i] * magic) >> shift
        if used[index] == 0:
            used[index] = 1
            table[index] = atts[i]
        elif table[index] != atts[i]:
            return False
    return True


def find(sq: int, deltas: np.ndarray, rng: np.random.Generator) -> tuple[int, int]:
    mask = relevant_mask(sq, deltas)
    bits = int(bin(int(mask)).count("1"))
    occs = np.array(subsets(mask), dtype=np.uint64)
    atts = np.array([ray_attacks(sq, o, deltas) for o in occs], dtype=np.uint64)
    shift = np.uint64(64 - bits)
    table = np.zeros(1 << bits, dtype=np.uint64)
    while True:
        candidate = np.uint64(
            int(rng.integers(0, 1 << 63, dtype=np.int64))
            & int(rng.integers(0, 1 << 63, dtype=np.int64))
            & int(rng.integers(0, 1 << 63, dtype=np.int64))
        )
        if bin(int((mask * candidate) & np.uint64(0xFF00000000000000))).count("1") < 6:
            continue
        if try_magic(candidate, shift, occs, atts, table):
            return int(candidate), bits


def main() -> None:
    rng = np.random.default_rng(20260908)
    for name, deltas in (("ROOK", ROOK_DELTAS), ("BISHOP", BISHOP_DELTAS)):
        magics = []
        bitcounts = []
        for sq in range(64):
            magic, bits = find(sq, deltas, rng)
            magics.append(magic)
            bitcounts.append(bits)
        print(f"{name}_MAGICS = (")
        for i in range(0, 64, 4):
            row = ", ".join(f"0x{m:016X}" for m in magics[i : i + 4])
            print(f"    {row},")
        print(")")
        print(f"{name}_BITS = {tuple(bitcounts)}")
        print()


if __name__ == "__main__":
    main()
