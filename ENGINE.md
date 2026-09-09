# The engine

`agent.py` is the submission entry point. Everything under it is a conventional bitboard chess
engine written so that numba can compile it, which is the only reason the node counts are in
the millions rather than the thousands.

```
agent.py        entry point: FEN in, UCI move out, and the clock
bitboards.py    attack tables, magic bitboards, zobrist keys
position.py     board state, move encoding, move generation, make/unmake
evaluate.py     tapered hand-crafted evaluation
search.py       negamax with alpha-beta and the pruning that makes it worth running
fen.py          FEN parsing and UCI formatting
perft.py        move generation self-test, never imported by the agent
```

## Why it is shaped like this

A position is four plain arrays, not an object, because numba compiles free functions over
arrays far better than it compiles classes. That is why the functions take long argument lists
instead of closing over a context: it is the cost of the speed.

Every compiled function carries an explicit type signature. That is not decoration. Without it
numba compiles lazily, per argument type, and a single Python `int` arriving where the rest of
the code passes a `numpy.int32` causes a fresh compile **while the clock is running** — which is
exactly the bug that cost 1.2 seconds on the first move of every game until it was found.
Signatures force all of it into import time, where there is a 90-second budget and no clock.

## Search

Iterative deepening with aspiration windows around the root; principal variation search inside;
a transposition table that lives for the whole game.

- **Transposition table** — 2^22 slots in buckets of four, about 64 MB. A probe reads all four;
  a store replaces the least valuable, scored by depth minus a penalty for being from an older
  search. A plain always-replace table throws a deep result away for the next shallow one that
  collides with it.
- **Move ordering** — table move, then captures by most-valuable-victim with a static exchange
  search to separate the losing ones, then killers, then counter-moves, then quiet moves ranked
  by history and by *continuation* history: how a move has done as a reply to each of the last
  two moves played. Alpha-beta is worth only what its ordering is worth, and continuation
  history was the second largest measured gain in the engine.
- **Quiescence** — captures only, with static exchange and delta pruning, so the evaluation is
  only ever asked about positions where nothing is hanging.
- **Pruning** — null move, reverse futility, razoring, futility, late move pruning, late move
  reductions, and static-exchange pruning of losing captures. Most margins tighten when the
  position is not *improving*, meaning the static evaluation has fallen since our last turn.
- **Extensions** — checks only. Singular extensions were implemented and measured at −28 elo
  here: the verification search is effectively a second search of the node, and at these node
  counts the plies it buys cost more than they return. The code is behind a flag with that
  measurement recorded, because the trade moves with depth.

## Evaluation

Tapered: every term is scored twice, once with midgame weights and once with endgame weights,
blended by a phase counted from the non-pawn material left on the board.

Material and piece-square tables, mobility, pawn structure (isolated, doubled, backward,
connected, passed by rank with king distance in the endgame), bishop pair, rooks on open files
and the seventh, knight outposts, bad bishops.

Two terms need both sides' attack maps, which the first pass builds:

- **Threats** — pawns attacking pieces, minors attacking majors, rooks attacking queens, and
  pieces we attack that they do not defend.
- **King safety** — not just how much is pointed at the king zone, but **safe checks**: squares
  an enemy piece could check from without simply being taken. That is what actually kills kings,
  and adding it was the single largest measured gain in the engine.

Every term is computed symmetrically, and `tests/test_fuzz.py` asserts that a position and its
mirror evaluate to exactly the same number. That check has caught three separate bugs, all the
same shape: `sign * x // n` floors toward negative infinity, so it rounds the wrong way for
Black and quietly gives one side a centipawn the other does not get.

## Time management

The budget comes from the clock actually handed over, never a constant, and is capped at a
share of what remains. Two deadlines: a soft one past which a new iteration is not worth
starting, and a hard one the search abandons whatever it is doing at. The gap absorbs one
iteration that turns out to cost more than the last.

The soft deadline is a budget rather than a fixed point. A position whose best move has survived
five iterations gets 55% of it; one whose best move is still changing gets 135%; one whose score
has fallen more than 35 centipawns gets another 60% on top. Thinking longer when the position is
unclear and less when it is settled is most of what good time management is.

## How anything here was decided

`tools/match.py` plays two versions of the engine against each other from `tools/openings_big.txt`,
alternating colours, giving both sides the same **node** budget so the comparison carries no
timing noise, and prints a 95% interval. A change is kept when the interval clears zero.

That rig is the reason most of what was tried is not here. Measured over 100–350 games each:

| change | elo | kept |
|---|---|---|
| threats and safe-check king safety | +114 | yes |
| tuned evaluation weights | +55 | yes |
| continuation history | +35 | yes |
| capture history | −3 | no |
| search retune (IIR, LMR curve) | 0 | no |
| pawn storms and space | +5 | no |
| singular extensions | −28 | no |
| phalanx passers, rooks behind passers | −30 | no |

Several of the rejected ones read strongly positive at thirty games: capture history showed
+120 there and finished at −3. Thirty games is worth nothing at all.

The rig is structurally blind to one thing: with node counts fixed, a change that searches
faster scores exactly nothing. `tools/bench.py` measures that half separately.

`tools/tune.py` fits the evaluation's weights to labelled positions from self-play. The first
attempt used 190,000 positions from 5,000-node games labelled only with the game result: it cut
the fitting loss by 4% and won nothing, scoring 50.5% over 100 games. The second used 66,000
positions from **25,000-node** games, labelled with a blend of the result and the search's own
score at the time, which is far less noisy. That one is worth **+55 elo** over 350 games, and
its numbers are the ones in this file. The difference between the two attempts was entirely the
training data.
