# The engine

`agent.py` is the submission entry point. Everything under it is a conventional bitboard chess
engine written so that numba can compile it, which is the only reason the node counts are in
the millions rather than the thousands.

```
agent.py        entry point: FEN in, UCI move out, and the clock
bitboards.py    attack tables, magic bitboards, zobrist keys
position.py     board state, move encoding, move generation, make/unmake
evaluate.py     tapered hand-crafted evaluation, plus the network's correction
nnue.py         a small trained network that corrects it
weights/        the network's weights, int16
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

## The network

On top of all of that sits a small trained network — 768 inputs, one per (piece colour, piece
type, square), read once from each side's point of view into 64 numbers each, clamped, and dotted
down to a single number. It is fitted to what Stockfish thinks of a position at depth 8, and it
predicts not the score but **the difference between that score and what the evaluation above
already says**. Everything ships as int16 and the arithmetic is integer end to end.

Fitting the difference is what makes it work on the amount of data we have. Material and piece
placement are already priced and already tuned; a network asked to rediscover them from 600,000
positions would spend all of its capacity getting back to where the handcrafted evaluation
already is. Fitting the residual means every one of its 50,000 weights is spent on something the
evaluation is missing.

The rules ban shipping another engine or another team's network and explicitly allow learning
from one. Nothing from Stockfish is in the zip; what is in the zip is a file of numbers fitted
here, from positions labelled here.

### What it is worth, and the thing that nearly hid it

Statically, it is a large improvement, and not only on average. On held-out positions it cut the
error against Stockfish from 146 to 97 centipawns, most in the positions nearest to equal — where
the error fell by half — and it raised the share of position *pairs* it ranks the same way
Stockfish does from 87.3% to 92.4%. That last number is the one that should matter, because
ranking positions is the only thing an evaluation does inside a search.

Played, the first version lost 81 elo.

The fix was not more data and not a bigger network. It was to **bound the correction at ±128
centipawns**. Unbounded, it reached 786, and a search spends most of its time in positions no
game would ever reach — the branch about to be refuted, the piece hanging for a ply — where a
network fitted on positions from real games is extrapolating and its answer is a guess, not a
correction. Inside the bound it can still say everything it usefully has to say; outside it, it
can no longer overrule material.

| bound | elo |
|---|---|
| 0 (no network) | baseline |
| 64 | +109 |
| 96 | **+118** |
| 128 | +68 |
| 256 | −162 |
| unbounded | −31 |

The bound is a hard truncation, and that turns out to matter. Letting the excess through at an
eighth of its size past 96 — which keeps the order between two positions the truncation makes
identical, and looks like the more careful thing to do — cost 53 elo. Whatever the network is
doing out there, none of it is worth having.

The correction is also applied *before* the drawish-material scaling rather than after, so a
network that has barely seen a bare-king ending cannot claim an advantage in one.

More positions **do** help, and are the only thing that has. Another 210,000 from self-play,
labelled the same way, took the set from 602,000 to 812,000 and was worth **+66 elo** over 90
games; another 521,000 after that took it to 1.33 million and was worth **+51** over 130. Nothing
else about the network changed either time. Two thirds of what the network was worth in the first
place has come from data since, which says the fit is nowhere near what the architecture can hold.

What is scarce is starting points rather than positions. 738,000 generated positions deduplicated
to 521,000, because self-play from 260 openings keeps arriving at the same middlegames; the
generator now plays up to eight random moves before the engine takes over rather than three.

More positions of the wrong kind help not at all. A wider set — every position plus a copy of it
after a few random legal moves, 1.17 million labelled in all — was worth −21 elo over 50 games. It is easy to
see why afterwards: measured back on the positions that actually occur, the wider network was
worse, 123 centipawns of error against 97 and pair ordering down from 92.4% to 88.7%. Half its
capacity went on positions no game reaches. Neither did fitting the bound into the network
itself, as a smooth squash on its output rather than a truncation afterwards: it landed at the
same held-out loss as truncating, 0.00481 against 0.00481.

### What it costs

Rebuilding the accumulator from the board is 213 nanoseconds a node — 32 pieces, two weight rows
each — and the network as a whole costs 11% of the search's speed, which is about 8 elo handed
back. The standard answer is to update the accumulator incrementally in make/unmake instead of
rebuilding it at every leaf, and that was built twice.

It gains nothing. Not "a little"; nothing measurable. The implementation is exact — `_put` and
`_take` are the only two places a piece ever moves, so folding the update into them covers
captures, promotions, castling and en passant with no second copy of make_move's logic, and
`tests/test_accumulator.py` walks 44,615 moves checking it against a rebuild — and it benchmarks
inside the noise of the version it replaced. The arithmetic says why, once you count the right
thing: the rebuild happens once a node, but the update happens on every move made *and* unmade,
including the ones the search abandons a ply later, and there are more of those than there are
leaves. Four to eight row updates a node against thirty-two rebuilt rows is a factor of four on
paper, and the calls, the loop setup and the moves that are made only to be thrown away eat it.

Two intermediate results worth keeping, because both look like good ideas:

  * the accumulator has to be reachable from both the move maker and the evaluation, and the one
    array already threaded through both is the zobrist key's, which is 64 bits wide. Putting it
    there avoids adding a parameter to every signature in the search — and makes it *slower* than
    the rebuild, because a 64-bit lane holds four values where a 16-bit lane holds sixteen.
  * doing it properly, with an int16 accumulator threaded through every signature, is the version
    that benchmarks as a tie.

## Time management

The budget comes from the clock actually handed over, never a constant, and is capped at a
share of what remains. Two deadlines: a soft one past which a new iteration is not worth
starting, and a hard one the search abandons whatever it is doing at. The gap absorbs one
iteration that turns out to cost more than the last.

The soft deadline is a budget rather than a fixed point. A position whose best move has survived
five iterations gets 55% of it; one whose best move is still changing gets 135%; one whose score
has fallen more than 35 centipawns gets another 60% on top. Thinking longer when the position is
unclear and less when it is settled is most of what good time management is.

## Checking the clock over a full game

A flag is a whole point, and the failure mode is invisible in short games: the per-move spend
converges on slightly more than the increment and a long game bleeds toward zero. A real
seventy-move game finished with 2.8 seconds left before the reserve was added.

The check is a full-length self-play game under the real time control:

```
uv run python -m harness.play --white . --black . --base-ms 120000 --increment-ms 500 \
    --ply-cap 600 --fen "<a balanced position from tools/openings_big.txt>"
```

Read the clock column. It should fall to a floor and then hold or rise, never trend to zero.
On the current build a 144-ply game bottoms out near 17 seconds and climbs from there:

| move | 1 | 21 | 42 | 62 | 70 | 71 | 72 |
|---|---|---|---|---|---|---|---|
| clock | 120.0s | 58.5s | 37.2s | 23.2s | 17.4s | 17.5s | 17.5s |

## How anything here was decided

`tools/match.py` plays two versions of the engine against each other from `tools/openings_big.txt`,
alternating colours, giving both sides the same **node** budget so the comparison carries no
timing noise, and prints a 95% interval. A change is kept when the interval clears zero.

That rig is the reason most of what was tried is not here. Measured over 100–350 games each:

| change | elo | kept |
|---|---|---|
| the network, bounded at 96 centipawns | +118 | yes |
| a third more positions to fit it to | +66 | yes |
| another 521,000 positions, 1.33M in all | +51 | yes |
| threats and safe-check king safety | +114 | yes |
| tuned evaluation weights | +55 | yes |
| continuation history | +35 | yes |
| capture history | −3 | no |
| search retune (IIR, LMR curve) | 0 | no |
| pawn storms and space | +5 | no |
| singular extensions | −28 | no |
| phalanx passers, rooks behind passers | −30 | no |
| distilled piece-square tables | 0 | no |
| the network, unbounded | −31 | no |
| the network, bounded at 256 | −162 | no |
| the network, bound compressed rather than hard | −53 | no |
| incremental accumulator, int16, threaded | 0 | no |
| incremental accumulator in the key array | slower | no |
| network at L1=128 with output buckets | −81 | no |
| the network, trained on 2x the positions, half of them random | −21 | no |
| the network, fitted only inside the bound | −12 | no |
| singular extensions at 150,000 nodes | +9 | no |

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
