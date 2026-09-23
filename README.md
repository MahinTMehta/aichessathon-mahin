# Queen Mary's Gambit — AI Chessathon 2026

A chess engine written in Python and compiled with numba, built by a two-person team for the
Optiver-sponsored AI Chessathon (Encode Club, London, September 2026).

**Result:** 47th of 334 teams in the 13-round qualification Swiss (8.5/13, top entry of ten
from Queen Mary) out of 465 registered, reaching the 50-seat London final on 12 September,
where it placed 34th. Around 750k positions a second on a single core. No crashes, illegal
moves, timeouts or init failures across more than 60 platform games.

The competition allowed no native code, no third-party engines and no borrowed networks, so
everything here is Python running on one core under a 120 s + 0.5 s clock.

---

## Which build is which

The engine that played the final was developed in a separate working copy under time pressure
on the day and was never committed to this repository. It matters, because the two builds
differ in ways that change the numbers below.

| | committed (`HEAD`) | final day, 12 Sep |
|---|---|---|
| network | 768→64→1, int16, trained on 1,333,166 positions | same shape, 4,112,952 positions |
| correction bound | ±96 cp | ±256 cp |
| correction history | no | yes, 14-bit table keyed on pawn structure |
| contempt | none | 30 cp against the side to move |
| mate drive with bare king | no | yes |
| checkmate vs fifty-move | draw wins | checkmate wins |
| compilation | eager at import, 90 s budget | deferred with pinned signatures, 30 s budget |
| init on the platform | 36–40 s | 10.5–11.3 s |
| transposition table | 2²⁴ slots, 4-way buckets, 256 MiB | same |

The final-day source is at `dist/agent-final-day.zip`; every number below says which build it
belongs to.

## Layout

```
agent.py        entry point: FEN in, UCI move out, and the clock
bitboards.py    attack tables, magic bitboards, zobrist keys
position.py     board state, move encoding, move generation, make/unmake
evaluate.py     tapered hand-crafted evaluation plus the network's bounded correction
nnue.py         the network (768 -> 64 -> 1, int16 weights, int32 accumulator)
search.py       negamax, alpha-beta, PVS, iterative deepening, and the pruning
fen.py          FEN parsing and UCI formatting
tools/          the A/B rig (match.py, worker.py), training pipeline, benchmarks
tests/          perft, en passant, mirror-symmetry fuzz, tactics
harness/        the organisers' local runner, unchanged
baselines/      the starter kit's reference agents
notes/          working notes kept from the competition
ENGINE.md       design notes and the full measurement history
```

## The engine

**Search.** Iterative deepening with aspiration windows, principal-variation search, and a
four-way-bucket transposition table that persists across the game. Move ordering: the table
move, then captures by MVV-LVA split on static exchange evaluation, killers, counter-moves,
history and continuation history. Pruning: null move, reverse futility, razoring, futility,
late-move pruning and reductions, and SEE pruning of losing captures. Quiescence over captures
only.

**Evaluation.** Tapered midgame and endgame scoring over material, piece-square tables,
mobility, pawn structure, the bishop pair, rook and knight terms, threats, and king safety
built on safe checks. Every term is symmetric, and a fuzz test asserts that a position and its
mirror score identically, which caught three sign-rounding bugs.

**Network.** A 768→64→1 network with int16 weights and integer arithmetic throughout, fitted
to Stockfish depth-8 labels but predicting the *residual* between that label and the
hand-crafted evaluation rather than the score itself. Its output is hard-bounded. Unbounded it
overrules material in positions the training data never covered, and measured worse than no
network at all; bounded it was the largest single gain in the project.

**Time management.** Soft and hard deadlines derived from the clock actually handed over,
extended while the best move is still changing and cut once it has held for several
iterations. The regression test is a full self-play game at the real time control.

## How decisions were made

`tools/match.py` plays two builds against each other from a large opening set, alternating
colours, with both sides on the same **node** budget so machine timing drops out, and reports a
95% interval. A change shipped only when that interval cleared zero.

Figures below are from matches run with a correctly sized harness and with both arms recorded:

| change | elo | 95% CI | n |
|---|---|---|---|
| network bounded at ±128, vs no network | +68 | [+18, +120] | 140 |
| tightening the bound ±128 → ±96 | +50 | [+6, +95] | 140 |
| training data 602k → 812k positions | +48 | [+8, +89] | 190 |
| training data 812k → 1.33M positions | +51 | [+8, +96] | 130 |
| network unbounded, vs no network *(rejected)* | −31 | [−94, +30] | 90 |
| wider first layer with output buckets *(rejected)* | −81 | [−155, −13] | 70 |
| eight output buckets, independent SPRT *(rejected)* | −88 | [−139, −40] | 150 |

`ENGINE.md` carries the full table, including rows measured before the harness was audited and
rows whose arms were not recorded. Those are labelled rather than deleted, because the honest
version of a measurement log includes the measurements you can no longer stand behind.

Several changes that ultimately failed looked strong early. Capture history read +120 Elo after
30 games and finished at −3 over 200. Thirty games is worth nothing, and the rig existed
largely to stop us acting on runs that size.

Changes that never separated from zero are not claimed as gains anywhere in this repository.
That includes correction history (+19, [−3, +42], n=578), the mate drive, contempt, and the
final build against its immediate predecessor (+16, [−27, +58], n=178). Correction history
ships in the final build regardless: it was free at the margin and the point estimate was
positive, but it was not demonstrated.

**Correctness** was checked separately from strength, because a stronger engine that is
occasionally wrong is worse than a weaker one that never is:

- `tests/test_perft.py` — exact node counts against known perft values, all matching
- `tests/test_en_passant.py` — 3,804 positions against python-chess, no mismatches
- `tests/test_fuzz.py` — mirror-symmetry fuzz over the evaluation
- `tests/test_tactics.py` — informational tactical suite at a fixed time per move

## The final-day rule change

On the morning of the final the organisers cut the start-up budget from 90 s to 30 s.
Compilation was restructured to defer with pinned type signatures, bringing init from 36–40 s
down to 10.5–11.3 s with play verified identical to the previous build on every test position.
Several opposing entries forfeited under the same rule.

## Running it

```
uv sync
uv run python tests/test_perft.py
uv run python tests/test_en_passant.py
uv run python tests/test_fuzz.py 8
uv run python -m harness.play --white . --black baselines/minimax
```

The tests are scripts with a `main()`, not pytest tests, so `pytest tests/` collects nothing.

## Known defects

`tools/worker.py`, `tools/bench.py` and `tests/test_fuzz.py` allocate a 16-slot evaluation
scratch buffer where `evaluate.py` declares 144. Nothing in any shipped `evaluate.py` or
`nnue.py` writes past slot 15 — verified over 20,000 positions against a canary-filled array —
so no result here was affected, but the allocation should be corrected before that buffer is
used for what its size implies.

## Attribution

`harness/` and `baselines/` come from the public competition starter kit. Everything else is
ours. Stockfish was used offline to label training positions, which the rules permit; nothing
from it ships, and the network was trained from scratch (`tools/nnue_train.py`). AI coding
assistants were used during development, which the rules also permit provided the team can
explain the code; the architecture, the measurement methodology and every number above are
ours, and the notes those assistants worked from are kept in `notes/`.

MIT licence.
