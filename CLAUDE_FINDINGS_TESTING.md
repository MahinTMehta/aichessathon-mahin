# Testing findings — for the session doing fixes

Written by the testing/benchmark session (running an SPRT + evaluation suite against the shipped
engine and two retrained NNUE candidates). This document exists because two sessions are working
this folder concurrently and Mahin can't referee between two Claudes by trusting either one — so
every claim below comes with the exact command or `file:line` to check it yourself. Where you
disagree, that's the point: the repro steps should make it easy to prove us wrong, not just
easy to take our word.

Last updated: see git blame / file mtime on this document. If you're reading this stale, the
open questions in Section 5 are the ones most likely to have moved.

---

## SECTION 1 — Confirmed defects

### 1. `tools/worker.py:55` — undersized NNUE scratch buffer

```python
self.escratch = np.zeros(16, dtype=np.uint64)
```

**What's wrong**: should be `np.zeros(ESCRATCH_SIZE, dtype=np.uint64)`, imported from
`evaluate.py` (`from evaluate import ESCRATCH_SIZE`), exactly as `agent.py` already does.
`ESCRATCH_SIZE = ACC_BASE + ACC_SIZE` (`evaluate.py:296`), `ACC_BASE = 16`, `ACC_SIZE = 2 * L1`
(`nnue.py:67`). At the shipped/candidate L1=64, `ESCRATCH_SIZE = 144`. The hardcoded buffer is
**128 uint64 slots short**.

**Why it matters**: `evaluate.py`'s NNUE accumulator path writes into `escratch` at indices up to
143. With numba's default bounds-checking off, writing 128 slots past the end of a 16-slot numpy
array is undefined behavior — not guaranteed to crash, entirely capable of silently corrupting
adjacent heap memory and producing plausible-but-wrong numbers instead. Every `tools/match.py`
A/B comparison routes every move through `tools/worker.py`'s `Engine`, so every match result ever
produced this way inherited this.

**The git archaeology that makes this conclusive** (run these yourself):
```
git log --follow --oneline -- tools/worker.py
```
→ exactly one commit, `5377cb2` ("Updated model"). worker.py has never been touched since.
```
git show 5377cb2:evaluate.py | grep -n "ESCRATCH\|ACC_BASE\|nnue"
```
→ empty. At the one and only commit that ever wrote `tools/worker.py`, `evaluate.py` had no NNUE
accumulator at all — the scratch buffer genuinely only needed 16 slots back then ("sixteen slots
of attack maps", see the comment at `evaluate.py:293-294` in the current tree). `16` was correct
the day it was written. The NNUE work that made `ESCRATCH_SIZE` grow past 16 existed only in
local, uncommitted form until this session's `bb0b823` push — `tools/worker.py` was never
revisited to match. This is the same defect class the collaborator already found and fixed in
`tools/make_positions.py` (see that file's current `from evaluate import ESCRATCH_SIZE` import) —
this is the same bug, missed in the sibling file.

**Also present in**: `tools/bench.py:57` (same hardcoded 16, same fix), `tools/make_openings.py:45`
(same pattern, not currently used by any benchmark here, lower priority but worth the same fix
for consistency).

**Repro / verification**:
```bash
grep -rn "escratch = np.zeros(16" --include="*.py" .
```
Compare against `agent.py`'s correct version:
```bash
grep -n "escratch\|ESCRATCH_SIZE" agent.py
```

**Suggested fix**: add `from evaluate import ESCRATCH_SIZE` to the imports in each affected file,
change `np.zeros(16, dtype=np.uint64)` to `np.zeros(ESCRATCH_SIZE, dtype=np.uint64)`. Three-line
diff per file. We have already verified this fix works (patched isolated copies boot and play
correctly) — see Section 4 for why we didn't patch the tracked file ourselves.

---

### 2. `search.py:1033-1051` — iterative-deepening loop has no exit condition but wall-clock

The loop:
```python
for depth in range(1, max_depth + 1):
    if depth > 1:
        ...
        if now() >= effective:
            break
```
The only stopping signal is elapsed real time versus a computed deadline. There is no "this
position resolved to a degenerate value, further depth won't change it" cutoff.

**Why it matters**: when a position causes `negamax` to return `0` almost immediately at every
depth (see defect 3 below for why), each iterative-deepening pass completes in microseconds.
Real elapsed time never advances enough to trip the time check, so the loop runs all the way to
`max_depth = MAX_PLY - 16 = 112` (the value `agent.py:209` passes in) even though nothing useful
is happening.

**Log evidence** (from Mahin's uploaded round logs, stderr from `agent.py`'s own print at
`agent.py:222-227`):
- Round 91: `move 148 depth 108/44 score 0 nodes 947542 527ms` → `move 190 depth 112/2 score 0
  nodes 452 1ms` — depth races to the ceiling, node count collapses to a few hundred, seldepth
  (`info[3]`, the second number) shrinks toward nothing.
- Round 98: `move 51 depth 112/24 score 0 nodes 177329 160ms` and `move 60 depth 82/35 score 0
  nodes 609939 582ms` — same pathology recurring within one game.

**Repro**: grep any uploaded round log for `depth 112` or `score 0` lines adjacent to large
negative scores on nearby moves — the pattern is: normal negative score, then a `depth
###/<small>` line with `score 0`, then back to the normal negative score.

**Suggested fix**: not ours to make (contempt/repetition question is explicitly Mahin's to
decide — see Section 4). If/when he greenlights a fix, an early-exit when the position is
provably a dead draw (rather than iterating to the ceiling for no benefit) is one candidate; a
contempt/tie-break fix (defect 4) may make this less urgent since it addresses the move-choice
symptom directly rather than the wasted iterations.

---

### 3. `search.py:664-668` + `is_repetition` (`search.py:319-333`) — single-occurrence draw heuristic, not the referee's threefold

```python
if not root and excluded == NO_MOVE:
    if st[ST_HALF] >= 100 or insufficient_material(bb):
        return 0
    if is_repetition(repetition, st, position_index, key[0]):
        return 0
```
`is_repetition`'s own docstring: *"A single earlier occurrence is treated as a draw inside the
search... a line that returns to a position already on the board is worth exactly nothing to
search further."* This is a deliberate efficiency heuristic (avoid re-searching a line that will
just repeat), not a bug in the sense of "wrong code" — but it does not correspond to the
referee's actual threefold rule, and it fires on positions the search reaches by recursing into
its own hypothetical lines, not only on positions that have genuinely occurred twice on the real
board.

**The r98 mechanism, precisely** (moves 51 and 60, both scored `0` in an otherwise `~-1040` to
`-1140` position, king-shuffle context per Mahin's read of the surrounding moves):

1. `search_root`'s printed score/move (`info[6]`/`info[8]`, set at `search.py:1093-1095`) are
   only updated when an iteration completes — so `score 0` at move 51 is not log noise, it's the
   score of the move that actually got played.
2. That `0` reaches the root because the root's own move loop (`search.py:907-926`) recurses one
   ply down, that child hits the non-root return-`0` path above, and the negated value (`-0 = 0`)
   becomes the *root move's* score.
3. Ordinary `if score > best_score` (`search.py:926`, strict `>`) then correctly prefers `0` over
   `-1043` — this comparison is not buggy; it does exactly what it's supposed to given the input.
4. The `0` reflects a **single occurrence inside the search's own exploration of one hypothetical
   continuation** — not two prior real occurrences on the board. Reaching it assumes the opponent
   replies the way that one explored line assumed. The actual opponent did not cooperate — it
   played something else — so the position never became a real repeat, and the next move's
   search correctly re-evaluated the real, still-losing position (back to `-1043`/`-1140`).

**So**: the engine did not have a real, referee-recognized draw available and decline it. It
picked a move whose value was inflated by an expectation of a draw contingent on opponent
cooperation it never received, passing over whatever alternative the search had scored more
honestly (however negative). This is a real behavioral defect that chose an actual move in an
actual game — not a cosmetic log artifact.

**Repro**: not independently re-creatable without the exact FEN at move 51 (we only have the
stderr excerpt, not the full game record) — this diagnosis is derived from tracing the code path
against the printed evidence, not from replaying the position. If you have the full PGN/move list
for round 98, replaying it and dumping `agent.py`'s internal state at move 51 would confirm this
directly; we'd welcome that check.

---

### 4. No contempt / repetition-avoidance tie-break anywhere

```bash
grep -n "contempt\|CONTEMPT\|avoid.*repeat\|repeat.*avoid" search.py evaluate.py agent.py
```
→ empty. Root move selection (`search.py:926`, `if score > best_score`) has no awareness of
whether a tied or near-tied move repeats a position versus one that doesn't, and no directional
bias based on whether the engine is ahead or behind. `agent.py`'s `_record_history`/`observe`/
`_advance` (`agent.py:153-163`, `233-239`) do feed real played-game positions into the same
`repetition[]` array `is_repetition()` scans — so the engine *does* track positions it's been
asked about (worth correcting if this has been described otherwise) — but that tracking is
detection-only, with zero directional bias. Nothing says "I'm ahead, treat an offered repetition
as worse than pressing," or the inverse when behind.

---

### 5. No fifty-move-clock awareness in evaluation

```bash
grep -n "ST_HALF" *.py
```
→ exactly three uses, all in `search.py`: the `is_repetition` scan-window bound (`search.py:326`),
and two hard `>= 100` cutoffs (`search.py:479` in quiescence, `search.py:665` in negamax).
`evaluate.py` never references it. There is no scaling of evaluation as the clock approaches 100,
no move-ordering preference for clock-preserving moves when losing, no penalty for a
clock-resetting capture when behind. The concept "should I avoid this capture because it resets
fifty-move progress while I'm losing" does not exist in the code, full stop.

---

### 6. `tools/retrain.py` — four separate configuration issues, all confirmed and worked around locally (not fixed in the tracked repo)

- `retrain.py:22`: `WORK = Path(os.environ.get("RETRAIN_WORK", "/home/claude/work/data"))` —
  doesn't exist on this Windows machine, nothing creates it. Workaround: set `RETRAIN_WORK` to a
  real writable directory before running (we used `C:\Users\mahin\retrain_work`).
- `tools/nnue_train.py:47`: `BUCKETS = int(os.environ.get("NNUE_BUCKETS", "8"))` — defaults to 8
  output buckets. The currently-shipped net has 1 (`weights/net.npz`'s `W2` shape is `(1, 128)`).
  `retrain.py` never sets `NNUE_BUCKETS`, so a naive `retrain.py` invocation silently trains an
  architecture that doesn't match what's shipped. We ran both variants explicitly (see Section 5).
- `retrain.py`'s own default bound arg is `"0"` (`retrain.py:36`), which `nnue_train.py:41`
  documents as "set BOUND to 0 to fit the unbounded network instead" — the variant `ENGINE.md`
  documents at -31 elo. We passed `bound=96` explicitly to match what's shipped, but see Section 3
  for why even that reference number is now in question.
- `scipy` is imported by `tools/nnue_train.py` (`from scipy import sparse`) but is absent from
  `pyproject.toml`/`uv.lock`. `uv add scipy` resolves it (now reflected in the tracked
  `pyproject.toml`/`uv.lock`, pushed in `eba00b9`/`bb0b823` — check `git show bb0b823 --stat`).

---

## SECTION 2 — What is *not* broken (please don't re-check these, we already have)

- **MAX_PLY margins are sound, no out-of-bounds risk from the depth-112 pathology.** Arithmetic:
  `negamax` bails at `ply >= MAX_PLY - 8 = 120` (`search.py:685`) before recursing further;
  `quiescence` bails independently at `ply >= MAX_PLY - 2 = 126` (`search.py:477`). The
  `repetition[]` array (`agent.py:70`) is sized `MAX_GAME_PLIES + MAX_PLY + 8 = 1200 + 128 + 8 =
  1336`. `info[4]` (game length so far) is kept under 1200 by `agent.py`'s own halving logic
  (`agent.py:156-160`). Worst-case index into `repetition[]` is `info[4] + ply ≈ 1200 + 120 =
  1320`, safely under 1336. We traced every write site; nothing here needs fixing.
- **`agent.py` correctly imports `ESCRATCH_SIZE`** (`agent.py:29,73`) — defect 1 is a dev-tooling
  bug only. The shipped engine has never had this problem.
- **Init timing is fine.** Measured on real competition hardware (per Mahin, not us): 32.0s,
  33.2s, 33.6s, 36.3s, 36.4s against the 90s budget — 36-40% used, comfortable margin. Our own
  cold-import timing on this dev machine (24-25s for shipped and both candidates, effectively
  identical) scales consistently with those numbers.
- **`agent.zip` compliance**: 9 files (`agent.py, bitboards.py, evaluate.py, fen.py, nnue.py,
  perft.py, position.py, search.py, weights/net.npz`), 221,960 bytes unzipped (~0.22MB, nowhere
  near the 50MB cap), every import is stdlib/numpy/numba (the one `chess` reference is a lazy,
  defensive fallback), `get_move(fen: str, time_left_ms: int) -> str` matches exactly and returns
  UCI never SAN, no filename shadows `chess`/`types`/`random`/etc., and the net is verifiably
  trained from scratch (`tools/nnue_train.py` initializes `W1`/`W2` via `rng.normal(...)`, no
  external checkpoint loaded anywhere in the pipeline). Full audit re-run against the current zip
  on request.
- **Clock discipline is fine.** 15-33s remaining at game end across every logged game we've seen
  (four originally, now five with round 98) — no flag risk observed.

---

## SECTION 3 — The measurement problem (read this before trusting any historical elo number)

`ENGINE.md`'s NNUE-era elo history table — the ±96 clamp at +118, the two data-expansion rows at
+66/+51, the "output buckets" row at −81, the accumulator-threading tie, the singular-extensions
row at −28, all of it — was measured via `tools/match.py`, which routes every game through
`tools/worker.py`. Given defect 1, and given the git archaeology showing this bug has been live
throughout the entire period Mahin's local (then-uncommitted) NNUE work took place, **we cannot
currently treat that table as reliable evidence for anything**. This is not an accusation of bad
work — the bug predates the NNUE effort by construction (worker.py's `16` was correct when
written) and would have been invisible without exactly this kind of git-archaeology cross-check.
It's also not necessarily *wrong* — the true elo differences could turn out to match the old
numbers once re-measured cleanly. We just don't know yet, and neither should anyone be citing
that table as settled until it's re-run on a patched harness.

We are re-running the load-bearing comparisons (shipped vs. two retrained candidates, one at 8
buckets and one at 1) via a from-scratch SPRT on a **patched, isolated** copy of the harness — see
Section 5 for where that stands. We are *not* re-litigating the historical table entry by entry;
that would burn hours neither of us has. We're treating it as unverified, full stop, and letting
the new numbers speak for themselves.

**Addendum — one calibration data point, not a reversal of the above.** Our clean-harness SPRT
for the 8-bucket candidate has now **resolved**: `H0 accepted (not better)` at n=150, LLR_pair
-3.09 (crossed the -2.94 bound), final score 37.7% (+38=37-75), elo **-88**. Zero illegal moves,
zero crashes across all 150 games. ENGINE.md's "output buckets" row, measured on the broken
harness, says -81. Those two numbers landing within 7 elo of each other, now that both are final
rather than interim, is a meaningful sign that, at least for an
effect this large, the harness bug was not distorting the direction or rough magnitude of the
result -- buckets really do appear to cost real elo, independent of which harness measured it.
That's worth Mahin knowing plainly: it means the historical table is more trustworthy than the
rest of this section might suggest, at least here. We are not walking back this section's core
point -- a bug this structurally serious (undersized buffer, bounds-checking off, silent
corruption) can easily behave inconsistently across different configurations, sample sizes, or
positions, and one large effect calibrating cleanly says nothing about whether a small real
effect (a handful of elo, the kind several other rows in that table report) would have survived
the same corruption undetected. Treat this as one corroborated data point, not a blanket
all-clear for the table.

---

## SECTION 4 — Division of labour

We (this session) own testing and hold the SPRT harness running right now. Please:

1. **Don't edit `tools/worker.py` without telling us first** — even though it's demonstrably
   broken, it's our measuring instrument for anything already in flight. If you fix it, ping this
   document (or however Mahin wants us to hand off) and we'll pick up the patched version for
   future runs; we've already independently verified the fix (`ESCRATCH_SIZE` import + correct
   allocation) works, so there's no daylight between us on what the fix should look like.
2. **Don't rebuild `agent.zip` without coordinating** — that's what actually gets submitted, and
   we don't want a rebuild landing mid-comparison and silently changing what "shipped" means in
   our running numbers.
3. **Tell us when a fix lands** (which file, what changed) so we can test it against the deployed
   build and hand back a number with a confidence interval, not just a vibe. We can turn around a
   fresh comparison quickly — our harness is already warm.

We are running our own SPRT against **isolated copies** of the engine and the harness precisely so
your edits to the live tree can't corrupt a benchmark mid-run — see the note in Section 5 on how
we handled that. If you need a clean baseline copy of the shipped engine to test against
yourselves, `C:\Users\mahin\bench_shipped` is one (source + weights only, patched `worker.py`),
built from the tree as of commit `eba00b9`.

---

## SECTION 5 — Open questions we have not resolved

- **Does the 8-bucket architecture actually help on a clean harness? RESOLVED: no.** SPRT
  concluded `H0 accepted (not better)` at n=150, LLR_pair -3.09, final elo -88 (+38=37-75, 37.7%),
  zero illegal moves, zero crashes. This is now a settled result, not a trend. 1-bucket SPRT is
  still running (see snapshot below) — that one remains open.
- **Is `bound=96` actually right?** We used it because it matches what's shipped and because the
  historical table (now suspect, see Section 3) said so. Not independently re-verified against
  alternatives (e.g. re-testing 64/128/unbounded) — that would cost more SPRT time than we've
  spent on the primary comparison and wasn't asked for. Open.
- **Is the contempt/repetition-avoidance fix (defects 3+4) worth the risk this close to the
  deadline?** We've confirmed the mechanism is real (Section 1, item 3) and has plausibly cost at
  least one full point already (round 98). We have not implemented or tested a fix — that's
  explicitly staying with Mahin's decision, and with whichever of us he asks to write it once
  decided.
- **A/A sanity check on the harness bug's actual impact — done, and the result is worth reading
  carefully rather than as reassurance.** 30 games, shipped-vs-itself, on both the patched and
  unpatched harness: both finished at the exact same result, `+11=8-11, score 50.0%, elo -0`,
  down to identical W/D/L. On this specific run, the buffer overflow in the unpatched version
  produced bitwise-identical game outcomes to the patched one. We do **not** read this as "the bug
  doesn't matter" — undefined behavior from an out-of-bounds write is inherently
  allocator-/layout-dependent, so a null result here is consistent with the corruption sometimes
  landing somewhere that doesn't affect the computed move and sometimes not (the 8-bucket result
  above shows it clearly *can* matter when the affected configuration differs — more output
  buckets means a larger `W2`/`b2` array likely sitting at a different heap offset relative to the
  overflowing `escratch`, so whether the corruption bites may depend on exactly this kind of
  detail). Diagnostic, not proof of either "safe" or "unsafe" in general — 30 games of one
  configuration can't establish that either way.

**Snapshot at time of writing** (will be stale fast — ask the testing session directly for current
numbers rather than trusting this table for long):

| Comparison | n | Score | Elo | Note |
|---|---|---|---|---|
| shipped vs 8-bucket (SPRT, patched+isolated harness) | 150 (FINAL) | 37.7% | -88 | **RESOLVED: H0 accepted, not better.** LLR_pair -3.09, crossed -2.94 bound. 0 illegal, 0 crashes. |
| shipped vs 1-bucket (SPRT, patched+isolated harness) | 150, running | 55.7% | +40 | positive at every checkpoint since n=10, LLR_pair +0.59, not yet decisive |
| A/A, shipped vs itself, patched vs unpatched harness | 30 each | 50.0% both | -0 both | identical result on both harnesses — see open-questions note, not a clean-bill-of-health |
| Held-out eval error, self-play data (both candidates vs shipped) | 112,574 | — | — | shipped 188.8cp SD, both candidates ~176.5-176.7cp SD — clear win over shipped, candidates indistinguishable from each other |
| Held-out eval error, real out-of-distribution games | 136 | — | — | shipped notably *worse* than pure handcrafted eval (269.0cp vs 224.5cp); both candidates degrade far less (229.0/237.0cp) — small n, directional not decisive, but a bigger and separate concern from the bucket question |

If you're reading this and the SPRT has since resolved, trust the live numbers over this table.
