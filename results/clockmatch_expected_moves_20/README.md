# clockmatch: EXPECTED_MOVES 28 (baseline) vs 20 (candidate)

Run 2026-09-10, completed 2026-09-11. ~4.5 h wall time on a 16-core Windows machine.

## Result

**Null result. 160 games, pooled elo +2, 95% CI [-28, +33].**
The candidate is indistinguishable from baseline in the 20-28 range at this sample size.

| Shard | n | W-D-L | Score | Elo | Worst clock | Median clock |
|---|---|---|---|---|---|---|
| `--start 0`   | 40 | +8 =23 -9 | 48.8% | -9  | 4.1 s | 12.7 s |
| `--start 200` | 40 | +2 =35 -3 | 48.8% | -9  | 5.8 s | 17.2 s |
| `--start 400` | 40 | +9 =24 -7 | 52.5% | +17 | 5.7 s | 12.4 s |
| `--start 600` | 40 | +7 =27 -6 | 51.2% | +9  | 5.6 s | 12.9 s |
| **Pooled**    | **160** | **+26 =109 -25** | **50.3% [45.9, 54.7]** | **+2 [-28, +33]** | **4.1 s** | — |

Zero FAILURE lines across all 160 games: no illegal moves, no crashes, no flag falls.

All four shards straddle zero and no shard's interval excludes it. Draw rate 68.1%
(109/160), which is why 160 games only resolves to about +/-30 elo. Detecting a ~10 elo
effect at this draw rate needs roughly 1500-2000 games (~2 days on this machine).

## Exact method

Baseline = repo tree at the state below. Candidate = a copy with exactly one line changed,
`agent.py:39`, `EXPECTED_MOVES = 28` -> `EXPECTED_MOVES = 20` (verified by diff: one line).

```
robocopy <repo> <repo>/../cand_time /E /XD .git .venv __pycache__ shards label_shards docs baselines \
                                       /XF *.gz all_positions.txt stockfish.exe agent.zip
sed -i 's/^EXPECTED_MOVES = 28$/EXPECTED_MOVES = 20/' ../cand_time/agent.py

uv run python tools/clockmatch.py . ../cand_time --games 40 --start 0     # and 200, 400, 600
```

Four shards run concurrently. Time control 120 s + 0.5 s (`harness/rules.py`), the
competition referee via `harness/referee.py`.

Openings drawn as `(index // 2) % 260` from `tools/openings_big.txt` (260 lines), so the
four shards used disjoint opening sets: 0-19, 100-119, 200-219, and 40-59 respectively.
Note the wraparound: `--start 600` maps to openings 40-59, not 300-319. Offsets that are
multiples of 520 collide with `--start 0`.

Repo state: branch `main`, HEAD `eba00b9`, working tree dirty (modified AGENTS.md,
CLAUDE.md, agent.zip; untracked CLAUDE_FINDINGS_TESTING.md, tools/accuracy.py,
tools/clockmatch.py, tools/reference.py). The engine sources themselves were unmodified.

## Caveats (read before citing these numbers)

1. **Clock floors are not attributable to a side.** `clock_floor()`
   (`tools/clockmatch.py:57-67`) takes a single global minimum over all `[%clk ...]`
   annotations without separating white from black, and the candidate alternates colour
   every game. "Worst 4.1 s" means *one of the two sides* reached 4.1 s; there is no way to
   tell which. Since the whole question is whether the faster-spending candidate runs its
   clock lower, this is the measurement that would most directly answer it, and the script
   as written cannot. Fix would be to split the PGN parse by side.

2. **Agent suspension was a no-op.** `harness/sandbox.py:18` gates `suspend()`/`resume()`
   on `hasattr(signal, "SIGSTOP")`, which is False on Windows. The waiting agent was never
   frozen. Believed harmless here because this engine only searches inside `get_move` and
   otherwise blocks on stdin consuming no CPU — but this is the one test whose purpose is
   wall-clock fidelity, so a marginal result should be reconfirmed on Linux.

3. **Per-game PGNs were not retained.** The referee returns `outcome.pgn`, clockmatch
   extracts one number from it and discards it. The move-by-move games from this run no
   longer exist. To keep them, clockmatch would need to write `outcome.pgn` out per game.

4. These numbers come from `tools/clockmatch.py`, which does **not** route through
   `tools/worker.py`, so the undersized-`escratch` defect described in
   `CLAUDE_FINDINGS_TESTING.md` Section 1 does not apply to this run. Both sides ran the
   real `agent.py` through the real referee.

## Raw logs

`shard_start{0,200,400,600}.log` — unedited stdout from each of the four runs.
