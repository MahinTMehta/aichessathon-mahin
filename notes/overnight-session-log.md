# Overnight results

---

## 2026-09-11 — session start (exact local time unavailable, see note)

**STATUS: NO JOBS RAN. NO GAMES WERE PLAYED. NO DATA WAS COLLECTED.**

Read this section before you plan anything around tonight's run. There are no
numbers below because none were produced. Nothing in this file should be pooled,
averaged, or entered into an interval.

### Why

The test executor has no code execution environment. The Linux workspace this
session would run jobs in fails to start, with an identical error on every
attempt:

> failed to mount ... under Plan9 share "c" which is not mounted
> A Windows update released September 8 prevents Claude's workspace from reaching
> your files. We're tracking this issue. **Claude Code is unaffected.**

This is a host-level fault, not a configuration problem in the repo or the kit.
Retried twice this session, identical failure both times; stopped retrying per the
tool's own guidance.

Consequence: no Python, no `run_one.py`, no Stockfish, no unzip, no downloads.
File reads and writes still work, which is the only reason this file exists.

Note on the timestamp: the clock is read through the same tool that is down, so the
date above is reliable and the time of day is not.

### Job disposition

| Job | Status | Blocker |
|---|---|---|
| 1 — decision test (`run_one.py` shards) | **NOT RUN** | No interpreter. Kit also still packed; no unzip available. |
| 2 — init time × 3, bundle and baseline | **NOT RUN** | No interpreter. |
| 3 — blunder census vs Stockfish depth 18 | **NOT RUN** | No interpreter, no Stockfish binary, no PGN download path. |
| 4 — opening-phase eval and clock cost | **NOT RUN** | Depends on Job 3 data, which does not exist. |

### State on disk, verified by direct file read

- `RESULTS/` did not exist before this file was written. It contains this file and
  nothing else. `RESULTS/raw/` was not created — there is no raw data to put in it.
- The kit is still a zip in the session upload area. `baseline/`, `bundle/`,
  `sharp_openings.txt` and `run_one.py` are **not** unpacked next to the repo, and
  no `campaign/results/*.jsonl` exists anywhere.
- The repo working tree was not touched. Nothing in `baseline/`, `bundle/` or
  `run_one.py` was modified, because none of it was reachable.
- Archive listing of the kit was readable without unpacking, and it looks complete:
  full `bundle/` and `baseline/` trees, each with `weights/net.npz` present, plus
  `tools/match.py`, `tools/worker.py`, `tools/bench_warm.py` and
  `sharp_openings.txt`. The kit is fine. The executor is not.

### What to do instead

**Claude Code is unaffected by this fault.** It does not go through the broken
mount. Every job in the brief can be run there unchanged, on the same box, with the
same commands. That is the shortest path to having data by morning.

### One caveat to carry into the ship decision

This is arithmetic, not a result, and it holds regardless of who runs the games.

At roughly 40% draws, the 95% interval on a fixed-node A/B is about **±26 elo at
400 games** and **±21 elo at 600**. A true +20 effect therefore returns an interval
of roughly [−6, +46] at 400 games, or [−1, +41] at 600. Neither clears zero.

Resolving a 20 elo effect with reasonable power needs somewhere **north of 2,000
games**. A single overnight run on two cores does not get there.

The ship rule — interval clears zero or it does not ship — is correct and should
hold. But an underpowered null is not evidence of no effect. "The interval did not
clear zero" and "the feature does not work" are different findings, and below about
2,000 games only the first one is available. Record which one you actually have.

---
