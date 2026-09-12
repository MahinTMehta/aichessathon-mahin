> **Note from the testing session, 2026-09-10**: another Claude session has been running a
> benchmark/SPRT suite against this engine concurrently with whatever session is reading this.
> We found a real bug affecting past A/B testing (`tools/worker.py`'s NNUE scratch buffer is
> undersized) plus some search-behavior findings, all written up with exact repro commands in
> `CLAUDE_FINDINGS_TESTING.md` at repo root — please read it before touching `tools/worker.py`,
> `search.py`, `evaluate.py`, or rebuilding `agent.zip`. Everything in it is meant to be checked,
> not taken on faith — if something looks wrong, the doc gives you the command to prove it. We
> hold the SPRT harness and can test anything you produce against the deployed build on request.

AGENTS.md