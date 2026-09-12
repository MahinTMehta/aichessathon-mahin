"""A reference engine over UCI, for measuring against something that is not ourselves.

Everything the engine has been measured with so far plays it against a previous version of
itself. That is the right way to test a change — same opponent, same openings, the difference is
the change — but it has one blind spot that matters here: an engine tuned to beat its own
previous version is tuned against one style. The tournament is other people's engines.

So this wraps Stockfish as (a) an opponent that can be turned down to roughly our strength, and
(b) an oracle that says what a move was worth. Neither ships. Both are the ordinary use of a
reference engine that the rules describe.
"""

import subprocess
from pathlib import Path

MATE = 30000


class Reference:
    """One long-lived UCI process. `analyse` asks its opinion, `best` asks it to play."""

    def __init__(self, path: str, hash_mb: int = 128, threads: int = 1) -> None:
        self.process = subprocess.Popen(
            [path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )
        assert self.process.stdin is not None and self.process.stdout is not None
        self._send("uci")
        self._read_until("uciok")
        self._send(f"setoption name Threads value {threads}")
        self._send(f"setoption name Hash value {hash_mb}")
        self._send("isready")
        self._read_until("readyok")

    def _send(self, line: str) -> None:
        assert self.process.stdin is not None
        self.process.stdin.write(line + "\n")
        self.process.stdin.flush()

    def _read_until(self, prefix: str) -> list[str]:
        assert self.process.stdout is not None
        seen = []
        for line in self.process.stdout:
            seen.append(line.rstrip())
            if line.startswith(prefix):
                return seen
        raise RuntimeError(f"reference engine died waiting for {prefix}")

    def new_game(self) -> None:
        self._send("ucinewgame")
        self._send("isready")
        self._read_until("readyok")

    def go(self, fen: str, depth: int = 0, nodes: int = 0, movetime: int = 0) -> tuple[str, int]:
        """Return (best move in UCI, score in centipawns from the side to move's point of view)."""
        self._send(f"position fen {fen}")
        if nodes:
            self._send(f"go nodes {nodes}")
        elif movetime:
            self._send(f"go movetime {movetime}")
        else:
            self._send(f"go depth {depth or 16}")
        score = 0
        for line in self._read_until("bestmove"):
            if line.startswith("info ") and " score " in line and " pv " in line:
                parts = line.split()
                at = parts.index("score")
                if parts[at + 1] == "cp":
                    score = int(parts[at + 2])
                else:
                    plies = int(parts[at + 2])
                    score = MATE - abs(plies) if plies > 0 else -(MATE - abs(plies))
            elif line.startswith("bestmove"):
                return line.split()[1], score
        raise RuntimeError("no bestmove")

    def close(self) -> None:
        try:
            self._send("quit")
            self.process.wait(timeout=5)
        except Exception:
            self.process.kill()


def find() -> str:
    for candidate in (
        "/tmp/stockfish/stockfish-ubuntu-x86-64-avx2",
        "/usr/local/bin/stockfish",
        "/usr/bin/stockfish",
    ):
        if Path(candidate).exists():
            return candidate
    raise SystemExit("no reference engine found")
