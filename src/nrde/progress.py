"""Stderr progress for long CLI stages.

Library calls stay quiet unless :func:`progress_session` is active. JSON results
stay on stdout. Bars use a known total (edges, nodes, types, steps) and advance
while that many items are processed.
"""

from __future__ import annotations

import sys
import time
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Iterator


class Progress:
    def __init__(self, enabled: bool, chunk: int = 1 << 20) -> None:
        self.enabled = enabled
        self.chunk = int(chunk)
        self._open = False
        self._bar = False
        self._label = ""
        self._t = 0.0
        self._drawn = 0.0

    def start(self, label: str) -> None:
        self.done()
        self._open = True
        self._bar = False
        self._label = label
        self._t = time.perf_counter()
        if self.enabled:
            sys.stderr.write(f"{label} ...\n")
            sys.stderr.flush()

    def tick(self, done: int, total: int, label: str, unit: str = "") -> None:
        if not self.enabled or total <= 0:
            return
        done_i = max(int(done), 0)
        total_i = int(total)
        finished = done_i >= total_i
        now = time.perf_counter()
        same = self._bar and self._label == label
        if same and not finished and (now - self._drawn) < 0.05:
            return
        if self._open and not same:
            self.done()
        if not self._bar:
            self._open = True
            self._bar = True
            self._label = label
            self._t = now
        self._drawn = now
        width = 28
        frac = min(max(done_i / total_i, 0.0), 1.0)
        filled = int(width * frac)
        bar = "#" * filled + "-" * (width - filled)
        suffix = f" {unit}" if unit else ""
        line = f"\r{label} [{bar}] {done_i:,}/{total_i:,}{suffix}"
        if finished:
            line += f"  {now - self._t:.1f}s\n"
            self._open = False
            self._bar = False
            self._label = ""
        sys.stderr.write(line)
        sys.stderr.flush()

    def done(self, detail: str = "") -> None:
        if not self._open:
            return
        was_bar = self._bar
        self._open = False
        self._bar = False
        self._label = ""
        if not self.enabled:
            return
        elapsed = time.perf_counter() - self._t
        suffix = f"  {detail}" if detail else ""
        if was_bar:
            sys.stderr.write("\n")
        sys.stderr.write(f"  {elapsed:.1f}s{suffix}\n")
        sys.stderr.flush()

    def close(self) -> None:
        self.done()


class _NullProgress:
    enabled = False
    chunk = 1 << 20

    def start(self, label: str) -> None:
        del label

    def tick(self, done: int, total: int, label: str, unit: str = "") -> None:
        del done, total, label, unit

    def done(self, detail: str = "") -> None:
        del detail

    def close(self) -> None:
        return None


_NULL = _NullProgress()
_current: ContextVar[Progress | _NullProgress] = ContextVar("nrde_progress", default=_NULL)


def get_progress() -> Progress | _NullProgress:
    return _current.get()


def iter_progress(
    total: int,
    label: str,
    unit: str = "",
    chunk: int | None = None,
) -> Iterator[tuple[int, int]]:
    """Yield ``[start, end)`` slices. Draw a bar when progress is on and ``total`` is large."""
    progress = get_progress()
    total = int(total)
    if total <= 0:
        return
    step = max(int(chunk if chunk is not None else progress.chunk), 1)
    if not progress.enabled or total <= step:
        yield 0, total
        return
    progress.tick(0, total, label, unit)
    done = 0
    while done < total:
        end = min(done + step, total)
        yield done, end
        done = end
        progress.tick(done, total, label, unit)


@contextmanager
def progress_session(*, enabled: bool | None = None) -> Iterator[Progress]:
    """Enable stage lines and bars for the current task. Default: a stderr TTY."""
    if enabled is None:
        enabled = sys.stderr.isatty()
    prog = Progress(enabled=bool(enabled))
    token = _current.set(prog)
    try:
        yield prog
    finally:
        prog.close()
        _current.reset(token)
