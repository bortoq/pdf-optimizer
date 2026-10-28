"""Progress of one file as a fraction that only grows."""

from __future__ import annotations

from collections.abc import Callable
import threading

Callback = Callable[[float], None]


def nothing(fraction: float) -> None:
    """The report used when nobody listens."""


class Span:
    """A part of a progress: ``span(f)`` reports ``start + (end - start) * f``.

    Calls may come from several threads; the value passed on never goes down.
    """

    def __init__(self, report: Callback, start: float = 0.0, end: float = 1.0):
        self.report = report
        self.start = start
        self.end = end
        self._value = 0.0
        self._lock = threading.Lock()

    def __call__(self, fraction: float) -> None:
        fraction = min(1.0, max(0.0, fraction))
        with self._lock:
            if fraction <= self._value:
                return
            self._value = fraction
        self.report(self.start + (self.end - self.start) * fraction)

    def part(self, start: float, end: float) -> Span:
        """A sub-span covering ``start..end`` of this one."""
        return Span(self, start, end)

    def counter(self, total: int) -> Callable[[], None]:
        """A function to call once per finished item of ``total``."""
        lock = threading.Lock()
        done = [0]

        def tick() -> None:
            with lock:
                done[0] += 1
                fraction = done[0] / max(1, total)
            self(fraction)

        return tick
