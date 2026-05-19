from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Protocol


class PhaseTimer(Protocol):
    """Common interface for CUDA-event and CPU-clock timers."""

    def start(self) -> None: ...
    def stop(self) -> None: ...
    def elapsed_ms(self) -> float: ...


@dataclass
class CUDAPhaseTimer:
    """Measures wall time between start/stop using paired CUDA events.

    elapsed_ms() is only valid after torch.cuda.synchronize() has been called.
    """

    _start_event: object = field(init=False)
    _end_event: object = field(init=False)

    def __post_init__(self) -> None:
        import torch

        self._start_event = torch.cuda.Event(enable_timing=True)
        self._end_event = torch.cuda.Event(enable_timing=True)

    def start(self) -> None:
        self._start_event.record()

    def stop(self) -> None:
        self._end_event.record()

    def elapsed_ms(self) -> float:
        return self._start_event.elapsed_time(self._end_event)


@dataclass
class CPUPhaseTimer:
    """Fallback timer using time.perf_counter() for CPU-only runs."""

    _t0: float = field(default=0.0, init=False)
    _t1: float = field(default=0.0, init=False)

    def start(self) -> None:
        self._t0 = time.perf_counter()

    def stop(self) -> None:
        self._t1 = time.perf_counter()

    def elapsed_ms(self) -> float:
        return (self._t1 - self._t0) * 1000.0


def make_timer(device: str) -> PhaseTimer:
    """Return the appropriate timer for the given device."""
    if device == "cuda":
        return CUDAPhaseTimer()
    return CPUPhaseTimer()
