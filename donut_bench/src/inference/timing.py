import time

import torch


def cuda_sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


class CudaTimer:
    """Wall-clock timer with cuda sync. Correct for both CPU and GPU."""

    def __init__(self):
        self._start = None

    def start(self):
        cuda_sync()
        self._start = time.perf_counter()

    def stop(self) -> float:
        cuda_sync()
        return (time.perf_counter() - self._start) * 1000.0  # ms


class LayerTimer:
    """Collects per-call GPU timing via CUDA event pairs attached as hooks."""

    def __init__(self, name: str):
        self.name = name
        self.events: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
        self._start: torch.cuda.Event | None = None

    def pre_hook(self, module, input):
        e = torch.cuda.Event(enable_timing=True)
        e.record()
        self._start = e

    def post_hook(self, module, input, output):
        e = torch.cuda.Event(enable_timing=True)
        e.record()
        self.events.append((self._start, e))

    def reset(self) -> None:
        self.events.clear()
        self._start = None

    def elapsed_no_sync(self) -> list[float]:
        """Read elapsed times. Caller must have called torch.cuda.synchronize() first."""
        return [s.elapsed_time(end) for s, end in self.events]
