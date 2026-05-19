from __future__ import annotations

from dataclasses import dataclass


@dataclass
class MemorySnapshot:
    allocated_mb: float
    reserved_mb: float
    peak_allocated_mb: float
    peak_reserved_mb: float


_ZERO = MemorySnapshot(0.0, 0.0, 0.0, 0.0)


def reset_memory_stats(device: str) -> None:
    """Reset peak memory accumulators so each run's peak is measured in isolation."""
    if device == "cuda":
        import torch

        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()


def capture_memory_snapshot(device: str) -> MemorySnapshot:
    """Read current and peak GPU memory from the CUDA caching allocator."""
    if device != "cuda":
        return _ZERO

    import torch

    stats = torch.cuda.memory_stats()
    return MemorySnapshot(
        allocated_mb=stats["allocated_bytes.all.current"] / 1024**2,
        reserved_mb=stats["reserved_bytes.all.current"] / 1024**2,
        peak_allocated_mb=stats["allocated_bytes.all.peak"] / 1024**2,
        peak_reserved_mb=stats["reserved_bytes.all.peak"] / 1024**2,
    )
