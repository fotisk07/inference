"""Benchmark the real FA4 kernel against SDPA backends on synthetic q/k/v.

Decoupled from the full Donut model -- isolates whether flash attention is
theoretically faster at a given shape, independent of model/HF call overhead.
Fixed at the real decoder's head count/head_dim so results are comparable to
what `--backends fa` vs `--backends sdpa` actually run in scripts/inference/bench_speed.py.

"decode" mode sets query_len=1 (matches generate()'s real per-step shape
against a growing KV cache -- the worst case for flash attention's tiling).
"prefill" mode sets query_len=kv_len with causal masking (the regime flash
attention is designed for).
"""

import resource

import itertools
from pathlib import Path
from typing import Literal

import torch
import torch.nn.functional as F
import typer
from prettytable import PrettyTable

from donut.accel import sdpa_backend
from donut.bench import time_fn
from donut.runio import parse_ints, save_record, resolve_device_dtype

SDPA_BACKENDS = ["math", "efficient", "flash", "cudnn"]

app = typer.Typer(add_completion=False)


def _make_qkv(batch_size, num_heads, q_len, kv_len, head_dim, dtype, device, seed):
    """q/k/v in flash_attn's native (batch, seqlen, nheads, headdim) layout."""
    gen = torch.Generator(device=device).manual_seed(seed)
    shape_q = (batch_size, q_len, num_heads, head_dim)
    shape_kv = (batch_size, kv_len, num_heads, head_dim)
    q = torch.randn(shape_q, generator=gen, device=device, dtype=dtype)
    k = torch.randn(shape_kv, generator=gen, device=device, dtype=dtype)
    v = torch.randn(shape_kv, generator=gen, device=device, dtype=dtype)
    return q, k, v


def _bench_fa4(q, k, v, causal: bool, n_warmup: int, n_runs: int) -> dict:
    from flash_attn.cute import flash_attn_func  # ty: ignore[unresolved-import]

    def fn():
        flash_attn_func(q, k, v, causal=causal)

    return time_fn(fn, n_warmup, n_runs, verbose=False)


def _bench_sdpa(
    q, k, v, causal: bool, backend: str, n_warmup: int, n_runs: int
) -> dict:
    # SDPA wants (batch, nheads, seqlen, headdim) -- transpose once, outside the timed fn.
    qt = q.transpose(1, 2).contiguous()
    kt = k.transpose(1, 2).contiguous()
    vt = v.transpose(1, 2).contiguous()

    def fn():
        with sdpa_backend(backend):
            F.scaled_dot_product_attention(qt, kt, vt, is_causal=causal)

    return time_fn(fn, n_warmup, n_runs, verbose=False)


@app.command()
def main(
    num_heads: int = 16,
    head_dim: int = 64,
    dtype: Literal["bf16", "f16", "f32"] = "bf16",
    device: str | None = None,
    kv_lens: str = "1,8,32,128,512,2048,4096",
    batch_sizes: str = "1,8,32",
    modes: str = "decode,prefill",
    n_runs: int = 20,
    n_warmup: int = 5,
    seed: int = 42,
    out: Path | None = None,
) -> None:
    device, torch_dtype = resolve_device_dtype(device, dtype)
    kv_len_list = parse_ints(kv_lens)
    batch_size_list = parse_ints(batch_sizes)
    mode_list = [m.strip() for m in modes.split(",") if m.strip()]
    kernels = ["fa4"] + [f"sdpa-{b}" for b in SDPA_BACKENDS]

    table = PrettyTable()
    table.field_names = ["mode", "kv_len", "bs", *kernels]
    records = []

    for mode, kv_len, bs in itertools.product(mode_list, kv_len_list, batch_size_list):
        q_len = 1 if mode == "decode" else kv_len
        causal = mode == "prefill"
        q, k, v = _make_qkv(
            bs, num_heads, q_len, kv_len, head_dim, torch_dtype, device, seed
        )

        row = [mode, kv_len, bs]
        rec = {"mode": mode, "kv_len": kv_len, "batch_size": bs, "kernels": {}}
        for kernel in kernels:
            try:
                if kernel == "fa4":
                    stats = _bench_fa4(q, k, v, causal, n_warmup, n_runs)
                else:
                    backend = kernel.removeprefix("sdpa-")
                    stats = _bench_sdpa(q, k, v, causal, backend, n_warmup, n_runs)
                row.append(stats["mean_ms"])
                rec["kernels"][kernel] = {
                    "status": "ok",
                    "mean_ms": stats["mean_ms"],
                    "std_ms": stats["std_ms"],
                }
            except Exception as e:
                row.append(f"n/a ({type(e).__name__})")
                rec["kernels"][kernel] = {"status": "error", "error": type(e).__name__}
        table.add_row(row)
        records.append(rec)

    print(table)

    if out is not None:
        meta = {
            "num_heads": num_heads,
            "head_dim": head_dim,
            "dtype": dtype,
            "device": device,
            "n_runs": n_runs,
            "n_warmup": n_warmup,
        }
        save_record(
            out, "bench_attention_kernels.json", {"meta": meta, "records": records}
        )


if __name__ == "__main__":
    app()
