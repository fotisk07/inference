"""Benchmark the real FA4 kernel against SDPA backends on synthetic q/k/v.

Decoupled from the full Donut model -- isolates whether flash attention is
theoretically faster at a given shape, independent of model/HF call overhead.
Fixed at the real decoder's head count/head_dim so results are comparable to
what `--backends fa` vs `--backends sdpa` actually run in scripts/bench_speed.py.

"decode" mode sets query_len=1 (matches generate()'s real per-step shape
against a growing KV cache -- the worst case for flash attention's tiling).
"prefill" mode sets query_len=kv_len with causal masking (the regime flash
attention is designed for).
"""

import itertools
from typing import Literal

import torch
import torch.nn.functional as F
import typer
from prettytable import PrettyTable

from donut.accel import sdpa_backend
from donut.bench import time_fn

DTYPES = {"bf16": torch.bfloat16, "f16": torch.float16, "f32": torch.float32}
SDPA_BACKENDS = ["math", "efficient", "flash", "cudnn"]

app = typer.Typer()


def _parse_ints(s: str) -> list[int]:
    return [int(tok.strip()) for tok in s.split(",") if tok.strip()]


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
    from flash_attn.cute import flash_attn_func

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
    device: str = "cuda",
    kv_lens: str = "1,8,32,128,512,2048,4096",
    batch_sizes: str = "1,8,32",
    modes: str = "decode,prefill",
    n_runs: int = 20,
    n_warmup: int = 5,
    seed: int = 42,
) -> None:
    torch_dtype = DTYPES[dtype]
    kv_len_list = _parse_ints(kv_lens)
    batch_size_list = _parse_ints(batch_sizes)
    mode_list = [m.strip() for m in modes.split(",") if m.strip()]
    kernels = ["fa4"] + [f"sdpa-{b}" for b in SDPA_BACKENDS]

    table = PrettyTable()
    table.field_names = ["mode", "kv_len", "bs", *kernels]

    for mode, kv_len, bs in itertools.product(mode_list, kv_len_list, batch_size_list):
        q_len = 1 if mode == "decode" else kv_len
        causal = mode == "prefill"
        q, k, v = _make_qkv(
            bs, num_heads, q_len, kv_len, head_dim, torch_dtype, device, seed
        )

        row = [mode, kv_len, bs]
        for kernel in kernels:
            try:
                if kernel == "fa4":
                    stats = _bench_fa4(q, k, v, causal, n_warmup, n_runs)
                else:
                    backend = kernel.removeprefix("sdpa-")
                    stats = _bench_sdpa(q, k, v, causal, backend, n_warmup, n_runs)
                row.append(stats["mean_ms"])
            except Exception as e:
                row.append(f"n/a ({type(e).__name__})")
        table.add_row(row)

    print(table)


if __name__ == "__main__":
    app()
