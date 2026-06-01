import torch
import time
import matplotlib.pyplot as plt
from flash_attn.cute import flash_attn_func

device = "cuda"
dtype = torch.float16

B = 8
H = 16
D = 64

seq_lengths = [256, 512, 1024, 2048, 4096, 8192, 16384]

def torch_attention(q, k, v):
    return torch.nn.functional.scaled_dot_product_attention(
        q.transpose(1, 2),
        k.transpose(1, 2),
        v.transpose(1, 2),
    )

def flash_attention(q, k, v):
    return flash_attn_func(q, k, v, causal=False)

def flash_attention_causal(q, k, v):
    return flash_attn_func(q, k, v, causal=True)

def benchmark(fn, q, k, v, iters=100, warmup=20):
    for _ in range(warmup):
        fn(q, k, v)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn(q, k, v)
    end.record()
    torch.cuda.synchronize()

    return start.elapsed_time(end) / iters

print(f"{'SeqLen':>8} | {'PyTorch SDPA':>14} | {'Flash':>10} | {'Flash Causal':>14} | {'Speedup Flash':>14} | {'Speedup Causal':>15}")
print("-" * 90)

results = []
for S in seq_lengths:
    q = torch.randn(B, S, H, D, device=device, dtype=dtype)
    k = torch.randn(B, S, H, D, device=device, dtype=dtype)
    v = torch.randn(B, S, H, D, device=device, dtype=dtype)

    t_sdpa   = benchmark(torch_attention,        q, k, v)
    t_flash  = benchmark(flash_attention,        q, k, v)
    t_causal = benchmark(flash_attention_causal, q, k, v)

    speedup_flash  = t_sdpa / t_flash
    speedup_causal = t_sdpa / t_causal

    results.append((S, t_sdpa, t_flash, t_causal, speedup_flash, speedup_causal))
    print(f"{S:>8} | {t_sdpa:>12.2f}ms | {t_flash:>8.2f}ms | {t_causal:>12.2f}ms | {speedup_flash:>13.2f}x | {speedup_causal:>14.2f}x")

print()
best = max(results, key=lambda r: r[4])
print(f"Peak Flash speedup:        {best[4]:.2f}x at S={best[0]}")
best_c = max(results, key=lambda r: r[5])
print(f"Peak Flash Causal speedup: {best_c[5]:.2f}x at S={best_c[0]}")

seq_lens = [r[0] for r in results]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(seq_lens, [r[1] for r in results], marker='o', label='PyTorch SDPA')
ax1.plot(seq_lens, [r[2] for r in results], marker='s', label='Flash')
ax1.plot(seq_lens, [r[3] for r in results], marker='^', label='Flash Causal')
ax1.set_xlabel('Sequence length')
ax1.set_ylabel('Latency (ms)')
ax1.set_title('Latency vs sequence length')
ax1.set_xscale('log', base=2)
ax1.set_yscale('log')
ax1.legend()
ax1.grid(True, which='both', alpha=0.3)

ax2.plot(seq_lens, [r[4] for r in results], marker='s', label='Flash')
ax2.plot(seq_lens, [r[5] for r in results], marker='^', label='Flash Causal')
ax2.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='Breakeven')
ax2.set_xlabel('Sequence length')
ax2.set_ylabel('Speedup vs PyTorch SDPA')
ax2.set_title('Speedup vs sequence length')
ax2.set_xscale('log', base=2)
ax2.legend()
ax2.grid(True, which='both', alpha=0.3)

plt.suptitle(f'FlashAttention scaling  —  B={B}, H={H}, D={D}', fontsize=13)
plt.tight_layout()
plt.show()
