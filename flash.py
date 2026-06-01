import torch
import time

# Try importing FlashAttention
try:
    from flash_attn import flash_attn_func
    HAS_FLASH = True
except ImportError:
    HAS_FLASH = False

device = "cuda"
dtype = torch.float16  # important for flash attention

# Config (tune these for your GPU)
B = 8        # batch size
S = 1024     # sequence length
H = 8        # number of heads
D = 64       # head dimension

print(f"Running on {device} with dtype={dtype}")
print(f"B={B}, S={S}, H={H}, D={D}")

# Create random QKV
q = torch.randn(B, S, H, D, device=device, dtype=dtype)
k = torch.randn(B, S, H, D, device=device, dtype=dtype)
v = torch.randn(B, S, H, D, device=device, dtype=dtype)

# ---- PyTorch attention ----
def torch_attention(q, k, v):
    return torch.nn.functional.scaled_dot_product_attention(
        q.transpose(1, 2),  # (B, H, S, D)
        k.transpose(1, 2),
        v.transpose(1, 2),
        is_causal=False
    ).transpose(1, 2)

# ---- FlashAttention ----
def flash_attention(q, k, v):
    return flash_attn_func(q, k, v, causal=False)

def benchmark(fn, name, iters=50, warmup=10):
    # Warmup
    for _ in range(warmup):
        out = fn(q, k, v)
    torch.cuda.synchronize()

    start = time.time()
    for _ in range(iters):
        out = fn(q, k, v)
    torch.cuda.synchronize()
    end = time.time()

    avg_ms = (end - start) / iters * 1000
    print(f"{name:20s}: {avg_ms:.2f} ms")

# ---- Run ----
benchmark(torch_attention, "PyTorch SDPA")

if HAS_FLASH:
    benchmark(flash_attention, "FlashAttention")
else:
    print("FlashAttention not installed")
