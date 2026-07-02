"""Diagnose the SDPA-backend training NaN on big images (e.g. 2560x1920).

Symptom (reported): with backend="sdpa" and bf16/fp16 autocast, loss goes NaN on
the *second* step at large resolutions (>~2400px). backend="fa" and "baseline"
are fine; fp32 autocast is fine but slow.

Preset composition (donut/accel/__init__.py) narrows the suspect before we run a
thing: "sdpa" and "fa" share the SAME encoder SDPA patch and differ ONLY in the
decoder kernel, and "fa" works -> the decoder SDPA path is the prime suspect.

This script proves it by measurement and finds the fastest NaN-free fix:

    phase repro   -- reproduce + LOCATE (per-module fwd/bwd finiteness hooks,
                     per-SDPA-call input stats, per-step grad-norm / param-max
                     timeline). Says whether corruption is step-1 backward
                     (inf grad) or step-2 forward, and which SDPA call fires.
    phase bisect  -- decoder-kernel sweep, encoder-kernel sweep, resolution
                     sweep (find the >2400 threshold), precision sweep.
    phase fixes   -- apply candidate fixes, confirm NaN gone, time each vs bf16.
    phase all     -- run all three.

Run (GPU box):
    uv run python scripts/diagnose/nan_sdpa.py --phase all \
        --data-json test_data/train.json --n-samples 8 --batch-size 4
"""

import contextlib
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import typer
from prettytable import PrettyTable
from torch.utils.data import DataLoader

from donut.accel.decoder_fa import fa_available
from donut.accel.sdpa_backend import sdpa_backend
from donut.constants import DEFAULT_MAX_LENGTH, MODEL_ID
from donut.dataset import DonutDataset, load_samples, register_field_tokens
from donut.model import (
    encoder_image_size,
    fit_decoder_to_vocab,
    load_model,
    set_donut_shift_tokens,
    set_image_size,
)

_REPO_ROOT = Path(__file__).resolve().parents[3]
app = typer.Typer(add_completion=False)


# ── build / data ──────────────────────────────────────────────────────────────
def build(model_name: str, device: str, backend: str, height: int, width: int):
    """A big-image fine-tune-ready model+processor, mirroring train.build_model.

    fp32 master weights (dtype=float32) + the chosen accel backend; bf16/fp16 is
    applied at the forward via autocast, exactly like the real training loop.
    """
    model, processor = load_model(
        model_id=model_name, device=device, dtype=torch.float32, backend=backend
    )
    register_field_tokens(processor)
    fit_decoder_to_vocab(model, processor)
    set_donut_shift_tokens(model, processor)
    set_image_size(model, processor, height, width)
    return model, processor


def make_batches(processor, data_json: str, n_samples: int, batch_size: int, device):
    """A handful of REAL image batches, cycled to feed a few steps.

    Uses the same DonutDataset / processor path as training, so pixel_values are
    resized to the configured (big) resolution exactly as in the real run.
    """
    samples = load_samples(Path(data_json))[:n_samples]
    ds = DonutDataset(samples, processor, DEFAULT_MAX_LENGTH)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    batches = [(b["pixel_values"].to(device), b["labels"].to(device)) for b in loader]
    if not batches:
        raise RuntimeError(f"No samples loaded from {data_json}")
    return batches


def autocast_ctx(device: str, precision: str):
    """bf16/fp16 autocast on CUDA; a no-op for fp32 or CPU.

    No GradScaler on purpose: the user's fp16 run has none either, and a scaler
    would mask the very overflow we are hunting.
    """
    if precision == "fp32" or not device.startswith("cuda"):
        return contextlib.nullcontext()
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def _sync(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize()


# ── instrumentation ─────────────────────────────────────────────────────────--
_PHASE = "?"  # "encoder" | "decoder", set by the wrappers below
_SDPA_LOG: list[dict] = []


def _tag_submodel_phases(model):
    """Wrap encoder.forward / decoder.forward so every SDPA call knows who called it."""

    def wrap(mod, tag):
        orig = mod.forward

        def fwd(*a, **k):
            global _PHASE
            prev = _PHASE
            _PHASE = tag
            try:
                return orig(*a, **k)
            finally:
                _PHASE = prev

        mod.forward = fwd

    wrap(model.encoder, "encoder")
    wrap(model.decoder, "decoder")


def _patch_sdpa():
    """Log every scaled_dot_product_attention call: phase, |q|/|k|/|v| max, mask
    min, output finite? Catches BOTH the encoder patch (F.sdpa) and the decoder
    (transformers sdpa_attention_forward -> F.sdpa), since both resolve the name
    on torch.nn.functional at call time."""
    orig = F.scaled_dot_product_attention

    def wrapped(q, k, v, attn_mask=None, *args, **kwargs):
        out = orig(q, k, v, attn_mask, *args, **kwargs)
        rec = {
            "phase": _PHASE,
            "kv_len": k.shape[-2],
            "qmax": q.detach().abs().max().item(),
            "kmax": k.detach().abs().max().item(),
            "vmax": v.detach().abs().max().item(),
            "mask_min": (
                attn_mask.detach().min().item() if attn_mask is not None else None
            ),
            "out_finite": bool(torch.isfinite(out.detach()).all()),
        }
        _SDPA_LOG.append(rec)
        return out

    F.scaled_dot_product_attention = wrapped  # ty: ignore[invalid-assignment]
    return orig


def _unpatch_sdpa(orig):
    F.scaled_dot_product_attention = orig


def _tensors(obj):
    """Flatten the tensors out of a module output (Tensor / tuple / ModelOutput)."""
    if isinstance(obj, torch.Tensor):
        return [obj]
    if isinstance(obj, (tuple, list)):
        return [t for o in obj for t in _tensors(o)]
    if hasattr(obj, "to_tuple"):
        return _tensors(obj.to_tuple())
    return []


def _register_finite_hooks(model, state: dict):
    """First non-finite module output (fwd) and grad (bwd) per step -> state."""

    def fwd_hook(name):
        def hook(_m, _inp, out):
            if state["fwd"] is None:
                for t in _tensors(out):
                    if not torch.isfinite(t.detach()).all():
                        state["fwd"] = name
                        break

        return hook

    def bwd_hook(name):
        def hook(_m, grad_in, _grad_out):
            if state["bwd"] is None:
                for t in grad_in:
                    if t is not None and not torch.isfinite(t.detach()).all():
                        state["bwd"] = name
                        break

        return hook

    # Leaf modules only: they return plain Tensors, so the backward hook fires
    # cleanly (container ModelOutputs trip the "output should be a Tensor" warning
    # and are never the true origin anyway — the offending Linear/attn is a leaf).
    handles = []
    for name, mod in model.named_modules():
        if name and not list(mod.children()):
            handles.append(mod.register_forward_hook(fwd_hook(name)))
            handles.append(mod.register_full_backward_hook(bwd_hook(name)))
    return handles


# ── training step ─────────────────────────────────────────────────────────────
def train_steps(
    model, batches, device, precision, n_steps, grad_clip=1.0, skip_nan=False
):
    """Faithful copy of the real train step; returns a per-step record list."""
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    model.train()
    records = []
    for step in range(n_steps):
        pv, labels = batches[step % len(batches)]
        with autocast_ctx(device, precision):
            loss = model(pixel_values=pv, labels=labels).loss
        loss_finite = bool(torch.isfinite(loss.detach()))
        opt.zero_grad()
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        gnorm_finite = bool(torch.isfinite(gnorm))
        if not (skip_nan and not (loss_finite and gnorm_finite)):
            opt.step()
        pmax = max(
            (p.detach().abs().max().item() for p in model.parameters() if p.numel()),
            default=0.0,
        )
        records.append(
            {
                "step": step,
                "loss": loss.item(),
                "loss_finite": loss_finite,
                "grad_norm": float(gnorm),
                "grad_finite": gnorm_finite,
                "param_max": pmax,
            }
        )
    return records


def first_nan_step(records) -> int | None:
    for r in records:
        if not r["loss_finite"]:
            return r["step"]
    return None


# ── phase A: reproduce + locate ─────────────────────────────────────────────--
def phase_repro(model_name, data_json, device, height, width, batch_size, n_samples):
    print("\n" + "=" * 70)
    print(f"  PHASE REPRO  backend=sdpa  precision=bf16  {height}x{width}")
    print("=" * 70)
    model, processor = build(model_name, device, "sdpa", height, width)
    batches = make_batches(processor, data_json, n_samples, batch_size, device)

    h, w = encoder_image_size(model)
    with torch.no_grad(), autocast_ctx(device, "bf16"):
        enc = model.encoder(batches[0][0]).last_hidden_state
    print(f"  encoder tokens = {enc.shape[1]}  (image {h}x{w})")

    _tag_submodel_phases(model)
    orig = _patch_sdpa()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    model.train()
    try:
        for step in range(3):
            state = {"fwd": None, "bwd": None}
            handles = _register_finite_hooks(model, state)
            _SDPA_LOG.clear()
            pv, labels = batches[step % len(batches)]
            with autocast_ctx(device, "bf16"):
                loss = model(pixel_values=pv, labels=labels).loss
            loss_finite = bool(torch.isfinite(loss.detach()))
            opt.zero_grad()
            loss.backward()
            gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            for hd in handles:
                hd.remove()

            first_bad = next((r for r in _SDPA_LOG if not r["out_finite"]), None)
            print(f"\n  --- step {step} ---")
            print(
                f"  loss={loss.item():.4f} finite={loss_finite}  "
                f"grad_norm={float(gnorm):.3e} finite={bool(torch.isfinite(gnorm))}"
            )
            print(f"  first non-finite fwd module: {state['fwd']}")
            print(f"  first non-finite bwd module: {state['bwd']}")
            if first_bad:
                print(f"  first non-finite SDPA call:  {first_bad}")
            worst = max(
                (r for r in _SDPA_LOG),
                key=lambda r: max(r["qmax"], r["kmax"], r["vmax"]),
                default=None,
            )
            if worst:
                print(
                    f"  largest SDPA input:  phase={worst['phase']} "
                    f"kv_len={worst['kv_len']} qmax={worst['qmax']:.1f} "
                    f"kmax={worst['kmax']:.1f} vmax={worst['vmax']:.1f} "
                    f"mask_min={worst['mask_min']}"
                )
    finally:
        _unpatch_sdpa(orig)
    print(
        "\n  READ: if step-0 grad is non-finite -> corruption is in the backward, "
        "and step-1+ forward is all-NaN from poisoned weights (the 'second pass')."
    )


# ── phase B: bisect ─────────────────────────────────────────────────────────--
def _run_backend(
    model_name,
    batches_fn,
    device,
    backend,
    height,
    width,
    n_steps=3,
    precision="bf16",
    enc_kernel=None,
):
    model, processor = build(model_name, device, backend, height, width)
    batches = batches_fn(processor)
    ctx = sdpa_backend(enc_kernel) if enc_kernel else contextlib.nullcontext()
    with ctx:
        recs = train_steps(model, batches, device, precision, n_steps)
    del model
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return first_nan_step(recs)


def phase_bisect(model_name, data_json, device, height, width, batch_size, n_samples):
    def batches_fn(processor):
        return make_batches(processor, data_json, n_samples, batch_size, device)

    print("\n" + "=" * 70)
    print(f"  PHASE BISECT  {height}x{width}")
    print("=" * 70)

    # 1) decoder kernel sweep (encoder held at SDPA) -- isolates the culprit kernel
    print("\n  [1] decoder kernel sweep (bf16, encoder=SDPA):")
    t = PrettyTable(["decoder backend", "first NaN step", "verdict"])
    # sdpa_cudnn is the tell: plain "sdpa" lets PyTorch auto-pick, and cuDNN is
    # highest-priority in recent builds. If auto("sdpa") NaNs but every FORCED
    # single kernel is clean, cuDNN is what auto picked -> its bf16 backward at
    # large kv_len is the bug.
    decoders = ["sdpa", "sdpa_cudnn", "sdpa_math", "sdpa_efficient", "sdpa_flash", "fa"]
    for be in decoders:
        if be == "fa" and not fa_available():
            t.add_row([be, "-", "skipped (no flash-attn)"])
            continue
        try:
            step = _run_backend(model_name, batches_fn, device, be, height, width)
            t.add_row(
                [
                    be,
                    step if step is not None else "-",
                    "NaN" if step is not None else "clean",
                ]
            )
        except Exception as e:  # noqa: BLE001 - a kernel may reject a shape
            t.add_row([be, "err", str(e)[:40]])
    print(t)

    # 2) encoder kernel sweep: decoder=fa (no torch-sdpa) so sdpa_backend()
    #    scopes ONLY the encoder patch. Confirms the encoder is not the culprit.
    print("\n  [2] encoder kernel sweep (bf16, decoder=fa):")
    t = PrettyTable(["encoder SDPA kernel", "first NaN step", "verdict"])
    if not fa_available():
        print("    skipped: needs decoder=fa (flash-attn unavailable)")
    else:
        for k in ["math", "efficient", "flash"]:
            try:
                step = _run_backend(
                    model_name, batches_fn, device, "fa", height, width, enc_kernel=k
                )
                t.add_row(
                    [
                        k,
                        step if step is not None else "-",
                        "NaN" if step is not None else "clean",
                    ]
                )
            except Exception as e:  # noqa: BLE001
                t.add_row([k, "err", str(e)[:40]])
        print(t)

    # 3) resolution sweep (backend=sdpa, bf16) -- find the >2400 threshold
    print("\n  [3] resolution sweep (backend=sdpa, bf16):")
    t = PrettyTable(["h x w", "enc tokens", "first NaN step", "verdict"])
    for hh, ww in [(1280, 960), (1920, 1440), (2304, 1728), (2400, 1800), (2560, 1920)]:
        model, processor = build(model_name, device, "sdpa", hh, ww)
        b = make_batches(processor, data_json, n_samples, batch_size, device)
        with torch.no_grad(), autocast_ctx(device, "bf16"):
            ntok = model.encoder(b[0][0]).last_hidden_state.shape[1]
        step = first_nan_step(train_steps(model, b, device, "bf16", 3))
        t.add_row(
            [
                f"{hh}x{ww}",
                ntok,
                step if step is not None else "-",
                "NaN" if step is not None else "clean",
            ]
        )
        del model
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
    print(t)

    # 4) precision sweep at the big resolution
    print(f"\n  [4] precision sweep (backend=sdpa, {height}x{width}):")
    t = PrettyTable(["precision", "first NaN step", "verdict"])
    for prec in ["fp32", "bf16", "fp16"]:
        step = _run_backend(
            model_name, batches_fn, device, "sdpa", height, width, precision=prec
        )
        t.add_row(
            [
                prec,
                step if step is not None else "-",
                "NaN" if step is not None else "clean",
            ]
        )
    print(t)


# ── phase C: fixes ──────────────────────────────────────────────────────────--
def _register_fp32_decoder_softmax(model):
    """Fix 1: run ONLY the decoder attention math in fp32 (upcast q/k/v), keep the
    rest of the decoder in bf16. Registered as a custom attn_implementation."""
    from transformers import AttentionInterface
    from transformers.integrations.sdpa_attention import sdpa_attention_forward

    def fp32_forward(module, query, key, value, attention_mask, **kwargs):
        out, w = sdpa_attention_forward(
            module,
            query.float(),
            key.float(),
            value.float(),
            None if attention_mask is None else attention_mask.float(),
            **kwargs,
        )
        return out.to(query.dtype), w

    AttentionInterface.register("sdpa_fp32softmax", fp32_forward)
    model.decoder.config._attn_implementation = "sdpa_fp32softmax"


def _wrap_decoder_fp32(model):
    """Fix 2: run the WHOLE decoder outside autocast (fp32 compute on fp32 weights);
    encoder stays bf16."""
    orig = model.decoder.forward

    def up(x):
        # Encoder ran under autocast -> encoder_hidden_states arrive as bf16. With
        # autocast disabled the decoder's fp32 weights would mismatch them in
        # F.linear, so upcast every float tensor arg to fp32 first.
        return x.float() if torch.is_tensor(x) and x.is_floating_point() else x

    def fwd(*a, **k):
        a = tuple(up(x) for x in a)
        k = {kk: up(v) for kk, v in k.items()}
        with torch.autocast(device_type="cuda", enabled=False):
            return orig(*a, **k)

    model.decoder.forward = fwd


def phase_fixes(model_name, data_json, device, height, width, batch_size, n_samples):
    print("\n" + "=" * 70)
    print(f"  PHASE FIXES  {height}x{width}  (each vs bf16 baseline)")
    print("=" * 70)

    def timed(model, batches, precision, skip_nan=False):
        _sync(device)
        recs = train_steps(model, batches, device, precision, 4, skip_nan=skip_nan)
        _sync(device)
        t0 = time.perf_counter()
        recs = train_steps(model, batches, device, precision, 4, skip_nan=skip_nan)
        _sync(device)
        ms = (time.perf_counter() - t0) / 4 * 1e3
        return first_nan_step(recs), ms

    def fresh(backend="sdpa", precision="bf16", mutate=None, skip_nan=False):
        model, processor = build(model_name, device, backend, height, width)
        if mutate:
            mutate(model)
        batches = make_batches(processor, data_json, n_samples, batch_size, device)
        nan, ms = timed(model, batches, precision, skip_nan=skip_nan)
        del model
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
        return nan, ms

    # baselines
    bf16_nan, bf16_ms = fresh("sdpa", "bf16")
    fp32_nan, fp32_ms = fresh("sdpa", "fp32")

    rows = []
    rows.append(("sdpa bf16 (current, broken)", bf16_nan, bf16_ms))
    rows.append(("sdpa fp32 (safe, slow)", fp32_nan, fp32_ms))
    rows.append(
        (
            "fix1: decoder fp32 softmax",
            *fresh("sdpa", "bf16", _register_fp32_decoder_softmax),
        )
    )
    rows.append(
        ("fix2: whole decoder fp32", *fresh("sdpa", "bf16", _wrap_decoder_fp32))
    )
    # fix3: exclude cuDNN — the real fix. Full bf16 speed, no precision change.
    rows.append(("fix3: decoder sdpa_efficient", *fresh("sdpa_efficient", "bf16")))
    rows.append(("fix3b: decoder sdpa_flash", *fresh("sdpa_flash", "bf16")))
    rows.append(("fix3c: decoder sdpa_math", *fresh("sdpa_math", "bf16")))
    if fa_available():
        rows.append(("fix4: decoder fa", *fresh("fa", "bf16")))
    rows.append(("fix5: skip-NaN + clip guard", *fresh("sdpa", "bf16", skip_nan=True)))

    t = PrettyTable(["variant", "NaN?", "ms/step", "vs bf16", "vs fp32"])
    for name, nan, ms in rows:
        t.add_row(
            [
                name,
                "NaN" if nan is not None else "clean",
                f"{ms:.0f}",
                f"{ms / bf16_ms:.2f}x" if bf16_ms else "-",
                f"{ms / fp32_ms:.2f}x" if fp32_ms else "-",
            ]
        )
    print(t)

    # Exclude the two baselines AND the skip-NaN guard: skip-NaN reports "clean"
    # only because it drops every big-image step (grad is NaN -> optimizer.step
    # skipped), so it never actually learns from them — not a real fix.
    clean = [
        (n, ms)
        for n, nan, ms in rows
        if nan is None and not any(w in n for w in ("safe", "current", "skip-NaN"))
    ]
    if clean:
        best = min(clean, key=lambda r: r[1])
        print(
            f"\n  RECOMMEND fastest NaN-free fix: {best[0]}  ({best[1]:.0f} ms/step, "
            f"{best[1] / fp32_ms:.2f}x of fp32)"
        )


# ── CLI ───────────────────────────────────────────────────────────────────────
@app.command()
def main(
    phase: str = "all",
    model_name: str = MODEL_ID,
    data_json: str = str(_REPO_ROOT / "test_data" / "train.json"),
    image_height: int = 2560,
    image_width: int = 1920,
    batch_size: int = 4,
    n_samples: int = 8,
    device: str | None = None,
) -> None:
    """Investigate + fix the SDPA big-image training NaN."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if not device.startswith("cuda"):
        print("WARNING: NaN is CUDA/autocast-specific; results on CPU are meaningless.")
    args = (
        model_name,
        data_json,
        device,
        image_height,
        image_width,
        batch_size,
        n_samples,
    )
    if phase in ("repro", "all"):
        phase_repro(*args)
    if phase in ("bisect", "all"):
        phase_bisect(*args)
    if phase in ("fixes", "all"):
        phase_fixes(*args)


if __name__ == "__main__":
    app()
