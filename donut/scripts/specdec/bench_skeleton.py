"""Benchmark the skeleton speculative-decoding proposer against vanilla greedy.

One run measures, on real images (docs/speculative-decoding.md protocol):
  1. exact-match gate  — assisted output ids == vanilla greedy ids, per doc
  2. mechanism         — acceptance rate, tokens per verify step (specdec.last_stats)
  3. end-to-end        — ms/doc at bs=1, vanilla vs assisted (CUDA-synced)
  4. vanilla frontier  — docs/s at --batch-sizes, since assisted is bs=1-only

Real acceptance/timing numbers need a fine-tuned checkpoint; --register-vocab
makes the script runnable on the base model (machinery + gate only, the
untrained model emits junk so mechanism metrics say nothing there).
"""

import time
from datetime import datetime
from pathlib import Path

import torch
import typer
from PIL import Image
from prettytable import PrettyTable
from tqdm import tqdm

from donut.bench import _cuda_sync
from donut.constants import DEFAULT_MAX_NEW_TOKENS, GLOBAL_OUT_DIR, MODEL_ID
from donut.dataset import load_samples, register_field_tokens
from donut.model import (
    decoder_start_ids,
    fit_decoder_to_vocab,
    load_model,
    set_donut_shift_tokens,
)
from donut.runio import parse_ints, run_meta, save_record
from donut.specdec import (
    apply_specdec_skeleton,
    check_specdec,
    last_stats,
    revert_specdec,
)

# Repo root (…/inference) so the default --data-json resolves no matter the CWD.
_REPO_ROOT = Path(__file__).resolve().parents[3]


def _pixel_tensors(samples: list[dict], processor) -> list[torch.Tensor]:
    """Preprocess every sample image once — (1, 3, H, W) each, kept on CPU."""
    tensors = []
    for sample in tqdm(samples, desc="preprocessing"):
        image = Image.open(sample["image"]).convert("RGB")
        tensors.append(processor(image, return_tensors="pt").pixel_values)
    return tensors


def _generate_pass(
    model, pixels: list[torch.Tensor], *, device, dtype, max_new_tokens, n_warmup
) -> list[dict]:
    """Per-doc greedy generate at bs=1 with CUDA-synced timing.

    Returns [{"ids": list[int], "ms": float, "stats": dict | None}]; stats is
    filled only when specdec is active (last_stats returns None otherwise).
    """
    warmup = pixels[0].to(device=device, dtype=dtype)
    for _ in range(n_warmup):
        with torch.no_grad():
            model.generate(
                warmup,
                decoder_input_ids=decoder_start_ids(model),
                max_new_tokens=max_new_tokens,
            )

    results = []
    for pv in tqdm(pixels, desc="generating"):
        pv = pv.to(device=device, dtype=dtype)
        _cuda_sync()
        t0 = time.perf_counter()
        with torch.no_grad():
            output_ids = model.generate(
                pv,
                decoder_input_ids=decoder_start_ids(model),
                max_new_tokens=max_new_tokens,
            )
        _cuda_sync()
        results.append(
            {
                "ids": output_ids[0].tolist(),
                "ms": round((time.perf_counter() - t0) * 1000, 3),
                "stats": last_stats(model),
            }
        )
    return results


def _vanilla_frontier(
    model, pixels: list[torch.Tensor], batch_sizes, *, device, dtype, max_new_tokens
) -> list[dict]:
    """docs/s of vanilla batched greedy — the throughput bar bs=1 assisted must beat."""
    frontier = []
    for bs in batch_sizes:
        batches = [
            torch.cat(pixels[i : i + bs]).to(device=device, dtype=dtype)
            for i in range(0, len(pixels), bs)
        ]
        with torch.no_grad():  # one warmup batch
            model.generate(
                batches[0],
                decoder_input_ids=decoder_start_ids(model, batches[0].size(0)),
                max_new_tokens=max_new_tokens,
            )
        _cuda_sync()
        t0 = time.perf_counter()
        for batch in batches:
            with torch.no_grad():
                model.generate(
                    batch,
                    decoder_input_ids=decoder_start_ids(model, batch.size(0)),
                    max_new_tokens=max_new_tokens,
                )
        _cuda_sync()
        elapsed = time.perf_counter() - t0
        frontier.append({"batch_size": bs, "docs_s": round(len(pixels) / elapsed, 3)})
    return frontier


app = typer.Typer(add_completion=False)


@app.command()
def main(
    checkpoint: str = MODEL_ID,
    data_json: str = str(_REPO_ROOT / "test_data" / "train.json"),
    out: Path = GLOBAL_OUT_DIR / "results" / "specdec",
    backend: str = "sdpa",
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    batch_sizes: str = "1,2,4,8",
    max_draft: int = 8,
    missing_chain: bool = True,
    n_warmup: int = 2,
    device: str | None = None,
    register_vocab: bool = typer.Option(
        False,
        help="run on a non-fine-tuned model: register the field vocab first "
        "(machinery/gate check only — mechanism metrics are meaningless untrained)",
    ),
) -> None:
    """Skeleton specdec vs vanilla greedy: exact match, acceptance, ms/doc, frontier."""
    model, processor = load_model(model_id=checkpoint, device=device, backend=backend)
    device = next(model.parameters()).device.type
    dtype = next(model.parameters()).dtype

    if register_vocab:
        print("register-vocab: untrained vocab — mechanism metrics are meaningless")
        register_field_tokens(processor)
        fit_decoder_to_vocab(model, processor)
        set_donut_shift_tokens(model, processor)

    samples = load_samples(Path(data_json))
    pixels = _pixel_tensors(samples, processor)
    common = dict(
        device=device, dtype=dtype, max_new_tokens=max_new_tokens, n_warmup=n_warmup
    )

    print(f"\nPass A — vanilla greedy, bs=1 ({len(pixels)} docs)")
    vanilla = _generate_pass(model, pixels, **common)

    print("\nPass B — skeleton assisted, bs=1")
    apply_specdec_skeleton(
        model, processor, max_draft=max_draft, missing_chain=missing_chain
    )
    check_specdec(model)
    assisted = _generate_pass(model, pixels, **common)
    revert_specdec(model)

    matches = [v["ids"] == a["ids"] for v, a in zip(vanilla, assisted)]
    exact_match_rate = sum(matches) / len(matches)

    print(f"\nVanilla batched frontier (bs={batch_sizes})")
    frontier = _vanilla_frontier(
        model,
        pixels,
        parse_ints(batch_sizes),
        device=device,
        dtype=dtype,
        max_new_tokens=max_new_tokens,
    )

    n = len(pixels)
    ms_vanilla = sum(r["ms"] for r in vanilla) / n
    ms_assisted = sum(r["ms"] for r in assisted) / n
    stats = [r["stats"] for r in assisted]
    proposed = sum(s["proposed"] for s in stats)
    accepted = sum(s["accepted"] for s in stats)
    aggregates = {
        "exact_match_rate": exact_match_rate,
        "ms_doc_vanilla": round(ms_vanilla, 3),
        "ms_doc_assisted": round(ms_assisted, 3),
        "speedup": round(ms_vanilla / ms_assisted, 3),
        "acceptance_rate": round(accepted / proposed, 4) if proposed else None,
        "mean_tokens_per_step": round(
            sum(s["new_tokens"] for s in stats) / sum(s["steps"] for s in stats), 3
        ),
        "assisted_docs_s_bs1": round(1000 / ms_assisted, 3),
    }

    meta = run_meta(device, None, checkpoint)
    meta["dtype"] = str(dtype).removeprefix("torch.")
    meta["backend"] = backend
    record = {
        "meta": meta,
        "config": {
            "checkpoint": checkpoint,
            "data_json": data_json,
            "backend": backend,
            "max_new_tokens": max_new_tokens,
            "max_draft": max_draft,
            "missing_chain": missing_chain,
            "n_warmup": n_warmup,
            "register_vocab": register_vocab,
            "n_docs": n,
        },
        "aggregates": aggregates,
        "frontier": frontier,
        "per_doc": [
            {
                "image": str(s["image"]),
                "exact_match": m,
                "ms_vanilla": v["ms"],
                "ms_assisted": a["ms"],
                "stats": a["stats"],
            }
            for s, m, v, a in zip(samples, matches, vanilla, assisted)
        ],
    }
    name = f"specdec__{Path(checkpoint).name}__{Path(data_json).stem}__{datetime.now():%Y%m%d-%H%M%S}.json"
    save_record(out, name, record)
    print(f"Saved record → {out / name}")

    table = PrettyTable()
    table.field_names = ["metric", "value"]
    for key, value in aggregates.items():
        table.add_row([key, value])
    for f in frontier:
        table.add_row([f"vanilla docs/s @bs={f['batch_size']}", f["docs_s"]])
    print(table)

    if exact_match_rate != 1.0:
        raise SystemExit(
            f"exact-match gate FAILED: {exact_match_rate:.3f} — timing numbers invalid"
        )


if __name__ == "__main__":
    app()
