"""Faithfulness audit: does each accel preset change the model's output?

For every preset in donut.accel.PRESETS, applies it, runs the same synthetic
inputs through the encoder and generate(), and diffs against a one-time
unaccelerated (baseline) reference -- then reverts. `baseline` itself is
included as a built-in sanity check: it must diff to exactly zero against
itself, or the audit tool (not the model) has a bug.

Usage:
    uv run python scripts/inference/audit_accel.py          # real model, auto device
    uv run python scripts/inference/audit_accel.py --tiny   # offline smoke run
"""

from pathlib import Path
from typing import Literal

import torch
import typer
from prettytable import PrettyTable

from donut.accel import PRESETS, apply_accel, revert_accel
from donut.audit import diff_stats
from donut.constants import MODEL_ID
from donut.model import load_baseline_model
from donut.runio import resolve_device_dtype, run_meta, save_record
from donut.synthetic import make_pixel_values

app = typer.Typer(add_completion=False)


def _encode_and_generate(model, pixel_values, max_new_tokens: int):
    """Run encoder + generate() once. model left untyped -- nn.Module's stub
    resolves dynamic submodule access (.encoder) through a Tensor union,
    so a precisely-typed model makes type checkers flag .encoder(...) as
    calling a Tensor; bench.py's functions dodge this the same way.
    """
    with torch.no_grad():
        encoder_out = model.encoder(pixel_values, return_dict=True).last_hidden_state
        gen = model.generate(
            pixel_values=pixel_values,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            output_scores=True,
            return_dict_in_generate=True,
        )
    # First step, not last: once a sequence hits EOS, HF masks every other
    # vocab entry to -inf at later steps, and inf - inf = nan in the diff.
    return encoder_out, gen.sequences, gen.scores[0]


def audit_one_preset(
    model,
    *,
    preset: str,
    ref_encoder_out: torch.Tensor,
    ref_sequences: torch.Tensor,
    ref_first_logits: torch.Tensor,
    pixel_values: torch.Tensor,
    max_new_tokens: int,
) -> dict:
    """Apply one preset, diff its encoder + generate() output against the
    eager reference, then revert. Self-contained like bench_infer_step.
    """
    try:
        apply_accel(model, preset)
        encoder_out, sequences, first_logits = _encode_and_generate(
            model, pixel_values, max_new_tokens
        )
        return {
            "preset": preset,
            "status": "ok",
            "encoder": diff_stats(ref_encoder_out, encoder_out),
            "sequences_match": bool(torch.equal(ref_sequences, sequences)),
            "generate_logits": diff_stats(ref_first_logits, first_logits),
        }
    except Exception as e:
        return {"preset": preset, "status": "error", "error": str(e)}
    finally:
        revert_accel(model)


@app.command()
def main(
    model_id: str = MODEL_ID,
    device: str | None = None,
    dtype: Literal["bf16", "f16", "f32"] = "bf16",
    seed: int = 42,
    out: Path = Path("results/audit_accel"),
    tiny: bool = False,
    batch_size: int = 1,
    max_new_tokens: int = 32,
) -> None:
    device, torch_dtype = resolve_device_dtype(device, dtype)
    model, model_id = load_baseline_model(model_id, device, torch_dtype, tiny=tiny)
    pixel_values = make_pixel_values(model, batch_size=batch_size, seed=seed)

    ref_encoder_out, ref_sequences, ref_first_logits = _encode_and_generate(
        model, pixel_values, max_new_tokens
    )

    records = []
    for preset in PRESETS:
        record = audit_one_preset(
            model,
            preset=preset,
            ref_encoder_out=ref_encoder_out,
            ref_sequences=ref_sequences,
            ref_first_logits=ref_first_logits,
            pixel_values=pixel_values,
            max_new_tokens=max_new_tokens,
        )
        records.append(record)

    meta = run_meta(device, dtype, model_id)
    save_record(out, "audit_accel.json", {**meta, "records": records})
    print(f"wrote {out / 'audit_accel.json'}")

    table = PrettyTable()
    table.field_names = [
        "preset",
        "status",
        "enc max_ae",
        "enc cos_sim",
        "seq match",
        "logits max_ae",
    ]
    for r in records:
        if r["status"] == "ok":
            table.add_row(
                [
                    r["preset"],
                    r["status"],
                    round(r["encoder"]["max_ae"], 6),
                    round(r["encoder"]["cosine_sim"], 6),
                    r["sequences_match"],
                    round(r["generate_logits"]["max_ae"], 6),
                ]
            )
        else:
            table.add_row([r["preset"], "ERROR", "-", "-", "-", "-"])
    print(table)


if __name__ == "__main__":
    app()
