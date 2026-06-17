"""Generate ablation.ipynb and pareto.ipynb from source cells.

Run once with `uv run python notebooks/_build_notebooks.py`. Kept in-repo so the
notebooks are reproducible/diffable rather than hand-edited JSON.
"""

import json
from pathlib import Path

HERE = Path(__file__).parent

_counter = [0]


def _next_id():
    _counter[0] += 1
    return f"cell{_counter[0]:02d}"


def md(text):
    return {
        "cell_type": "markdown",
        "id": _next_id(),
        "metadata": {},
        "source": text.strip("\n"),
    }


def code(text):
    return {
        "cell_type": "code",
        "id": _next_id(),
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": text.strip("\n"),
    }


def notebook(cells):
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


# ── ablation.ipynb — training quality vs image size / batch size ───────────────
LOAD = """
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# One JSON per fine-tuning run, written by train.py --ablation_out via the
# scripts/exp_*.sh sweeps.
RESULTS = Path("../results/ablation")

rows = []
for p in sorted(RESULTS.glob("*.json")):
    d = json.loads(p.read_text())
    if "f1_strict" not in d:  # skip non-run JSONs
        continue
    h, w = d["image_size"]
    rows.append(
        {
            "tag": p.stem,
            "image_label": f"{h}x{w}",
            "image_px": h * w,
            "batch_size": d["batch_size"],
            "lr": d["lr"],
            "backend": d["backend"],
            "n_train": d["n_train"],
            "final_val_loss": d["final_val_loss"],
            "docs_per_sec": d["docs_per_sec"],
            "f1_strict": d["f1_strict"],
            "f1_soft": d["f1_soft"],
            "perfect": d["quality"]["doc_stats"]["perfect"],
            "n_with_gt": d["quality"]["n_with_gt"],
        }
    )

df = pd.DataFrame(rows)
print(f"loaded {len(df)} runs from {RESULTS}")
df
"""

IMG_PLOT = """
# F1 vs image size — locate the smallest resolution that still reads the document.
img = df[df.tag.str.startswith("imgsize_")].sort_values("image_px")
if img.empty:
    print("No imgsize_* runs — run scripts/exp_image_size.sh (DATA_JSON=... DEVICE=cuda).")
else:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(img.image_px, img.f1_strict, marker="o", label="F1 strict")
    ax.plot(img.image_px, img.f1_soft, marker="s", label="F1 soft")
    for _, r in img.iterrows():
        ax.annotate(
            r.image_label,
            (r.image_px, (r.f1_strict if r.f1_strict is not None else 0)),
            textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8,
        )
    ax.set_xlabel("image pixels (H x W)")
    ax.set_ylabel("micro F1")
    ax.set_title("Field-extraction F1 vs image size")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.show()
"""

BATCH_PLOT = """
# F1 and training throughput vs batch size.
bat = df[df.tag.str.startswith("batch_")].sort_values("batch_size")
if bat.empty:
    print("No batch_* runs — run scripts/exp_batch_size.sh (DATA_JSON=... DEVICE=cuda).")
else:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(bat.batch_size, bat.f1_strict, marker="o", label="F1 strict")
    axes[0].plot(bat.batch_size, bat.f1_soft, marker="s", label="F1 soft")
    axes[0].set_ylabel("micro F1")
    axes[0].set_title("F1 vs batch size")
    axes[0].legend()
    axes[1].plot(bat.batch_size, bat.docs_per_sec, marker="o", color="tab:green")
    axes[1].set_ylabel("train docs / s")
    axes[1].set_title("Training throughput vs batch size")
    for ax in axes:
        ax.set_xlabel("batch size")
        ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
"""

TABLE = """
# All runs ranked by strict F1. Top row = best field extraction so far.
cols = [
    "tag", "image_label", "batch_size", "f1_strict", "f1_soft",
    "perfect", "n_with_gt", "final_val_loss", "docs_per_sec",
]
df.sort_values("f1_strict", ascending=False, na_position="last")[cols]
"""

ablation = notebook(
    [
        md(
            "# Donut training ablation — quality\n\n"
            "Field-level F1 from the `scripts/exp_*.sh` sweeps (one JSON per run in\n"
            "`results/ablation/`, written by `train.py --ablation_out`).\n\n"
            "- **F1 vs image size** — how much resolution training needs.\n"
            "- **F1 / throughput vs batch size** — the most effective batch.\n\n"
            "Join these with inference latency in `pareto.ipynb`."
        ),
        code(LOAD),
        md("## F1 vs image size"),
        code(IMG_PLOT),
        md("## F1 and throughput vs batch size"),
        code(BATCH_PLOT),
        md("## All runs ranked by F1"),
        code(TABLE),
    ]
)

# ── pareto.ipynb — quality (F1) vs inference latency, joined on image size ──────
PARETO_LOAD = """
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# Quality: one JSON per fine-tuning run (this package).
Q_DIR = Path("../results/ablation")
# Speed: bench_speed.json from the donut inference package.
SPEED_JSON = Path("../../donut/scripts/results/ablation/bench_speed.json")

qrows = []
for p in sorted(Q_DIR.glob("*.json")):
    d = json.loads(p.read_text())
    if "f1_strict" not in d:
        continue
    h, w = d["image_size"]
    qrows.append(
        {"image_label": f"{h}x{w}", "image_px": h * w,
         "f1_strict": d["f1_strict"], "f1_soft": d["f1_soft"]}
    )
quality = pd.DataFrame(qrows)

speed = pd.read_json(SPEED_JSON) if False else None  # see next cell
print(f"quality runs: {len(quality)}; speed file exists: {SPEED_JSON.exists()}")
quality
"""

PARETO_SPEED = """
# Load inference speed records and reduce to one latency per image size.
# Pick the backend you intend to ship (default: fastest non-baseline at bs=1).
raw = json.loads(SPEED_JSON.read_text())
sp = pd.json_normalize(raw["records"])
sp["image_label"] = sp.image_height.astype(str) + "x" + sp.image_width.astype(str)
sp["image_px"] = sp.image_height * sp.image_width

SHIP_BATCH = 1
cand = sp[(sp.backend != "baseline") & (sp.batch_size == SHIP_BATCH)].copy()
# fastest backend per image size (lowest generate latency)
best = cand.loc[cand.groupby("image_label")["generate.mean_ms"].idxmin()]
speed = best[
    ["image_label", "image_px", "backend", "generate.mean_ms",
     "throughput.images_per_s"]
].rename(columns={"generate.mean_ms": "latency_ms", "throughput.images_per_s": "img_per_s"})
speed
"""

PARETO_PLOT = """
# Join quality and speed on image size → F1 vs inference latency Pareto.
m = quality.merge(speed, on=["image_label", "image_px"], how="inner").sort_values("latency_ms")
if m.empty:
    print("No overlap. Use the SAME HxW values in exp_image_size.sh and run_ablation.sh.")
else:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(m.latency_ms, m.f1_strict, marker="o")
    for _, r in m.iterrows():
        ax.annotate(
            f"{r.image_label}\\n({r.backend})",
            (r.latency_ms, (r.f1_strict if r.f1_strict is not None else 0)),
            textcoords="offset points", xytext=(6, 6), fontsize=8,
        )
    ax.set_xlabel("inference latency  (generate ms, lower is better)")
    ax.set_ylabel("micro F1 strict (higher is better)")
    ax.set_title("Pareto: accuracy vs inference latency, by image size")
    ax.grid(alpha=0.3)
    plt.show()
    m
"""

PARETO_TABLE = """
# Decision: smallest/fastest image meeting an F1 floor.
F1_FLOOR = 0.90  # set to your acceptable accuracy
f1 = pd.to_numeric(m.f1_strict, errors="coerce").fillna(0.0)
ok = m[f1 >= F1_FLOOR].sort_values("latency_ms")
print(f"Configs with F1_strict >= {F1_FLOOR}:")
if ok.empty:
    print("  none yet — lower the floor or train larger images/longer.")
    m.sort_values("f1_strict", ascending=False)
else:
    print(f"  recommended: {ok.iloc[0].image_label} via {ok.iloc[0].backend} "
          f"({ok.iloc[0].latency_ms:.1f} ms, F1={ok.iloc[0].f1_strict:.3f})")
    ok
"""

pareto = notebook(
    [
        md(
            "# Pareto: accuracy vs inference latency\n\n"
            "Joins training **F1** (`results/ablation/*.json`, this package) with\n"
            "inference **latency** (`donut/scripts/results/ablation/bench_speed.json`)\n"
            "on **image size** — the knob that sets both. Pick the knee: the\n"
            "smallest/fastest image that still hits your accuracy floor.\n\n"
            "Run both sweeps first, with the **same** HxW values:\n"
            "`donut/scripts/run_ablation.sh` and `scripts/exp_image_size.sh`."
        ),
        code(PARETO_LOAD),
        code(PARETO_SPEED),
        md("## F1 vs inference latency"),
        code(PARETO_PLOT),
        md("## Decision table"),
        code(PARETO_TABLE),
    ]
)


def write(name, nb):
    # nbformat wants source as a list of lines (or a string; lists are canonical).
    for cell in nb["cells"]:
        if isinstance(cell["source"], str):
            cell["source"] = cell["source"].splitlines(keepends=True)
    (HERE / name).write_text(json.dumps(nb, indent=1, ensure_ascii=False))
    print(f"wrote {HERE / name}")


write("ablation.ipynb", ablation)
write("pareto.ipynb", pareto)
