"""
Document extraction metrics — importable from Jupyter or called by predict.py.

The central data structure is a list of result dicts, one per document:

    results = [
        {
            "image": "path/to/image.jpg",   # str — for reference / filtering
            "pred": {"E-mail": "a@b.com", "valor_da_nota": "100"},  # model output
            "gt":   {"E-mail": "a@b.com", "valor_da_nota": "99"},   # ground truth
        },
        ...
    ]

"gt" may be empty ({}) if you're running inference without labels — metric
functions that require ground truth will skip those documents gracefully.

── Classification ───────────────────────────────────────────────────────────────
For each (document, field) pair in the vocabulary we assign one of four labels:

  TP  (True Positive)   GT has value  ∧  model predicted  ∧  value is correct
  FP  (False Positive)  GT has NO value  ∧  model predicted something
                        → "predicting a box when there is nothing"
  FN  (False Negative)  GT has value  ∧  model was wrong or silent
                        → "bad / missing prediction" — covers both:
                            (a) model predicted the wrong value
                            (b) model predicted nothing at all
  TN  (True Negative)   GT has NO value  ∧  model predicted nothing
                        → "correctly abstaining"

The key insight: a wrong value is a FALSE NEGATIVE, not a false positive.
The model failed to produce the correct answer — that is a negative outcome.
FP is reserved for predicting a field that has nothing to predict.

From these four counts:

  precision  = TP / (TP + FP)   — of what the model output, how much was right?
  recall     = TP / (TP + FN)   — of what existed, how much did the model get?
  f1         = 2·P·R / (P + R)

── Matching modes ───────────────────────────────────────────────────────────────
All functions accept a `soft` keyword argument (default False).

  soft=False (strict):  exact string comparison
  soft=True  (lenient): compare normalize(pred) == normalize(gt)
                        normalize() lowercases and strips leading/trailing
                        whitespace — edit it to change what "lenient" means.
"""

from __future__ import annotations

from collections import defaultdict

from dataset import FIELD_TOKENS as _FIELD_TOKENS

# Leaf field names derived from the token vocabulary, e.g. "<E-mail>" → "E-mail".
# Used to enumerate every possible field for a document, which is required to
# compute TN (fields absent in both GT and prediction).
_VOCAB: list[str] = [tok[1:-1] for tok in _FIELD_TOKENS]


# ── Normalization ─────────────────────────────────────────────────────────────


def normalize(value: str) -> str:
    """
    Canonical form used for soft matching (soft=True).

    Current rules: strip leading/trailing whitespace, then lowercase.

    Add more rules here as needed, e.g.:
      - collapse internal whitespace:  " ".join(value.split())
      - strip punctuation:             re.sub(r"[^\w\s]", "", value)
      - remove accents:                unicodedata.normalize("NFD", value)...
    """
    return value.strip().lower()


def _match(a: str, b: str, soft: bool) -> bool:
    """Return True if values a and b are considered equal under the chosen mode."""
    if soft:
        return normalize(a) == normalize(b)
    return a == b


# ── Per-document field classification ─────────────────────────────────────────


def classify_fields(
    pred: dict[str, str], gt: dict[str, str], soft: bool = False
) -> dict:
    """
    Classify every field in the vocabulary for a single document.

    Iterates over every field the model knows about (_VOCAB) so that TN is
    correctly counted (fields where both GT and prediction are empty/absent).

    Parameters
    ----------
    pred : model predictions — {field_name: predicted_value}  (empty = absent)
    gt   : ground truth      — {field_name: true_value}       (empty = absent)
    soft : if True, use normalize() for value comparison

    Returns
    -------
    dict with integer counts and field-name lists for each class:

        tp      / tp_fields    — correct predictions
        fp      / fp_fields    — predicted something when GT had nothing
        fn      / fn_fields    — GT had value but model was wrong or silent
        tn      / tn_fields    — GT had nothing and model correctly said nothing
    """
    tp_fields: list[str] = []
    fp_fields: list[str] = []
    fn_fields: list[str] = []
    tn_fields: list[str] = []

    for field in _VOCAB:
        gt_val = gt.get(field, "")
        pred_val = pred.get(field, "")

        has_gt = bool(gt_val)
        has_pred = bool(pred_val)

        if has_gt and has_pred:
            # Both present — TP if values match, FN if they don't
            if _match(pred_val, gt_val, soft):
                tp_fields.append(field)
            else:
                # Wrong value: model tried but got it wrong → False Negative
                fn_fields.append(field)
        elif has_gt and not has_pred:
            # GT has a value but model said nothing → False Negative (silent miss)
            fn_fields.append(field)
        elif not has_gt and has_pred:
            # Nothing to predict but model output something → False Positive
            fp_fields.append(field)
        else:
            # Nothing to predict and model correctly said nothing → True Negative
            tn_fields.append(field)

    return {
        "tp": len(tp_fields),
        "fp": len(fp_fields),
        "fn": len(fn_fields),
        "tn": len(tn_fields),
        "tp_fields": tp_fields,
        "fp_fields": fp_fields,
        "fn_fields": fn_fields,
        "tn_fields": tn_fields,
    }


# ── Per-field aggregate stats across all documents ────────────────────────────


def field_stats(results: list[dict], soft: bool = False) -> dict[str, dict]:
    """
    Compute precision / recall / F1 / counts per field across all labelled docs.

    Only documents where result["gt"] is non-empty are included.

    Returns
    -------
    dict keyed by field_name (one entry per field in the vocabulary), each value:
        precision   float | None   TP / (TP + FP)
        recall      float | None   TP / (TP + FN)
        f1          float | None   harmonic mean of precision and recall
        tp, fp, fn, tn   int       raw counts across all documents

    precision/recall are None when the denominator is 0 (e.g. a field never
    appears in GT across the whole dataset → recall denominator is always 0).
    """
    counts: dict[str, dict[str, int]] = defaultdict(
        lambda: {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    )

    for r in results:
        if not r.get("gt"):
            continue
        clf = classify_fields(r["pred"], r["gt"], soft=soft)
        for field in clf["tp_fields"]:
            counts[field]["tp"] += 1
        for field in clf["fp_fields"]:
            counts[field]["fp"] += 1
        for field in clf["fn_fields"]:
            counts[field]["fn"] += 1
        for field in clf["tn_fields"]:
            counts[field]["tn"] += 1

    stats: dict[str, dict] = {}
    for field in _VOCAB:
        c = counts[field]

        prec_denom = c["tp"] + c["fp"]
        rec_denom = c["tp"] + c["fn"]

        precision = c["tp"] / prec_denom if prec_denom > 0 else None
        recall = c["tp"] / rec_denom if rec_denom > 0 else None

        if precision is not None and recall is not None and (precision + recall) > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = None

        stats[field] = {"precision": precision, "recall": recall, "f1": f1, **c}

    return stats


# ── Document-level summary ────────────────────────────────────────────────────


def doc_stats(results: list[dict], soft: bool = False) -> dict:
    """
    Categorize each document and return aggregate counts.

    Categories (mutually exclusive):
        perfect      FP == 0  and  FN == 0  — all extractions correct, no hallucinations
        fn_only      FN > 0   and  FP == 0  — some fields wrong/missing, no hallucinations
        fp_only      FP > 0   and  FN == 0  — hallucinations only, all GT fields correct
        mixed        FP > 0   and  FN > 0   — both wrong/missing and hallucinations

    Each entry in per_doc carries the raw clf dict for Jupyter-side slicing:
        {"image": ..., "category": ..., "tp": ..., "fp": ..., "fn": ..., "tn": ...}
    """
    per_doc = []
    category_counts: dict[str, int] = defaultdict(int)
    n_with_gt = 0

    for r in results:
        gt = r.get("gt", {})
        if not gt:
            per_doc.append({"image": r["image"], "category": "no_gt"})
            continue

        n_with_gt += 1
        clf = classify_fields(r["pred"], gt, soft=soft)

        if clf["fp"] == 0 and clf["fn"] == 0:
            category = "perfect"
        elif clf["fn"] > 0 and clf["fp"] == 0:
            category = "fn_only"
        elif clf["fp"] > 0 and clf["fn"] == 0:
            category = "fp_only"
        else:
            category = "mixed"

        category_counts[category] += 1
        per_doc.append({"image": r["image"], "category": category, **clf})

    return {
        "n_total": len(results),
        "n_with_gt": n_with_gt,
        "perfect": category_counts["perfect"],
        "fn_only": category_counts["fn_only"],
        "fp_only": category_counts["fp_only"],
        "mixed": category_counts["mixed"],
        "per_doc": per_doc,
    }


# ── Human-readable summary ────────────────────────────────────────────────────


def summarize(results: list[dict], soft: bool = False) -> None:
    """
    Print a human-readable metrics report to stdout.

    Suitable for both CLI use (called by predict.py) and Jupyter (call in a cell).

    Parameters
    ----------
    results : list of {"image", "pred", "gt"} dicts
    soft    : if True, use normalized matching (lowercase + strip)
    """
    mode = "soft (normalized)" if soft else "strict (exact match)"
    print(f"\n── Metrics [{mode}] ──────────────────────────────")

    fstats = field_stats(results, soft=soft)
    dstats = doc_stats(results, soft=soft)

    n_with_gt = dstats["n_with_gt"]
    if n_with_gt == 0:
        print("  No ground-truth labels found — skipping metrics.")
        return

    def _fmt(v) -> str:
        return f"{v:.1%}" if v is not None else "  N/A "

    # Per-field table
    print(f"\n  Per-field  (n_docs_with_gt={n_with_gt})\n")
    header = f"  {'Field':<28}  {'Prec':>6}  {'Rec':>6}  {'F1':>6}  {'TP':>4}  {'FP':>4}  {'FN':>4}  {'TN':>4}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    for field in sorted(fstats):
        s = fstats[field]
        print(
            f"  {field:<28}  {_fmt(s['precision']):>6}  {_fmt(s['recall']):>6}"
            f"  {_fmt(s['f1']):>6}  {s['tp']:>4}  {s['fp']:>4}  {s['fn']:>4}  {s['tn']:>4}"
        )

    # Document-level breakdown
    print(f"\n  Document-level  (n={n_with_gt})\n")
    cats = ["perfect", "fn_only", "fp_only", "mixed"]
    for cat in cats:
        n = dstats[cat]
        pct = n / n_with_gt
        bar = "█" * int(pct * 20)
        print(f"  {cat:<12}  {n:>4}  {pct:>6.1%}  {bar}")
    print()
