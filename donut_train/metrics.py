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

"gt" may be empty ({}) if you're running inference without labels — metric functions
that require ground truth will return None or skip those documents gracefully.

── Terminology ──────────────────────────────────────────────────────────────────
For each (document, field) pair we assign one of four labels:

  TP (True Positive)           field in GT  ∧  field in pred  ∧  values match
  FP_WRONG (False Positive)    field in GT  ∧  field in pred  ∧  values differ
  FP_HALL  (Hallucinated FP)   field NOT in GT  ∧  field in pred
  FN (False Negative)          field in GT  ∧  field NOT in pred

  TN (True Negative): field NOT in GT and model also skipped it.
  TN is intentionally not tracked — there are many possible "missing" fields and
  they carry no signal about extraction quality.

From these four primitives everything else is derived:

  precision  = TP / (TP + FP_WRONG + FP_HALL)
               "of what the model produced, how much was correct?"

  recall     = TP / (TP + FN)
               "of what existed in the document, how much did the model find?"

  f1         = 2 * precision * recall / (precision + recall)

── Matching modes ───────────────────────────────────────────────────────────────
All functions accept a `soft` keyword argument (default False).

  soft=False (strict):  exact string comparison after no normalization
  soft=True  (lenient): compare normalize(pred) == normalize(gt)
                        normalize() lowercases and strips leading/trailing whitespace

To change what "lenient" means, edit the `normalize()` function — it is the
single point of control for soft matching throughout this file.
"""

from __future__ import annotations

from collections import defaultdict


# ── Normalization ─────────────────────────────────────────────────────────────


def normalize(value: str) -> str:
    """
    Canonical form used for soft matching (soft=True).

    Current rules: strip leading/trailing whitespace, then lowercase.

    Add more rules here if needed, e.g.:
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
    Classify every field for a single document into TP / FP_WRONG / FP_HALL / FN.

    Parameters
    ----------
    pred : model predictions — {field_name: predicted_value}
    gt   : ground truth      — {field_name: true_value}
    soft : if True use normalize() for value comparison

    Returns
    -------
    dict with keys:
        tp            : int  — count of true positives
        fp_wrong      : int  — count of predicted-but-wrong fields
        fp_hall       : int  — count of hallucinated (not-in-GT) fields
        fn            : int  — count of missed GT fields

        tp_fields     : list[str] — field names that were TPs
        fp_wrong_fields  : list[str]
        fp_hall_fields   : list[str]
        fn_fields        : list[str]
    """
    tp_fields: list[str] = []
    fp_wrong_fields: list[str] = []
    fp_hall_fields: list[str] = []
    fn_fields: list[str] = []

    # Walk model predictions
    for field, pred_val in pred.items():
        if field in gt:
            if _match(pred_val, gt[field], soft):
                tp_fields.append(field)
            else:
                fp_wrong_fields.append(field)
        else:
            fp_hall_fields.append(field)

    # Walk GT fields the model missed entirely
    for field in gt:
        if field not in pred:
            fn_fields.append(field)

    return {
        "tp": len(tp_fields),
        "fp_wrong": len(fp_wrong_fields),
        "fp_hall": len(fp_hall_fields),
        "fn": len(fn_fields),
        "tp_fields": tp_fields,
        "fp_wrong_fields": fp_wrong_fields,
        "fp_hall_fields": fp_hall_fields,
        "fn_fields": fn_fields,
    }


# ── Per-field aggregate stats across all documents ────────────────────────────


def field_stats(results: list[dict], soft: bool = False) -> dict[str, dict]:
    """
    Compute precision / recall / F1 per field across all documents that have GT.

    Only documents where result["gt"] is non-empty are included.

    Returns
    -------
    dict keyed by field_name, each value a dict with:
        precision   float  TP / (TP + FP_WRONG + FP_HALL)
        recall      float  TP / (TP + FN)
        f1          float  harmonic mean
        tp          int
        fp_wrong    int
        fp_hall     int
        fn          int

    Precision/recall are None when the denominator is 0
    (e.g. a field never appears in GT → recall denominator is 0).
    """
    # Accumulators: per field, collect raw counts across docs
    counts: dict[str, dict[str, int]] = defaultdict(
        lambda: {"tp": 0, "fp_wrong": 0, "fp_hall": 0, "fn": 0}
    )

    for r in results:
        if not r.get("gt"):
            continue
        clf = classify_fields(r["pred"], r["gt"], soft=soft)
        # attribute tp/fp/fn to each specific field
        for field in clf["tp_fields"]:
            counts[field]["tp"] += 1
        for field in clf["fp_wrong_fields"]:
            counts[field]["fp_wrong"] += 1
        for field in clf["fp_hall_fields"]:
            counts[field]["fp_hall"] += 1
        for field in clf["fn_fields"]:
            counts[field]["fn"] += 1

    stats: dict[str, dict] = {}
    for field, c in counts.items():
        prec_denom = c["tp"] + c["fp_wrong"] + c["fp_hall"]
        rec_denom = c["tp"] + c["fn"]

        precision = c["tp"] / prec_denom if prec_denom > 0 else None
        recall = c["tp"] / rec_denom if rec_denom > 0 else None

        if precision is not None and recall is not None and (precision + recall) > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = None

        stats[field] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            **c,
        }
    return stats


# ── Document-level summary ────────────────────────────────────────────────────


def doc_stats(results: list[dict], soft: bool = False) -> dict:
    """
    Categorize each document and return aggregate counts.

    Categories (mutually exclusive, checked in this order):
        perfect          all GT fields found with correct values (FP_HALL allowed but
                         not counted as imperfect — model can produce extras and still
                         get "all GT right". Change this if over-extraction should
                         count against perfectness.)
        all_found_wrong  model found every GT field (FN == 0) but ≥1 value is wrong
        under_extracted  model missed ≥1 GT field (FN > 0), regardless of hallucinations
        over_extracted   model hallucinated ≥1 field (FP_HALL > 0) but FN == 0 and TP covers all GT
        mixed            both FN > 0 and FP_HALL > 0

    NOTE: "perfect" here means all GT fields found with correct values.
    A document with FP_HALL > 0 but TP == len(gt) is currently marked "perfect"
    because the GT is fully satisfied. If you want to penalise hallucinations,
    change the `perfect` condition below to also require fp_hall == 0.

    Returns
    -------
    dict with:
        n_total          int
        n_with_gt        int   — documents that had ground truth
        perfect          int
        all_found_wrong  int
        under_extracted  int
        over_extracted   int
        mixed            int
        per_doc          list[dict]  — raw clf + category per document (for Jupyter slicing)
    """
    per_doc = []
    category_counts: dict[str, int] = defaultdict(int)
    n_with_gt = 0

    for r in results:
        gt = r.get("gt", {})
        if not gt:
            per_doc.append({"image": r["image"], "category": "no_gt", **{}})
            continue

        n_with_gt += 1
        clf = classify_fields(r["pred"], gt, soft=soft)

        # Determine category
        if clf["fn"] == 0 and clf["fp_wrong"] == 0:
            # All GT fields found and correct (hallucinations allowed by default)
            category = "perfect"
        elif clf["fn"] == 0 and clf["fp_wrong"] > 0:
            # Found every GT field but at least one value is wrong
            category = "all_found_wrong"
        elif clf["fn"] > 0 and clf["fp_hall"] > 0:
            # Both missed fields and hallucinated ones
            category = "mixed"
        elif clf["fn"] > 0:
            # Missed GT fields, no hallucinations
            category = "under_extracted"
        else:
            # FP_HALL > 0, FN == 0, FP_WRONG == 0 — all GT correct but extra hallucinations
            category = "over_extracted"

        category_counts[category] += 1
        per_doc.append({"image": r["image"], "category": category, **clf})

    return {
        "n_total": len(results),
        "n_with_gt": n_with_gt,
        "perfect": category_counts["perfect"],
        "all_found_wrong": category_counts["all_found_wrong"],
        "under_extracted": category_counts["under_extracted"],
        "over_extracted": category_counts["over_extracted"],
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

    # Per-field table
    print(f"\n  Per-field  (n_docs_with_gt={n_with_gt})\n")
    header = f"  {'Field':<28}  {'Prec':>6}  {'Rec':>6}  {'F1':>6}  {'TP':>4}  {'FP_wrong':>8}  {'FP_hall':>7}  {'FN':>4}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    def _fmt(v) -> str:
        return f"{v:.1%}" if v is not None else "  N/A "

    for field in sorted(fstats):
        s = fstats[field]
        print(
            f"  {field:<28}  {_fmt(s['precision']):>6}  {_fmt(s['recall']):>6}"
            f"  {_fmt(s['f1']):>6}  {s['tp']:>4}  {s['fp_wrong']:>8}  {s['fp_hall']:>7}  {s['fn']:>4}"
        )

    # Document-level breakdown
    print(f"\n  Document-level  (n={n_with_gt})\n")
    cats = ["perfect", "all_found_wrong", "under_extracted", "over_extracted", "mixed"]
    for cat in cats:
        n = dstats[cat]
        pct = n / n_with_gt
        bar = "█" * int(pct * 20)
        print(f"  {cat:<20}  {n:>4}  {pct:>6.1%}  {bar}")
    print()
