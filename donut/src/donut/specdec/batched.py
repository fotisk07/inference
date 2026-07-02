"""STUB — Direction 4: batched verification loop (docs/speculative-decoding.md).

The stock assisted loop raises on batch_size > 1 (transformers 5.12.1,
generation/utils.py: "assisted generate is only supported for batch_size = 1").
Batching verification means owning the loop: per-row draft lengths, ragged
acceptance (rows accept different prefix lengths → per-row KV-cache crop or
recompute), right-padding and position-id bookkeeping.

Build only if bench_skeleton's frontier measurement shows bs=1 assisted
beating batched vanilla up to bs≈2 — otherwise batched vanilla already wins
on throughput and this loop has no headroom to claim.
"""


def generate_assisted_batched(model, pixel_values, candidate_generator, **kwargs):
    """Custom batched draft-and-verify decode loop."""
    raise NotImplementedError("Direction 4 stub — see module docstring.")
