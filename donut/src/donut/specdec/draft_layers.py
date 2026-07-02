"""STUB — Direction 2: layer-truncated self-draft (docs/speculative-decoding.md).

A draft model built from the first 1-2 of the decoder's 4 MBart layers
proposes the free-text value tokens the skeleton cannot. Two routes into the
stock assisted-generation loop (verified against transformers 5.12.1):

Route A: generate(..., assistant_early_exit=n) — HF's EarlyExitCandidateGenerator
    runs the main model with num_hidden_layers temporarily lowered. UNVERIFIED
    for VisionEncoderDecoderModel: it resolves the config via base_model_prefix,
    and Donut's layer count lives at model.config.decoder. Probe first.

Route B (likely cleaner): wrap the first n decoder layers, weight-shared
    (embeddings + lm_head included), as a decoder-only causal LM and pass it as
    generate(assistant_model=...). Hits the decoder-only-assistant path in
    generation/candidate_generator.py (the "DistilWhisper case"), which reuses
    the main model's encoder_outputs — no second Swin encoder pass.

Kill signal before investing further: untrained-truncation acceptance rate
below ~0.3. Follow-up if killed: distill a 2-layer draft (training-side).
"""


def make_layer_truncated_assistant(model, n_layers: int = 2):
    """Build a weight-shared draft from the first n decoder layers (Route B)."""
    raise NotImplementedError("Direction 2 stub — see module docstring.")
