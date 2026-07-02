# Speculative decoding: research plan and directions

Attention-kernel work ([attention-backends.md](attention-backends.md),
[encoder-optimizations.md](encoder-optimizations.md)) optimized the cost of
*each* decode step. This track attacks the *number* of steps: greedy
`generate()` runs one full decoder forward per token, ≤128 times per
document, and each of those forwards is memory-bound at `q_len=1` — the GPU
has compute to spare. Speculative decoding spends that spare compute to
verify several cheaply-proposed tokens in a single forward.

## Why Donut specifically is a good target

The fine-tuned output is a fixed template (see `format_label` in
`src/donut/dataset.py`): every document decodes to

```
<s_donut> <s_f1> value₁ </s_f1> <s_f2> value₂ </s_f2> … <s_f8> value₈ </s_f8> </s>
```

with all 8 fields always present, in `FIELD_TOKENS` order, and absent
fields holding the literal `<missing>` token. Every tag is a single
registered special token. So of the ≤128 generated tokens:

- the 8 open tags, 8 close tags, and final EOS (**17 tokens**) are
  *grammar-determined* — knowable with certainty from position alone;
- each `<missing>` field contributes its value token too — a fully missing
  document is ~26 deterministic tokens out of ~26;
- only the free-text values are genuinely uncertain.

A proposer that knows the grammar can hand those deterministic tokens to
the verifier in blocks, collapsing many decode steps into few.

## The substrate: HF assisted generation (transformers 5.12.1, pinned)

We do not write a decode loop. `generate()` already contains one —
"assisted generation" — with exactly the verification semantics we need.
Verified against the installed transformers **5.12.1** (uv.lock):

- `generate()` enters `_assisted_decoding` when
  `generation_config.prompt_lookup_num_tokens is not None`
  (`generation/configuration_utils.py`, `get_generation_mode`).
- Inside it, the proposer is built by `self._get_candidate_generator(...)`
  (`generation/utils.py:958`) — called **on the instance, keyword-args
  only**. An instance-bound override (`model.__dict__`) is honored, so
  `specdec.apply_specdec_skeleton` injects our proposer there without
  touching the class. The `prompt_lookup_num_tokens = 1` trigger itself is
  inert — the override ignores it.
- The proposer implements `CandidateGenerator`
  (`generation/candidate_generator.py`): `get_candidates(input_ids) ->
  (input_ids ++ k draft tokens, None)` with k ≥ 0, and
  `update_candidate_strategy(input_ids, scores, num_matches)` called after
  every verify step — our acceptance-metrics hook.
- **Greedy verification is exact**: the main model runs one forward over
  the k draft tokens, takes `argmax` over the k+1 resulting logits, and
  accepts the longest matching prefix plus one bonus token. The output is
  provably identical to vanilla greedy decoding, token for token. k = 0
  degrades to a plain greedy step.
- Hard constraints of the stock loop: **batch_size = 1** (explicit raise),
  `use_cache=True`, dynamic (not static) cache.

Risk to keep in mind: `_get_candidate_generator` is private API. The
integration is exact for 5.12.1; on any transformers bump, re-verify the
call site and run `check_specdec` + the exact-match gate before trusting
numbers. If the override ever silently fails to install, the stock
prompt-lookup proposer runs instead — output stays *correct* (the verifier
guarantees that) but mechanism metrics become meaningless; the bench
detects this via `specdec.last_stats` being absent.

## Metrics and protocol

Mechanism (per document, from `specdec.last_stats`):

| symbol | meaning |
|---|---|
| α | acceptance rate = accepted draft tokens / proposed draft tokens |
| τ | tokens per verify step = (accepted + bonus) / verify steps; vanilla ≡ 1.0 |
| steps | verify steps taken vs vanilla's token count |

End-to-end (per README.md "Metrics"): ms/doc and docs/s, CUDA-synced.

Protocol, in order:
1. **Exact-match gate**: assisted output ids == vanilla greedy output ids
   on every document, must be 100% before any timing claim.
2. Mechanism metrics on a fine-tuned checkpoint (untrained donut-base
   produces junk text, so α there says nothing — the base-model
   `--register-vocab` mode of the bench only validates machinery).
3. Timing: same checkpoint, same backend (`sdpa`), same images, declared
   warmup, bs=1 assisted vs bs=1 vanilla vs the batched-vanilla frontier.

The batched frontier matters because assisted generation is bs=1-only:
if vanilla at bs=4 already beats bs=1 assisted on docs/s, speculative
decoding only pays off in latency-sensitive (single-document) serving.
`scripts/specdec/bench_skeleton.py` measures all of the above in one run.

## Direction 1 — static skeleton proposer (implemented)

`src/donut/specdec/skeleton.py`. A grammar state machine over token ids:
after a close tag propose the next open tag; after an open tag propose
`<missing>`; after `<missing>` propose the close tag; after the last close
tag propose EOS; inside a free-text value propose nothing (k=0). The
`<missing>` chaining produces multi-token ladders
(`</s_fᵢ> <s_fᵢ₊₁> <missing> </s_fᵢ₊₁> <s_fᵢ₊₂> …`) — when the field
actually has a value, greedy verification truncates the ladder at the
value's first token at no extra cost.

- Hypothesis: α ≈ 1.0 on structural tokens (they are certain if the model
  learned the template at all), τ noticeably > 1, best case (all fields
  missing) ~9 verify steps instead of ~26.
- Continue signal: measurable ms/doc win at bs=1 on a real checkpoint.
- Kill signal: assisted ms/doc ≥ vanilla — plausible, because at ≤128
  tokens the assisted loop's extra Python work per step can outweigh the
  saved forwards. That result is worth documenting either way.

## Direction 2 — layer-truncated self-draft (stub)

`src/donut/specdec/draft_layers.py`. The decoder is a 4-layer MBart; a
draft built from its first 1–2 layers proposes free-text value tokens the
skeleton cannot. Two routes, recorded in the stub's docstring:

- Route A: `generate(..., assistant_early_exit=n)` — HF's
  `EarlyExitCandidateGenerator`. Unverified for VisionEncoderDecoder
  (`base_model_prefix` / `num_hidden_layers` resolution; decoder config
  lives at `model.config.decoder`). Probe before building on it.
- Route B (likely cleaner): wrap the first n decoder layers,
  weight-shared, as a decoder-only causal LM and pass it as
  `assistant_model` — hits the "DistilWhisper" path in
  `candidate_generator.py`, which reuses the main model's
  `encoder_outputs` (no second Swin encoder pass).

Kill signal: untrained-truncation α < ~0.3 (draft disagrees with full
model too often to pay for itself). Follow-up if killed: distill a 2-layer
draft against the fine-tuned model (training-side work, separate effort).

## Direction 3 — skeleton + draft composed

One `CandidateGenerator` that uses the grammar for structure tokens and
the truncated draft inside values. Strictly dominates both parents if
Direction 2 survives its probe. Build only after 1 and 2 have numbers.

## Direction 4 — batched verification loop (stub)

`src/donut/specdec/batched.py`. The stock loop hard-blocks bs>1; a custom
loop must handle ragged per-row acceptance (per-row cache crop or
recompute, right-padding, position bookkeeping). Substantial work — build
only if the frontier measurement shows bs=1 assisted beating vanilla up to
bs≈2, i.e. there is real headroom to extend.

## Direction 5 — smarter value proposers (ideas only)

- Per-field format ladders: `data_emissao` (dates), `cpf_cnpj_*`
  (14/11-digit patterns), `cep_prestador` (8 digits) are format-regular;
  a proposer can draft separators/digit-group shapes.
- Prompt-lookup hybrid: n-gram reuse *within* the generated sequence is
  near-useless here (tags don't repeat), but values repeated across fields
  (prestador/tomador documents) could be caught.
- Constrained decoding (masking invalid tokens) is *complementary*, not a
  speedup — it changes outputs; specdec by construction does not.
