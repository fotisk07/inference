"""Speculative decoding for the Donut model (research track).

Same in-place contract as donut.accel: apply_x / revert_x / check_x. See
docs/speculative-decoding.md for the research plan and the verified
transformers 5.12.1 integration surface.

The hook: generate() enters assisted decoding when
generation_config.prompt_lookup_num_tokens is set, and builds its proposer
via self._get_candidate_generator(...) — an instance-level keyword-args call,
so an instance-bound override is honored. apply_specdec_skeleton installs
both the (inert) trigger and the override. If the override were ever missing,
the stock prompt-lookup proposer would run instead: output stays correct
(greedy verification guarantees it) but stats vanish — check_specdec and
last_stats() is None catch that.

Stock-loop constraints: batch_size=1, use_cache=True, dynamic cache.
"""

import types

from donut.specdec.skeleton import SkeletonCandidateGenerator, SkeletonGrammar


def apply_specdec_skeleton(
    model, processor, *, max_draft: int = 8, missing_chain: bool = True
) -> None:
    """Route model.generate() through assisted decoding with the skeleton proposer.

    Idempotent. Builds the grammar from the processor once; each generate()
    call gets a fresh SkeletonCandidateGenerator, kept on the model for
    last_stats(). Call revert_specdec before save_pretrained — the trigger
    lives on generation_config and would be persisted with a checkpoint.
    """
    if getattr(model, "_specdec_active", False):
        return
    grammar = SkeletonGrammar.from_processor(processor)

    def _get_candidate_generator(self, **kwargs) -> SkeletonCandidateGenerator:
        generator = SkeletonCandidateGenerator(
            grammar, max_draft=max_draft, missing_chain=missing_chain
        )
        self._specdec_last = generator
        return generator

    model._get_candidate_generator = types.MethodType(_get_candidate_generator, model)
    # Inert mode trigger: routes generate() into _assisted_decoding; the value
    # is never read because the override above ignores it.
    model.generation_config.prompt_lookup_num_tokens = 1
    model._specdec_active = True


def revert_specdec(model) -> None:
    """Restore vanilla generate(). No-op if specdec is not applied."""
    model.__dict__.pop("_get_candidate_generator", None)
    model.generation_config.prompt_lookup_num_tokens = None
    model._specdec_active = False
    model._specdec_last = None


def check_specdec(model) -> None:
    """Assert the skeleton proposer is fully installed. Raises AssertionError."""
    assert "_get_candidate_generator" in model.__dict__, (
        "specdec: instance override of _get_candidate_generator is not installed"
    )
    assert model.generation_config.prompt_lookup_num_tokens is not None, (
        "specdec: assisted-decoding trigger (prompt_lookup_num_tokens) is not set"
    )


def last_stats(model) -> dict | None:
    """Mechanism metrics of the most recent generate(), or None if none ran."""
    generator = getattr(model, "_specdec_last", None)
    return generator.stats() if generator is not None else None


__all__ = [
    "SkeletonCandidateGenerator",
    "SkeletonGrammar",
    "apply_specdec_skeleton",
    "check_specdec",
    "last_stats",
    "revert_specdec",
]
