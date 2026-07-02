"""Static skeleton proposer for assisted generation.

The fine-tuned output is a fixed template (dataset.format_label):

    <s_donut> <s_f1> value </s_f1> ... <s_f8> value </s_f8> </s>

so the tag sequence is grammar-determined. SkeletonGrammar holds the token-id
view of that grammar; SkeletonCandidateGenerator walks it to draft the tokens
that are certain from position alone, and drafts nothing (k=0, a plain greedy
step) inside free-text values. Greedy verification makes the output token-for-
token identical to vanilla greedy regardless of what the proposer drafts.
"""

import torch
from transformers import DonutProcessor
from transformers.generation.candidate_generator import CandidateGenerator

from donut.constants import MISSING_TOKEN, TASK_TOKEN
from donut.dataset import _t2j_pairs


class SkeletonGrammar:
    """Token-id transitions of the fixed token2json template."""

    def __init__(
        self,
        task_id: int,
        open_ids: list[int],
        close_ids: list[int],
        missing_id: int,
        eos_id: int,
    ):
        self.task_id = task_id
        self.open_ids = open_ids
        self.close_ids = close_ids
        self.missing_id = missing_id
        self.eos_id = eos_id
        # close tag i -> open tag i+1, last close tag -> EOS
        self._after_close = dict(zip(close_ids[:-1], open_ids[1:]))
        self._after_close[close_ids[-1]] = eos_id
        self._field_of_open = {tok: i for i, tok in enumerate(open_ids)}
        self._close_set = set(close_ids)

    @classmethod
    def from_processor(cls, processor: DonutProcessor) -> "SkeletonGrammar":
        """Resolve the template's token ids from the (extended) tokenizer.

        Requires the fine-tuned vocabulary: dataset.register_field_tokens must
        have run (training checkpoints carry it; for the base model the caller
        registers it explicitly).
        """
        tokenizer = processor.tokenizer
        pairs = _t2j_pairs()
        names = [TASK_TOKEN, MISSING_TOKEN] + [t for pair in pairs for t in pair]
        ids = {name: tokenizer.convert_tokens_to_ids(name) for name in names}
        unknown = [n for n, i in ids.items() if i == tokenizer.unk_token_id]
        if unknown:
            raise ValueError(
                f"Tokenizer lacks the fine-tuned field vocabulary ({unknown}); "
                "load a fine-tuned checkpoint or register_field_tokens first."
            )
        return cls(
            task_id=ids[TASK_TOKEN],
            open_ids=[ids[o] for o, _ in pairs],
            close_ids=[ids[c] for _, c in pairs],
            missing_id=ids[MISSING_TOKEN],
            eos_id=tokenizer.eos_token_id,
        )

    def draft(
        self, prefix: list[int], max_draft: int, missing_chain: bool
    ) -> list[int]:
        """Grammar-determined continuation of `prefix`, up to max_draft tokens.

        Stops at any free-text frontier: after an open tag (unless
        missing_chain speculates <missing> there) and after any value token.
        Empty result means "no proposal" — the loop falls back to one plain
        greedy step.
        """
        chain: list[int] = []
        last = prefix[-1]
        while len(chain) < max_draft:
            if last == self.task_id:
                nxt = self.open_ids[0]
            elif last in self._after_close:
                nxt = self._after_close[last]
            elif last in self._field_of_open:
                if not missing_chain:
                    break
                nxt = self.missing_id
            elif last == self.missing_id:
                field = self._current_field(prefix, chain)
                if field is None:
                    break
                nxt = self.close_ids[field]
            else:  # EOS or a free-text value token
                break
            chain.append(nxt)
            last = nxt
        return chain

    def _current_field(self, prefix: list[int], chain: list[int]) -> int | None:
        """Field index of the nearest unclosed open tag, scanning backwards."""
        for tok in reversed(prefix + chain):
            if tok in self._field_of_open:
                return self._field_of_open[tok]
            if tok in self._close_set:
                return None
        return None


class SkeletonCandidateGenerator(CandidateGenerator):
    """CandidateGenerator that drafts from SkeletonGrammar.

    Stateless per step: the draft is re-derived from the authoritative
    input_ids on every call, so any rejection pattern self-corrects (the
    sequence is ≤ ~130 tokens; the CPU scan is negligible).
    """

    def __init__(
        self, grammar: SkeletonGrammar, max_draft: int = 8, missing_chain: bool = True
    ):
        self.grammar = grammar
        self.max_draft = max_draft
        self.missing_chain = missing_chain
        self.records: list[tuple[int, int]] = []  # (proposed k, accepted) per step
        self._last_k = 0

    # The base class annotates the second element as FloatTensor, but None is
    # the documented no-logits case (HF's own PromptLookup returns it too).
    def get_candidates(  # ty: ignore[invalid-method-override]
        self, input_ids: torch.LongTensor, **kwargs
    ) -> tuple[torch.LongTensor, None]:
        prefix = input_ids[0].tolist()
        chain = self.grammar.draft(prefix, self.max_draft, self.missing_chain)
        self._last_k = len(chain)
        if not chain:
            return input_ids, None
        draft = torch.tensor([chain], dtype=input_ids.dtype, device=input_ids.device)
        return torch.cat([input_ids, draft], dim=-1), None  # ty: ignore[invalid-return-type]

    def update_candidate_strategy(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor, num_matches: int
    ) -> None:
        self.records.append((self._last_k, int(num_matches)))

    def stats(self) -> dict:
        """Mechanism metrics for the generate() call this proposer served."""
        steps = len(self.records)
        proposed = sum(k for k, _ in self.records)
        accepted = sum(m for _, m in self.records)
        new_tokens = sum(m + 1 for _, m in self.records)  # +1 bonus token per step
        return {
            "steps": steps,
            "proposed": proposed,
            "accepted": accepted,
            "acceptance_rate": accepted / proposed if proposed else None,
            "mean_tokens_per_step": new_tokens / steps if steps else None,
            "new_tokens": new_tokens,
        }
