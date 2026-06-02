"""
Rigorous tests for the SDPA patch on DonutSwinSelfAttention.

Correctness is verified against an independent reference implementation of
the exact formula the original code computes:

    output = softmax( QKᵀ/√d  +  rel_pos_bias  [+  shift_mask] )  ×  V

Test groups
-----------
1. Reference sanity  — verify the reference impl matches eager (float32, CPU)
2. SDPA correctness  — SDPA matches reference and eager, parametrised over all
                       four Donut stages, with and without the cyclic-shift mask
3. dtype              — bfloat16 outputs are within expected precision
4. Edge cases         — idempotent patch, output_attentions fallback,
                       NaN/Inf absence with extreme mask values
5. Decoder config     — activate_decoder_sdpa / activate_decoder_fa2

Run
---
    uv run pytest tests/test_sdpa_patch.py -v
    uv run pytest tests/test_sdpa_patch.py -v -x          # stop on first failure
    uv run pytest tests/test_sdpa_patch.py -v -k float32  # only float32 tests
"""

import math
import types

import pytest
import torch
import torch.nn.functional as F
from transformers.models.donut.modeling_donut_swin import DonutSwinSelfAttention

from inference.accel.fa2 import activate_decoder_fa2
from inference.accel.sdpa import _sdpa_self_forward, activate_decoder_sdpa, patch_swin_sdpa


# ============================================================================
# Constants — Donut-base Swin architecture
# ============================================================================

WINDOW_SIZE = 10           # Donut uses 10×10 windows
SEQ_LEN = WINDOW_SIZE ** 2  # 100 tokens per window

# All four Donut-base Swin stages: (embed_dim, num_heads)
ALL_STAGES = pytest.mark.parametrize(
    ("dim", "num_heads"),
    [(128, 4), (256, 8), (512, 16), (1024, 32)],
    ids=["stage0_d128_h4", "stage1_d256_h8", "stage2_d512_h16", "stage3_d1024_h32"],
)


# ============================================================================
# Helpers
# ============================================================================

def _cfg():
    from types import SimpleNamespace
    return SimpleNamespace(qkv_bias=True, attention_probs_dropout_prob=0.0)


def _make_self_attn(dim: int, num_heads: int, dtype=torch.float32) -> DonutSwinSelfAttention:
    """Standalone DonutSwinSelfAttention with random weights, eval mode."""
    return DonutSwinSelfAttention(
        _cfg(), dim=dim, num_heads=num_heads, window_size=WINDOW_SIZE
    ).to(dtype).eval()


def _rand_hidden(batch_windows: int, embed_dim: int, dtype=torch.float32, seed=0):
    """(batch_windows, SEQ_LEN, embed_dim) input tensor."""
    torch.manual_seed(seed)
    return torch.randn(batch_windows, SEQ_LEN, embed_dim, dtype=dtype)


def _shift_mask(nw: int, dtype=torch.float32) -> torch.Tensor:
    """
    Build the cyclic-shift attention mask that patches.py generates.

    Shape: (nw, SEQ_LEN, SEQ_LEN). Values: 0.0 (attend) or -100.0 (block).
    This is the exact same algorithm as patches.py, isolated here so tests
    don't depend on the model being loaded.
    """
    ws, ss = WINDOW_SIZE, WINDOW_SIZE // 2
    h = w = ws * 4  # feature map large enough to produce ≥ nw windows
    img = torch.zeros(1, h, w, 1)
    cnt = 0
    for hs in (slice(0, -ws), slice(-ws, -ss), slice(-ss, None)):
        for ws2 in (slice(0, -ws), slice(-ws, -ss), slice(-ss, None)):
            img[:, hs, ws2, :] = cnt
            cnt += 1
    mw = img.view(1, h // ws, ws, w // ws, ws, 1).permute(0, 1, 3, 2, 4, 5).contiguous()
    mw = mw.view(-1, SEQ_LEN)
    mask = mw.unsqueeze(1) - mw.unsqueeze(2)
    mask = mask.masked_fill(mask != 0, -100.0).masked_fill(mask == 0, 0.0)
    n = mask.shape[0]
    reps = (nw + n - 1) // n
    return mask.repeat(reps, 1, 1)[:nw].to(dtype)


def _reference_forward(
    attn: DonutSwinSelfAttention,
    hidden_states: torch.Tensor,
    attention_mask=None,
) -> torch.Tensor:
    """
    Independent reference: the formula written out explicitly, no helper layers.

    This is NOT a copy of either the eager or SDPA implementation.
    It computes the formula directly:

        scores  = QKᵀ / √head_dim  +  rel_pos_bias  [+  shift_mask]
        output  = softmax(scores) @ V
        reshape → (batch_size, seq_len, embed_dim)

    Any divergence between this and the patched forward means the patch
    computes a different formula.
    """
    B, S, _ = hidden_states.shape
    d = attn.attention_head_size       # head_dim
    H = attn.num_attention_heads
    ws = attn.window_size[0]           # 10

    shape = (B, S, H, d)
    Q = attn.query(hidden_states).view(shape).transpose(1, 2)   # (B, H, S, d)
    K = attn.key(hidden_states).view(shape).transpose(1, 2)
    V = attn.value(hidden_states).view(shape).transpose(1, 2)

    # Scaled dot product
    scores = (Q @ K.transpose(-2, -1)) / math.sqrt(d)           # (B, H, S, S)

    # Additive relative position bias (learned, per-head)
    rel_bias = (
        attn.relative_position_bias_table[attn.relative_position_index.view(-1)]
        .view(ws * ws, ws * ws, H)
        .permute(2, 0, 1)
        .contiguous()
    )  # (H, S, S)
    scores = scores + rel_bias.unsqueeze(0)                      # broadcast over B

    # Additive cyclic-shift mask (0 or -100)
    if attention_mask is not None:
        nw = attention_mask.shape[0]
        Bx = B // nw
        scores = scores.view(Bx, nw, H, S, S)
        scores = scores + attention_mask.unsqueeze(1).unsqueeze(0)  # (1, nw, 1, S, S)
        scores = scores.view(B, H, S, S)

    probs = F.softmax(scores, dim=-1)
    out = (probs @ V).permute(0, 2, 1, 3).contiguous()
    return out.view(B, S, attn.all_head_size)


def _patch_module(attn: DonutSwinSelfAttention) -> None:
    """Apply the SDPA patch to a standalone self-attention module (not via model walker)."""
    if not hasattr(attn, "_sdpa_patched"):
        attn._original_forward = attn.forward
        attn.forward = types.MethodType(_sdpa_self_forward, attn)
        attn._sdpa_patched = True


def _eager_output(attn, hidden, mask=None):
    with torch.no_grad():
        return attn(hidden, attention_mask=mask)[0]


def _sdpa_output(attn, hidden, mask=None):
    _patch_module(attn)
    with torch.no_grad():
        return attn(hidden, attention_mask=mask)[0]


# ============================================================================
# Group 1 — Reference sanity checks
# ============================================================================
# These tests ensure the reference implementation matches eager BEFORE we test
# the SDPA patch.  If these fail, the reference is wrong — not the patch.


class TestReferenceSanity:
    """Verify the independent reference matches the original eager forward."""

    def test_reference_matches_eager_no_mask(self):
        """
        Reference formula must match the original eager forward when there is
        no cyclic-shift mask (non-shifted Swin blocks).
        Failure → reference implementation is incorrect.
        """
        attn = _make_self_attn(dim=128, num_heads=4)
        hidden = _rand_hidden(4, 128)

        expected = _eager_output(attn, hidden)
        actual = _reference_forward(attn, hidden)

        torch.testing.assert_close(
            actual, expected, atol=1e-5, rtol=1e-5,
            msg="Reference diverges from eager (no mask): formula is wrong",
        )

    def test_reference_matches_eager_with_mask(self):
        """
        Reference must match eager when the cyclic-shift mask is present
        (shifted Swin blocks).
        Failure → reference does not model the mask application correctly.
        """
        attn = _make_self_attn(dim=128, num_heads=4)
        nw = 8
        hidden = _rand_hidden(nw, 128)
        mask = _shift_mask(nw)

        expected = _eager_output(attn, hidden, mask)
        actual = _reference_forward(attn, hidden, mask)

        torch.testing.assert_close(
            actual, expected, atol=1e-5, rtol=1e-5,
            msg="Reference diverges from eager (with mask): formula is wrong",
        )

    @ALL_STAGES
    def test_reference_matches_eager_all_stages(self, dim, num_heads):
        """
        Reference must match eager for every Donut stage configuration.
        Different dims and head counts exercise different head_dim/scale combinations.
        """
        attn = _make_self_attn(dim=dim, num_heads=num_heads)
        hidden = _rand_hidden(4, dim)

        expected = _eager_output(attn, hidden)
        actual = _reference_forward(attn, hidden)

        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)


# ============================================================================
# Group 2 — SDPA correctness (float32, CPU)
# ============================================================================
# Float32 on CPU is deterministic and has enough precision to catch any
# mis-implementation.  These are the primary correctness tests.


class TestSdpaCorrectnessFloat32:
    """SDPA patch computes the exact same formula as eager, in float32."""

    def test_sdpa_matches_reference_no_mask(self):
        """
        SDPA output must equal the explicit reference formula (no mask, float32).
        This is the gold-standard test: it checks the formula, not just
        agreement between two implementations that could share the same bug.
        """
        attn = _make_self_attn(dim=128, num_heads=4)
        hidden = _rand_hidden(4, 128)

        expected = _reference_forward(attn, hidden)
        actual = _sdpa_output(attn, hidden)

        torch.testing.assert_close(
            actual, expected, atol=1e-5, rtol=1e-5,
            msg="SDPA diverges from the reference formula (no mask, float32)",
        )

    def test_sdpa_matches_reference_with_mask(self):
        """
        SDPA output must equal the explicit reference formula with the cyclic-
        shift mask (float32).  Verifies that the mask is broadcast correctly
        and that -100 values suppress the right attention pairs.
        """
        attn = _make_self_attn(dim=128, num_heads=4)
        nw = 8
        hidden = _rand_hidden(nw, 128)
        mask = _shift_mask(nw)

        expected = _reference_forward(attn, hidden, mask)
        actual = _sdpa_output(attn, hidden, mask)

        torch.testing.assert_close(
            actual, expected, atol=1e-5, rtol=1e-5,
            msg="SDPA diverges from the reference formula (with mask, float32)",
        )

    def test_sdpa_matches_eager_no_mask(self):
        """
        SDPA must produce the same output as the original eager forward (no mask).
        Complements the reference test by ensuring no regression vs. the real code.
        """
        attn = _make_self_attn(dim=256, num_heads=8)
        hidden = _rand_hidden(4, 256)

        expected = _eager_output(attn, hidden)
        actual = _sdpa_output(attn, hidden)

        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)

    def test_sdpa_matches_eager_with_mask(self):
        """
        SDPA must produce the same output as eager with the cyclic-shift mask.
        Failure means the mask broadcasting logic is wrong in the SDPA patch.
        """
        attn = _make_self_attn(dim=256, num_heads=8)
        nw = 8
        hidden = _rand_hidden(nw, 256)
        mask = _shift_mask(nw)

        expected = _eager_output(attn, hidden, mask)
        actual = _sdpa_output(attn, hidden, mask)

        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)

    @ALL_STAGES
    def test_sdpa_matches_eager_all_stages_no_mask(self, dim, num_heads):
        """SDPA matches eager across all four Donut Swin stages (no mask)."""
        attn = _make_self_attn(dim=dim, num_heads=num_heads)
        hidden = _rand_hidden(4, dim)

        expected = _eager_output(attn, hidden)
        actual = _sdpa_output(attn, hidden)

        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)

    @ALL_STAGES
    def test_sdpa_matches_eager_all_stages_with_mask(self, dim, num_heads):
        """SDPA matches eager across all four Donut Swin stages (with mask)."""
        attn = _make_self_attn(dim=dim, num_heads=num_heads)
        nw = 8
        hidden = _rand_hidden(nw, dim)
        mask = _shift_mask(nw)

        expected = _eager_output(attn, hidden, mask)
        actual = _sdpa_output(attn, hidden, mask)

        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)

    def test_batch_size_one(self):
        """SDPA with batch_size=1 (single window) — smallest valid input."""
        attn = _make_self_attn(dim=128, num_heads=4)
        hidden = _rand_hidden(1, 128)

        expected = _eager_output(attn, hidden)
        actual = _sdpa_output(attn, hidden)

        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)

    def test_larger_batch(self):
        """SDPA with batch_size=32 — verifies broadcast over many windows."""
        attn = _make_self_attn(dim=128, num_heads=4)
        nw = 16
        hidden = _rand_hidden(32, 128)
        mask = _shift_mask(nw)

        expected = _eager_output(attn, hidden, mask)
        actual = _sdpa_output(attn, hidden, mask)

        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)


# ============================================================================
# Group 3 — dtype (bfloat16)
# ============================================================================
# bfloat16 has ~0.8% relative precision so we allow more tolerance.
# The mean absolute error should still be small; NaN/Inf must never appear.


class TestSdpaCorrectnessB16:
    """SDPA outputs are numerically close to eager in bfloat16."""

    def test_sdpa_close_to_eager_bf16_no_mask(self):
        """
        In bfloat16 (production dtype) SDPA mean absolute error must be small.
        A large mean error means the patch is computing something fundamentally
        different.  A large MAX error alone may be a single bfloat16 outlier
        caused by SDPA's internal float32 accumulation — tolerated if mean is OK.
        """
        attn = _make_self_attn(dim=128, num_heads=4, dtype=torch.bfloat16)
        hidden = _rand_hidden(8, 128, dtype=torch.bfloat16)

        expected = _eager_output(attn, hidden)
        actual = _sdpa_output(attn, hidden)

        assert not torch.isnan(actual).any(), "NaN in SDPA output (bfloat16, no mask)"
        assert not torch.isinf(actual).any(), "Inf in SDPA output (bfloat16, no mask)"

        mae = (actual - expected).abs().mean().item()
        assert mae < 0.05, f"bfloat16 mean abs error {mae:.6f} exceeds threshold 0.05"

    def test_sdpa_close_to_eager_bf16_with_mask(self):
        """Same as above but with the cyclic-shift mask applied."""
        attn = _make_self_attn(dim=128, num_heads=4, dtype=torch.bfloat16)
        nw = 8
        hidden = _rand_hidden(nw, 128, dtype=torch.bfloat16)
        mask = _shift_mask(nw, dtype=torch.bfloat16)

        expected = _eager_output(attn, hidden, mask)
        actual = _sdpa_output(attn, hidden, mask)

        assert not torch.isnan(actual).any(), "NaN in SDPA output (bfloat16, with mask)"
        assert not torch.isinf(actual).any(), "Inf in SDPA output (bfloat16, with mask)"

        mae = (actual - expected).abs().mean().item()
        assert mae < 0.05, f"bfloat16 mean abs error {mae:.6f} exceeds threshold 0.05"

    @ALL_STAGES
    def test_sdpa_no_nan_inf_all_stages_bf16(self, dim, num_heads):
        """
        No NaN or Inf in SDPA output for any stage in bfloat16.
        Failure at a specific stage indicates a numerically degenerate case
        (e.g., overflow in the large-dim stages where head_dim=32 for all stages).
        """
        attn = _make_self_attn(dim=dim, num_heads=num_heads, dtype=torch.bfloat16)
        nw = 8
        hidden = _rand_hidden(nw, dim, dtype=torch.bfloat16)
        mask = _shift_mask(nw, dtype=torch.bfloat16)

        actual = _sdpa_output(attn, hidden, mask)

        assert not torch.isnan(actual).any(), f"NaN in SDPA output (stage dim={dim})"
        assert not torch.isinf(actual).any(), f"Inf in SDPA output (stage dim={dim})"


# ============================================================================
# Group 4 — Edge cases and correctness properties
# ============================================================================


class TestSdpaEdgeCases:
    """Edge cases: patch idempotency, fallback, mask semantics."""

    def test_patch_is_idempotent(self):
        """
        Applying the patch twice must produce the same result as applying it once.
        Failure means _sdpa_self_forward wraps itself, compounding the error.
        """
        attn = _make_self_attn(dim=128, num_heads=4)
        hidden = _rand_hidden(4, 128)

        # Patch once
        _patch_module(attn)
        with torch.no_grad():
            out_once = attn(hidden)[0]

        # Patch again — should be a no-op due to the _sdpa_patched guard
        _patch_module(attn)
        with torch.no_grad():
            out_twice = attn(hidden)[0]

        torch.testing.assert_close(
            out_twice, out_once, atol=0, rtol=0,
            msg="Patching twice gives different output — idempotency guard is broken",
        )

    def test_output_attentions_falls_back_to_eager(self):
        """
        When output_attentions=True the patch must fall back to the original eager
        path, because SDPA does not return attention weight tensors.
        Failure means users who inspect attention weights get wrong values.
        """
        attn = _make_self_attn(dim=128, num_heads=4)
        hidden = _rand_hidden(4, 128)

        # Record eager output WITH attention weights
        with torch.no_grad():
            eager_ctx, eager_attn_weights = attn(hidden, output_attentions=True)

        # Patch, then request attention weights — should use the original code path
        _patch_module(attn)
        with torch.no_grad():
            patched_ctx, patched_attn_weights = attn(hidden, output_attentions=True)

        # Context layer must match
        torch.testing.assert_close(
            patched_ctx, eager_ctx, atol=1e-5, rtol=1e-5,
            msg="Fallback context layer differs from eager",
        )
        # Attention weights must also match (same code path is used)
        torch.testing.assert_close(
            patched_attn_weights, eager_attn_weights, atol=1e-5, rtol=1e-5,
            msg="Fallback attention weights differ from eager",
        )

    def test_shift_mask_minus100_yields_no_nan(self):
        """
        Softmax over scores where some entries are -100 must not produce NaN.
        The -100 values should produce exp(-100) ≈ 0, not 0/0.
        Failure indicates a degenerate softmax for fully-masked windows.
        """
        attn = _make_self_attn(dim=128, num_heads=4)
        nw = 16
        hidden = _rand_hidden(nw, 128)
        mask = _shift_mask(nw)

        # Verify mask does contain -100 values (otherwise this test is vacuous)
        assert (mask == -100.0).any(), "Test mask contains no -100 entries — bug in _shift_mask"

        _patch_module(attn)
        with torch.no_grad():
            out = attn(hidden, attention_mask=mask)[0]

        assert not torch.isnan(out).any(), "NaN output when shift_mask contains -100"
        assert not torch.isinf(out).any(), "Inf output when shift_mask contains -100"

    def test_masked_positions_do_not_affect_output(self):
        """
        Tokens in positions that are completely blocked by the mask (entire row
        of -100) should receive zero attention weight, so changing their values
        must not change the output.

        Identifies windows where every row in the mask is all -100 for at least
        one token, then perturbs exactly those value projections and checks output
        is unchanged.

        This tests SEMANTIC correctness of the mask, independent of numerical
        comparison to eager.
        """
        attn = _make_self_attn(dim=128, num_heads=4)
        nw = 16
        mask = _shift_mask(nw)  # (nw, 100, 100)

        # Find (window_idx, token_idx) pairs where ALL other tokens are blocked
        # i.e., mask[w, q, :] == -100 for every k != q  (token q can only attend to itself)
        # and mask[w, :, k] == -100 for every q != k  (token k is never attended to)
        fully_blocked_keys = (mask == -100.0).all(dim=1)  # (nw, 100): True where col k is all -100
        has_fully_blocked = fully_blocked_keys.any()

        if not has_fully_blocked:
            pytest.skip("No fully-blocked key positions in the generated mask")

        torch.manual_seed(7)
        hidden = _rand_hidden(nw, 128)

        _patch_module(attn)
        with torch.no_grad():
            out_original = attn(hidden, attention_mask=mask)[0]

        # Perturb the values at fully-blocked key positions (large noise)
        hidden_perturbed = hidden.clone()
        for w in range(nw):
            for k in range(SEQ_LEN):
                if fully_blocked_keys[w, k]:
                    hidden_perturbed[w, k] = hidden_perturbed[w, k] * 1000.0

        with torch.no_grad():
            out_perturbed = attn(hidden_perturbed, attention_mask=mask)[0]

        torch.testing.assert_close(
            out_perturbed, out_original, atol=1e-4, rtol=1e-4,
            msg="Perturbing fully-masked tokens changed output — mask not applied correctly",
        )

    def test_relative_position_bias_is_included(self):
        """
        The relative position bias must be included in the SDPA computation.
        Removing it would silently break positional encoding.

        We verify: SDPA output is NOT the same as a computation with no bias.
        """
        attn = _make_self_attn(dim=128, num_heads=4)
        # HuggingFace zero-initialises relative_position_bias_table; fill with
        # non-zero random values so the bias has a measurable effect.
        with torch.no_grad():
            torch.manual_seed(42)
            attn.relative_position_bias_table.normal_()
        hidden = _rand_hidden(4, 128)

        # SDPA output (includes rel_pos_bias)
        actual = _sdpa_output(attn, hidden)

        # Manually zero out the bias table and run eager — if SDPA also ignores bias
        # the two would match, which would indicate the bias is missing.
        attn_no_bias = _make_self_attn(dim=128, num_heads=4)
        # Copy weights except zero the bias table
        attn_no_bias.load_state_dict(attn.state_dict())
        with torch.no_grad():
            attn_no_bias.relative_position_bias_table.zero_()

        output_no_bias = _sdpa_output(attn_no_bias, hidden)

        assert not torch.allclose(actual, output_no_bias, atol=1e-3), (
            "SDPA output is identical with and without rel_pos_bias — bias is being dropped"
        )

    def test_output_shape_matches_eager(self):
        """SDPA output shape must be identical to eager for all valid input shapes."""
        for dim, num_heads in [(128, 4), (512, 16)]:
            for batch_windows in [1, 4, 8, 16]:
                attn = _make_self_attn(dim=dim, num_heads=num_heads)
                hidden = _rand_hidden(batch_windows, dim)

                expected_shape = _eager_output(attn, hidden).shape
                actual_shape = _sdpa_output(attn, hidden).shape

                assert actual_shape == expected_shape, (
                    f"Shape mismatch at dim={dim}, batch_windows={batch_windows}: "
                    f"sdpa={actual_shape} vs eager={expected_shape}"
                )


# ============================================================================
# Group 5 — patch_swin_sdpa model-level walker
# ============================================================================


class TestPatchSwinSdpa:
    """patch_swin_sdpa() correctly applies the SDPA forward to the full model."""

    @pytest.fixture
    def mock_model(self):
        """
        A minimal fake model that has the same structure as model.encoder.encoder.layers
        that patch_swin_sdpa iterates over.  No weights are loaded from disk.
        """
        from types import SimpleNamespace

        dim, num_heads = 128, 4
        n_stages = 2
        blocks_per_stage = 2

        stages = []
        for _ in range(n_stages):
            blocks = [_make_self_attn(dim, num_heads) for _ in range(blocks_per_stage)]
            # Wrap each self-attention as block.attention.self (mirroring DonutSwinLayer)
            stage_blocks = []
            for sa in blocks:
                attn_obj = SimpleNamespace(self=sa)
                block = SimpleNamespace(attention=attn_obj)
                stage_blocks.append(block)
            stages.append(SimpleNamespace(blocks=stage_blocks))

        encoder = SimpleNamespace(layers=stages)
        encoder_outer = SimpleNamespace(encoder=encoder)
        return SimpleNamespace(encoder=encoder_outer)

    def test_patch_swin_sdpa_replaces_all_blocks(self, mock_model):
        """
        After patch_swin_sdpa, every block.attention.self in every stage must
        have _sdpa_patched=True and a replaced forward method.
        Failure means some blocks are still using the slow eager path.
        """
        patch_swin_sdpa(mock_model)

        for stage in mock_model.encoder.encoder.layers:
            for block in stage.blocks:
                sa = block.attention.self
                assert hasattr(sa, "_sdpa_patched"), (
                    "block.attention.self missing _sdpa_patched after patch_swin_sdpa"
                )
                assert hasattr(sa, "_original_forward"), (
                    "block.attention.self missing _original_forward (original not saved)"
                )

    def test_patch_swin_sdpa_idempotent_on_model(self, mock_model):
        """
        Calling patch_swin_sdpa twice on the same model must not double-wrap
        any block.  The second call must be a complete no-op.
        """
        patch_swin_sdpa(mock_model)
        # Record forward method references after first patch
        forwards_after_one = [
            block.attention.self.forward
            for stage in mock_model.encoder.encoder.layers
            for block in stage.blocks
        ]

        patch_swin_sdpa(mock_model)
        forwards_after_two = [
            block.attention.self.forward
            for stage in mock_model.encoder.encoder.layers
            for block in stage.blocks
        ]

        for f1, f2 in zip(forwards_after_one, forwards_after_two):
            assert f1 is f2, (
                "patch_swin_sdpa replaced a forward method that was already patched"
            )

    def test_patched_blocks_produce_correct_output(self, mock_model):
        """
        After patch_swin_sdpa, running a block through its patched forward
        must give the same result as the original eager forward.
        """
        # Record eager outputs for each block before patching
        dim = 128
        nw, hidden_states = 4, _rand_hidden(4, dim)
        eager_outputs = []
        for stage in mock_model.encoder.encoder.layers:
            for block in stage.blocks:
                with torch.no_grad():
                    eager_outputs.append(block.attention.self(hidden_states)[0].clone())

        # Now patch
        patch_swin_sdpa(mock_model)

        # Verify each block's patched output matches the saved eager output
        for i, (stage, out_eager) in enumerate(
            zip(mock_model.encoder.encoder.layers, [eager_outputs[i * 2 : i * 2 + 2] for i in range(len(mock_model.encoder.encoder.layers))])
        ):
            for block, expected in zip(stage.blocks, out_eager):
                with torch.no_grad():
                    actual = block.attention.self(hidden_states)[0]
                torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)


# ============================================================================
# Group 6 — Decoder config activation
# ============================================================================


class TestDecoderConfigActivation:
    """activate_decoder_sdpa and activate_decoder_fa2 set the right config values."""

    @pytest.fixture
    def mock_decoder_model(self):
        from types import SimpleNamespace
        config = SimpleNamespace()  # no _attn_implementation set yet
        decoder = SimpleNamespace(config=config)
        return SimpleNamespace(decoder=decoder)

    def test_activate_decoder_sdpa_sets_config(self, mock_decoder_model):
        """
        activate_decoder_sdpa must set decoder.config._attn_implementation = 'sdpa'.
        This is what triggers MBart's built-in SDPA dispatch.
        """
        activate_decoder_sdpa(mock_decoder_model)
        assert mock_decoder_model.decoder.config._attn_implementation == "sdpa"

    def test_activate_decoder_fa2_sets_config(self, mock_decoder_model):
        """
        activate_decoder_fa2 must set decoder.config._attn_implementation = 'flash_attention_2'.
        Failure means FA2 dispatch is never triggered in MBart.
        """
        activate_decoder_fa2(mock_decoder_model)
        assert mock_decoder_model.decoder.config._attn_implementation == "flash_attention_2"

    def test_sdpa_and_fa2_set_different_values(self, mock_decoder_model):
        """SDPA and FA2 activations must not be silently identical."""
        from types import SimpleNamespace
        m1 = SimpleNamespace(decoder=SimpleNamespace(config=SimpleNamespace()))
        m2 = SimpleNamespace(decoder=SimpleNamespace(config=SimpleNamespace()))

        activate_decoder_sdpa(m1)
        activate_decoder_fa2(m2)

        assert m1.decoder.config._attn_implementation != m2.decoder.config._attn_implementation
