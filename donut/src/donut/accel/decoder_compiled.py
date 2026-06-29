"""torch.compile + static KV cache for the MBart decoder (SKELETON).

The decode loop runs at query_len=1: many tiny kernels per token, GPU idling in
launch gaps (see research/decode-profiler.md). A *static* KV cache gives every
decode step the same tensor shapes, which lets torch.compile capture CUDA graphs
and replay them with near-zero per-step launch overhead.

This is a non-functional stub that matches the accel preset interface
(apply/revert/check, idempotent, revertable) so it can be wired into PRESETS once
fleshed out. The actual torch.compile / static-cache calls are TODOs — see
research/compile-static-cache.md for the plan and the risk list.
"""


def apply_decoder_compiled(model) -> None:
    """Pin a static KV cache and compile the decoder forward. Idempotent. STUB."""
    if getattr(model, "_compiled_decoder", False):
        return
    # Static cache = fixed-shape KV across steps -> stable shapes for graph capture.
    # generate() honours this via generation_config; no per-call kwarg needed.
    model.generation_config.cache_implementation = "static"
    # TODO: model.decoder.forward = torch.compile(
    #           model.decoder.forward, mode="reduce-overhead", fullgraph=False)
    #   - reduce-overhead => CUDA graphs (the whole point at q=1).
    #   - fullgraph: start False, tighten once graph breaks are gone.
    #   - Consider compiling model.forward so the cross-attn read of encoder KV is
    #     captured too; measure both. Watch for recompiles when batch_size / kv_len
    #     change (mark dynamic dims or pad to a bucket).
    model._compiled_decoder = True


def revert_decoder_compiled(model) -> None:
    """Undo apply. No-op if not applied."""
    if not getattr(model, "_compiled_decoder", False):
        return
    model.generation_config.cache_implementation = None
    # TODO: if we actually compiled, restore the original forward and call
    #       torch._dynamo.reset() so a later preset starts from a clean graph cache.
    model._compiled_decoder = False


def check_decoder_compiled(model) -> None:
    """Assert the compiled/static-cache state is active."""
    assert getattr(model, "_compiled_decoder", False), (
        "decoder_compiled not applied (call apply_decoder_compiled first)"
    )
    impl = model.generation_config.cache_implementation
    assert impl == "static", f"cache_implementation is {impl!r}, expected 'static'"
