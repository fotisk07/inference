"""torch.compile acceleration for Donut encoder and decoder.

Compile must always be the last optimization applied (and the first reverted):
it wraps model.encoder/model.decoder in OptimizedModule, and patching the
wrapped modules afterwards would mutate _orig_mod under a stale graph. The
preset ordering in donut.accel guarantees this.
"""

import torch


def _is_compiled(module) -> bool:
    return hasattr(module, "_orig_mod")


def apply_compile(model, *, encoder: bool = True, decoder: bool = True) -> None:
    """Wrap encoder and/or decoder with torch.compile(dynamic=True).

    dynamic=True handles variable sequence lengths in the decoder without
    recompilation. First-call compilation adds latency; subsequent calls
    use the compiled graph. Idempotent.
    """
    if encoder and not _is_compiled(model.encoder):
        model.encoder = torch.compile(model.encoder, dynamic=True)
    if decoder and not _is_compiled(model.decoder):
        model.decoder = torch.compile(model.decoder, dynamic=True)


def revert_compile(model, *, encoder: bool = True, decoder: bool = True) -> None:
    """Unwrap torch.compile, restoring the original (eager) submodules.

    torch.compile returns an OptimizedModule wrapping the original at _orig_mod;
    restoring that reference discards the compiled graph cleanly. No-op if not
    applied.
    """
    if encoder and _is_compiled(model.encoder):
        model.encoder = model.encoder._orig_mod
    if decoder and _is_compiled(model.decoder):
        model.decoder = model.decoder._orig_mod


def check_compile(model, *, encoder: bool = True, decoder: bool = True) -> None:
    """Assert encoder/decoder are torch.compile-wrapped."""
    if encoder:
        assert _is_compiled(model.encoder), "Encoder is not torch.compile-wrapped"
    if decoder:
        assert _is_compiled(model.decoder), "Decoder is not torch.compile-wrapped"
