"""torch.compile acceleration for Donut encoder and decoder."""

import torch


def apply_compile(model, *, encoder: bool = True, decoder: bool = True) -> None:
    """Wrap encoder and/or decoder with torch.compile(dynamic=True).

    dynamic=True handles variable sequence lengths in the decoder without
    recompilation. First-call compilation adds latency; subsequent calls
    use the compiled graph. Safe to call multiple times — re-compilation
    is skipped if the module is already compiled.
    """
    if encoder and not getattr(model.encoder, "_compiled", False):
        model.encoder = torch.compile(model.encoder, dynamic=True)
        model.encoder._compiled = True
    if decoder and not getattr(model.decoder, "_compiled", False):
        model.decoder = torch.compile(model.decoder, dynamic=True)
        model.decoder._compiled = True
