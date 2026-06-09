"""torch.compile acceleration for Donut encoder and decoder."""

import torch

from donut.accel.registry import Optimization, register


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


def revert_compile(model, *, encoder: bool = True, decoder: bool = True) -> None:
    """Unwrap torch.compile, restoring the original (eager) submodules.

    torch.compile returns an OptimizedModule wrapping the original at ``_orig_mod``;
    restoring that reference discards the compiled graph cleanly.
    """
    if encoder and getattr(model.encoder, "_compiled", False):
        model.encoder = model.encoder._orig_mod
    if decoder and getattr(model.decoder, "_compiled", False):
        model.decoder = model.decoder._orig_mod


@register
class Compile(Optimization):
    """Wrap encoder and/or decoder with torch.compile(dynamic=True).

    Layered on top of an attention backend. Construct with ``encoder``/``decoder``
    flags to compile only one side.
    """

    name = "compile"

    def __init__(self, *, encoder: bool = True, decoder: bool = True) -> None:
        self.encoder = encoder
        self.decoder = decoder

    def apply(self, model) -> None:
        apply_compile(model, encoder=self.encoder, decoder=self.decoder)

    def revert(self, model) -> None:
        revert_compile(model, encoder=self.encoder, decoder=self.decoder)

    def check_structural(self, model) -> None:
        if self.encoder:
            assert getattr(model.encoder, "_compiled", False), (
                "Encoder has not been compiled — apply Compile first"
            )
        if self.decoder:
            assert getattr(model.decoder, "_compiled", False), (
                "Decoder has not been compiled — apply Compile first"
            )
