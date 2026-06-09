"""Composable, reversible optimization registry for Donut acceleration.

An :class:`Optimization` is a single, named, reversible model transform that
carries its own structural and (optional) numerical checks. ``apply_accel`` runs
an ordered list of them and records the applied list on the model; ``revert_accel``
undoes them in reverse order.

This replaces the flat ``Backend`` enum's ``if/elif`` ladder. Because every
transform can cleanly revert, the old "apply backends least->most aggressive"
caveat goes away — you can build any subset, benchmark it, revert, and try
another on the same model instance.

Adding a new optimization is: subclass :class:`Optimization`, decorate it with
:func:`register`, and add it to a preset (or pass it directly to ``apply_accel``).
No edits to an enum or a dispatch ladder are required.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

if TYPE_CHECKING:
    from transformers import DonutProcessor, VisionEncoderDecoderModel


class Optimization:
    """Base class for a composable, reversible model optimization.

    Subclasses set a unique ``name`` and implement ``apply``/``revert``/
    ``check_structural``. ``check_numerical`` is optional (defaults to a no-op).
    Instances should be cheap to construct and may store per-apply state needed
    for ``revert`` (e.g. a saved config value).
    """

    name: ClassVar[str] = "optimization"

    def apply(self, model: VisionEncoderDecoderModel) -> None:
        raise NotImplementedError

    def revert(self, model: VisionEncoderDecoderModel) -> None:
        raise NotImplementedError

    def check_structural(self, model: VisionEncoderDecoderModel) -> None:
        """Assert the optimization is active on ``model``; raise on failure."""
        raise NotImplementedError

    def check_numerical(
        self, model: VisionEncoderDecoderModel, processor: DonutProcessor
    ) -> dict:
        """Optional accuracy-vs-eager check. Default: nothing to verify."""
        return {}

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r})"


REGISTRY: dict[str, type[Optimization]] = {}


def register(cls: type[Optimization]) -> type[Optimization]:
    """Class decorator that registers an Optimization under its ``name``."""
    if cls.name in REGISTRY:
        raise ValueError(f"Optimization name already registered: {cls.name!r}")
    REGISTRY[cls.name] = cls
    return cls


_APPLIED_ATTR = "_donut_optimizations"


def applied_optimizations(model) -> list[Optimization]:
    """Return the optimizations currently applied to ``model`` (in apply order)."""
    return list(getattr(model, _APPLIED_ATTR, []))


def apply_optimizations(model, opts: list[Optimization]) -> None:
    """Apply ``opts`` in order, skipping any whose ``name`` is already active.

    The applied instances are recorded on ``model`` so they can later be reverted
    and structurally checked. Idempotent by name: applying a list that overlaps
    an already-applied set only runs the new ones.
    """
    applied = getattr(model, _APPLIED_ATTR, None)
    if applied is None:
        applied = []
        setattr(model, _APPLIED_ATTR, applied)
    active = {o.name for o in applied}
    for opt in opts:
        if opt.name in active:
            continue
        opt.apply(model)
        applied.append(opt)
        active.add(opt.name)


def revert_optimizations(model) -> None:
    """Revert every applied optimization in reverse order and clear the record."""
    applied = getattr(model, _APPLIED_ATTR, [])
    for opt in reversed(applied):
        opt.revert(model)
    setattr(model, _APPLIED_ATTR, [])


def check_optimizations(model) -> None:
    """Run ``check_structural`` for every applied optimization."""
    for opt in applied_optimizations(model):
        opt.check_structural(model)
