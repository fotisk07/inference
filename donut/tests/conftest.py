import pytest
import torch

from donut.synthetic import make_pixel_values, make_tiny_model


@pytest.fixture
def tiny_model():
    # Function-scoped: apply/revert tests must not leak patches between tests.
    return make_tiny_model(seed=0)


@pytest.fixture
def pixel_values(tiny_model):
    return make_pixel_values(tiny_model, batch_size=2, seed=42)


@torch.no_grad()
def encode(model, pixel_values):
    return model.encoder(pixel_values, return_dict=True).last_hidden_state
