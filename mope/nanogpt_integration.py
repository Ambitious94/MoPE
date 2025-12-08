"""Helpers for attaching MoPE layers to a nanoGPT-style model."""

from __future__ import annotations

import importlib
import importlib.util
from types import SimpleNamespace
from typing import Callable, Iterable

from .mope_layer import MoPELayer, MoPELayerConfig
from .pipeline import PIPELINE_REGISTRY


def _is_torch_tensor(obj) -> bool:
    spec = importlib.util.find_spec("torch")
    if spec is None:
        return False
    torch = importlib.import_module("torch")
    return isinstance(obj, torch.Tensor)


class NanoGPTMoPEAdapter:
    """Replaces a nanoGPT MLP with a MoPE layer call.

    The adapter keeps the interface callable like an MLP block while allowing
    users to pass the originating prompt as a keyword argument. For simplicity
    the adapter updates only the last sequence position, which is often the
    target token during next-token prediction.
    """

    def __init__(self, mope_layer: MoPELayer, prompt_provider: Callable[[], str] | None = None):
        self.mope_layer = mope_layer
        self.prompt_provider = prompt_provider or (lambda: "")

    def __call__(self, hidden, *, prompt: str | None = None):
        prompt_text = prompt if prompt is not None else self.prompt_provider()

        if _is_torch_tensor(hidden):
            torch = importlib.import_module("torch")
            last_token = hidden[:, -1, :]
            vector = last_token.detach().cpu().tolist()[0]
            output = self.mope_layer.forward(vector, prompt_text)
            update = torch.tensor(output["hidden_state"], device=hidden.device, dtype=hidden.dtype)
            hidden[:, -1, :] = update
            return hidden

        if isinstance(hidden, (list, tuple)):
            output = self.mope_layer.forward(hidden, prompt_text)
            return output["hidden_state"]

        raise TypeError("unsupported hidden representation for nanoGPT adapter")


def attach_mope_to_nanogpt(
    model,
    *,
    hidden_size: int,
    layer_indices: Iterable[int],
    pipelines=None,
    prompt_provider: Callable[[], str] | None = None,
):
    """Replace the MLP of selected nanoGPT layers with MoPE adapters.

    The function assumes the model has a ``transformer.h`` list of blocks, each
    with an ``mlp`` attribute (matching the upstream nanoGPT structure). The
    returned model is mutated in-place for convenience.
    """

    pipelines = pipelines or PIPELINE_REGISTRY
    expert_names = list(pipelines.keys())
    config = MoPELayerConfig(hidden_size=hidden_size, expert_names=expert_names)

    for idx in layer_indices:
        block = model.transformer.h[idx]
        mope_layer = MoPELayer(config, pipelines=pipelines)
        block.mlp = NanoGPTMoPEAdapter(mope_layer, prompt_provider=prompt_provider)

    return model


def make_mock_nanogpt(num_layers: int = 2, hidden_size: int = 8):
    """Builds a toy nanoGPT-like model for offline testing."""

    class DummyMLP:
        def __call__(self, hidden, *_, **__):
            return hidden

    blocks = [SimpleNamespace(mlp=DummyMLP()) for _ in range(num_layers)]
    return SimpleNamespace(transformer=SimpleNamespace(h=blocks)), hidden_size
