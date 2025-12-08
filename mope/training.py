"""Training curriculum sketches for MoPE-Transformer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, List, Sequence

from .mope_layer import MoPELayer


@dataclass
class DistillationBatch:
    hidden_states: Sequence[Sequence[float]]
    ffn_outputs: Sequence[Sequence[float]]
    prompts: List[str]


@dataclass
class SFTBatch:
    prompts: List[str]
    labels: List[str]


@dataclass
class RLSample:
    prompt: str
    reward: float
    trajectory: dict


def _mean_squared_error(preds: Sequence[Sequence[float]], targets: Sequence[Sequence[float]]) -> float:
    total = 0.0
    count = 0
    for p_vec, t_vec in zip(preds, targets):
        for p, t in zip(p_vec, t_vec):
            diff = p - t
            total += diff * diff
            count += 1
    return total / max(count, 1)


def distill_ffn(layer: MoPELayer, batch: DistillationBatch, loss_fn: Callable[[Sequence[Sequence[float]], Sequence[Sequence[float]]], float] | None = None) -> float:
    """Stage 1: fit MoPE outputs to a frozen FFN target."""

    predictions = []
    for hidden, prompt in zip(batch.hidden_states, batch.prompts):
        output = layer.forward(hidden, prompt)
        predictions.append(output["hidden_state"])
    metric = loss_fn or _mean_squared_error
    return float(metric(predictions, batch.ffn_outputs))


def supervised_finetune(model, batch: SFTBatch, loss_fn: Callable[[List[str], List[str]], float] | None = None) -> float:
    """Stage 2: teach pipelines to emit correct answers for labeled prompts."""

    outputs = [model.forward(prompt)["layer_traces"][-1]["trace"][-1] for prompt in batch.prompts]
    metric = loss_fn or (lambda preds, targets: sum(p != t for p, t in zip(preds, targets)) / max(len(preds), 1))
    return float(metric(outputs, batch.labels))


def reinforce_gate(model, samples: Iterable[RLSample], update_fn: Callable[[float, dict], None]) -> float:
    """Stage 3: optimize gate using reinforcement learning style updates.

    The API is framework-agnostic; `update_fn` can log gradients or update
    parameters directly depending on the caller's setup.
    """

    total_reward = 0.0
    for sample in samples:
        trajectory = model.forward(sample.prompt)
        total_reward += sample.reward
        update_fn(sample.reward, trajectory)
    return float(total_reward)
