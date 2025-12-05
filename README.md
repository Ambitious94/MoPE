# MoPE-Transformer (Mixture of Pipeline Experts)

MoPE-Transformer explores how to replace a Transformer feed-forward network (FFN) with a **Mixture of Pipeline Experts (MoPE)** that executes tool- or reasoning-oriented pipelines (e.g., planner → search → reader → verifier). The project provides a lightweight research scaffold for experimenting with gate policies, pipeline composition, and training curricula for search / QA / fact-checking tasks.

## Motivation
Traditional FFNs apply a uniform nonlinear transformation to every token. MoPE treats the FFN slot as a **structural decision point** where the model can pick among multiple executable pipelines with different cost/accuracy trade-offs. This enables:

- **Adaptive strategy selection:** choose cheap or strong pipelines per token or step.
- **Explicit reasoning traces:** every pipeline emits a step-by-step trace for auditability.
- **Task-aware computation:** pipelines integrate search, retrieval, comparison, and verification without leaving the Transformer graph conceptually.

## Repository layout
- `mope/` — Python package that defines pipeline experts, gate logic, vectorizers, and MoPE layers.
- `docs/` — Architecture notes and training curriculum.
- `tests/` — Pytest coverage for pipeline routing and vectorization behavior.

## Quick start
1. (Optional) Install dev dependency for running tests (runtime uses only the standard library):
   ```bash
   pip install -r requirements.txt
   ```
2. Run tests:
   ```bash
   pytest
   ```
3. Explore the minimal demo in `mope/model.py` to see how a `MoPETransformer` routes to different pipeline experts.

## Key ideas
- **Pipelines as experts:** a pipeline is an ordered chain of atomic reasoning tools (planner, search, reader, verifier). Different chains capture different search/QA strategies.
- **Gate-driven routing:** a gate projects the hidden state to select the most promising pipeline while keeping probability distributions for analysis.
- **Vectorization:** pipeline textual outputs are mapped back to vector space so the Transformer can continue computation.
- **Training curriculum:** start with FFN distillation for stability, then supervised fine-tuning on search data, and finally RL to trade off accuracy, cost, and factuality.

See `docs/architecture.md` for a deeper dive.
