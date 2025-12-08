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
4. Build retrieval-ready pipelines with your own corpus:
   ```python
   from mope import DocumentStore, MoPELayer, MoPELayerConfig, build_retrieval_pipelines

   store = DocumentStore()
   store.add_many([
       ("wikipedia-water", "Water boils at 100C at sea level."),
       ("physics", "Boiling point decreases with altitude."),
   ])

   pipelines = build_retrieval_pipelines(store)
   config = MoPELayerConfig(hidden_size=8, expert_names=list(pipelines.keys()))
   layer = MoPELayer(config, pipelines=pipelines)
   result = layer.forward([0.0] * 8, prompt="When does water boil?")
   print(result["trace"])  # includes retrieval + reader evidence
   ```

5. Run a task-level search/QA/fact-checking flow and connect it to MoPE:
   ```python
   from mope import (
       DocumentStore,
       MoPELayer,
       MoPELayerConfig,
       SearchQAFactCheckingSystem,
       build_task_pipelines,
   )

   store = DocumentStore()
   store.add_many([
       ("wiki-boiling", "Water boils at 100 degrees Celsius at sea level."),
       ("wiki-altitude", "At high altitude, the boiling point drops."),
   ])

   system = SearchQAFactCheckingSystem(store)
   pipelines = build_task_pipelines(system)
   config = MoPELayerConfig(hidden_size=8, expert_names=list(pipelines.keys()))
   layer = MoPELayer(config, pipelines=pipelines)
   result = layer.forward([0.0] * config.hidden_size, prompt="When does water boil?")
   print(result["trace"])  # retrieval + reading + verdict
   ```

## Key ideas
- **Pipelines as experts:** a pipeline is an ordered chain of atomic reasoning tools (planner, search, reader, verifier). Different chains capture different search/QA strategies.
- **Gate-driven routing:** a gate projects the hidden state to select the most promising pipeline while keeping probability distributions for analysis.
- **Vectorization:** pipeline textual outputs are mapped back to vector space so the Transformer can continue computation.
- **Training curriculum:** start with FFN distillation for stability, then supervised fine-tuning on search data, and finally RL to trade off accuracy, cost, and factuality.

See `docs/architecture.md` for a deeper dive.

## nanoGPT integration
Use the `NanoGPTMoPEAdapter` to replace the MLP block of selected nanoGPT layers.
```python
from mope import (
    DocumentStore,
    SearchQAFactCheckingSystem,
    build_task_pipelines,
    attach_mope_to_nanogpt,
    make_mock_nanogpt,
)

mock_model, hidden_size = make_mock_nanogpt(num_layers=1, hidden_size=12)
store = DocumentStore()
store.add("wiki-boiling", "Water boils at 100 C at sea level.")
system = SearchQAFactCheckingSystem(store)
pipelines = build_task_pipelines(system)
attach_mope_to_nanogpt(
    mock_model,
    hidden_size=hidden_size,
    layer_indices=[0],
    pipelines=pipelines,
    prompt_provider=lambda: "When does water boil?",
)

# during generation, call the patched mlp with your hidden state tensor or list
hidden = mock_model.transformer.h[0].mlp([0.0] * hidden_size)
```
