# JURIS-AGI

A Neuro-Symbolic AGI System for ARC-style Reasoning.

## Overview

JURIS-AGI implements a "jurisdiction" approach to AI reasoning where **symbolic consistency is a hard veto**. Neural/LLM components provide advisory suggestions, but all outputs must pass through certified symbolic verification.

### Core Principles

1. **Certified Reasoning**: All solutions must be expressible in a typed DSL and verified symbolically
2. **Jurisdiction**: Symbolic critics have veto power over neural suggestions
3. **Auditability**: Complete traces of reasoning, synthesis, and verification
4. **Robustness**: Counterfactual testing and invariant checking

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      Meta-Controller                             │
│         (Routes, budgets, epistemic/aleatoric uncertainty)       │
└─────────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        ▼                     ▼                     ▼
┌───────────────┐    ┌───────────────┐    ┌───────────────┐
│     CRE       │    │     WME       │    │     MAL       │
│  (Certified   │◄───│ (World Model  │◄───│  (Memory &    │
│   Reasoning)  │    │    Expert)    │    │  Abstraction) │
└───────────────┘    └───────────────┘    └───────────────┘
        │
        ▼
┌───────────────────────────────────────────────────────────────┐
│  DSL Interpreter + Symbolic Critic (JURISDICTION - HARD VETO) │
└───────────────────────────────────────────────────────────────┘
```

### Components

- **CRE (Certified Reasoning Expert)**: Program synthesis with typed DSL
- **WME (World Model Expert)**: Priors and counterfactual generation (advisory)
- **MAL (Memory & Abstraction Library)**: Pattern retrieval and macro induction
- **Meta-Controller**: Task routing and expert budgeting

## Installation

```bash
pip install -e .
# Or with dev dependencies:
pip install -e ".[dev]"
```

## Usage

### Running on ARC Tasks

```bash
# Run on a single task
python -m juris_agi.eval.run_arc data/arc_public/training/task_id.json

# Run on a directory of tasks
python -m juris_agi.eval.run_arc data/arc_public/training/ --output results/
```

### Programmatic Usage

```python
from juris_agi.controller.router import MetaController
from juris_agi.core.types import ARCTask

# Load task
task = ARCTask.from_dict("task_id", task_json)

# Create controller and solve
controller = MetaController()
result = controller.solve(task)

# Check result
if result.is_certified:
    print(f"Certified solution found: {result.audit_trace.program_source}")
```

## DSL Primitives

The DSL provides typed primitives for grid manipulation:

- `identity`: Pass-through
- `crop_to_bbox`: Crop grid to object bounding box
- `extract_objects`: Extract discrete objects
- `recolor_map`: Apply color mapping
- `translate`: Shift grid contents
- `rotate90`: Rotate by 90 degrees
- `reflect`: Mirror horizontally or vertically
- `paste`: Overlay one grid onto another
- `fill`: Fill region with color

## Testing

```bash
pytest tests/ -v
```

## Project Structure

```
juris-agi/
├── src/juris_agi/
│   ├── core/           # Types, state, metrics, tracing
│   ├── representation/ # Tokenization, objects, relations
│   ├── dsl/            # Primitives, AST, interpreter
│   ├── cre/            # Synthesizer, critics, refinement
│   ├── wme/            # World model, priors, counterfactuals
│   ├── mal/            # Retrieval, macro induction
│   ├── controller/     # Router, scheduler, refusal
│   └── eval/           # Evaluation scripts
├── tests/
├── data/
└── pyproject.toml
```

## License

MIT
