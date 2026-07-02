# Model Card: BBO Sequential Optimisation Strategy

## Model Description

**Input**
- Function-specific vectors in `[0, 1)`
- Cumulative historical query-output pairs
- Dimensionality by function: 2D to 8D across F1-F8

**Output**
- One next-query vector per function per round
- Predicted objective score used to rank candidates

**Model Architecture**
- Hybrid strategy across project timeline:
  - Early rounds: weighted heuristic blend
  - Transition: SVM exploration experiments
  - Main pipeline (Rounds 4-13): neural surrogate (`MLPRegressor`) with structured query selection
- Main neural settings:
  - Two hidden layers
  - `tanh` activation
  - L2 regularisation (`alpha=1e-3`)
  - Candidate pool search + local gradient-style refinement

## Performance

### Evaluation approach
- Weekly observed portal output values
- Surrogate diagnostics (train/CV behavior)
- Predicted score and novelty-distance checks for selected queries

### Performance summary
- Lower-dimensional functions became progressively more stable.
- Higher-dimensional functions remained more variable due to sparse coverage.
- Strategy quality improved over time through better balance of exploitation and exploration.

### Data used for assessment
- Cumulative weekly records from all 13 rounds across F1-F8.

## Limitations

- Sparse sample coverage in higher dimensions.
- No fully calibrated probabilistic uncertainty in the core loop.
- Risk of local lock-in if exploration pressure is too low.
- Weekly one-query budget constrains breadth of experimentation.

## Trade-offs

- Prioritised reproducibility and operational stability over aggressive weekly retuning.
- Benefit: traceable and consistent workflow.
- Cost: some potential peak performance was likely left unrealised.

- Prioritised function-specific adjustments in later rounds instead of early full complexity.
- Benefit: simpler control early on.
- Cost: slower adaptation for difficult functions.

## Intended Use

Suitable for:
- Educational black-box optimisation projects
- Portfolio demonstration of iterative optimisation under constraints
- Reproducible strategy analysis and reflection

Not suitable for:
- Safety-critical decisions
- Regulated or high-stakes deployments without additional controls

## Responsible Use Notes

Before production adaptation, add:
- Calibrated uncertainty estimation
- Drift and stability monitoring
- Explicit governance and fail-safe criteria
- Deeper robustness testing and ablation reporting
