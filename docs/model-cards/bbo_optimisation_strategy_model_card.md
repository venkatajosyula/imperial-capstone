# Model Card: BBO Optimisation Strategy

## Overview: name of your approach, type and version.

- Name: BBO Sequential Optimisation Strategy
- Type: hybrid black-box optimisation pipeline (heuristic, SVM exploration, and neural surrogate)
- Version: v1.0 (documented at Week 10; challenge horizon is 13 rounds)

This approach proposes one query per function each week in an overall 13-round challenge. This document reports progress through Round 10, where the later rounds use a neural-surrogate-guided pipeline.

## Intended use: what tasks is it suitable for? What use cases should be avoided?

Suitable tasks:
- Educational black-box optimisation exercises
- Sequential query selection under low evaluation budgets
- Reproducible optimisation workflow demonstrations

Use cases to avoid:
- Safety-critical optimisation
- Regulatory or legal automation pipelines
- Contexts requiring calibrated uncertainty guarantees

## Details: explain your strategy across the ten rounds, including the techniques you used and how your approach evolved.

Round evolution:
- Rounds 1 to 2: weighted blend heuristic
- Round 3: heuristic plus SVM experimentation
- Rounds 4 to 10: neural surrogate pipeline for final submissions

Main technique used in Rounds 4 to 10:
1. Load cumulative function data.
2. Scale inputs and outputs.
3. Train `MLPRegressor` with two hidden layers, tanh activation, and L2 regularisation.
4. Sample 5000 random candidates.
5. Score by predicted value plus distance-based exploration term.
6. Refine top starts using numerical gradient ascent.
7. Export one bounded query per function.

Earlier rounds used simpler methods (weighted blending, then SVM-based exploration) before moving to the neural surrogate approach.

## Performance: summarise your results across the eight functions. What metrics did you use?

Metrics used:
- Training MSE per function
- Cross-validation MSE per function
- Predicted output at selected query
- Distance from nearest observed point

Summary across eight functions:
- Lower-dimensional functions became more stable through rounds.
- Higher-dimensional functions remained more variable and harder to generalise.
- Query formatting remained valid each week, and improvements were observed but uneven across functions.

## Assumptions and limitations: what assumptions underlie your strategy? What are its constraints or failure modes?

Assumptions:
- Local smoothness is enough for gradient-based refinement.
- Sparse cumulative observations still carry useful structure.
- Distance from observed points can act as a simple exploration signal.

Limitations and failure modes:
- Sparse high-dimensional coverage may miss strong regions.
- Surrogate uncertainty is not formally calibrated.
- Overfitting risk remains when train and CV errors diverge.
- Weekly query budget limits deeper search and ensemble alternatives.

## Ethical considerations: how does transparency support reproducibility and real-world adaptation?

Transparency and reproducibility are supported by versioned scripts, explicit data lineage, deterministic seeds, and saved diagnostics. Another researcher can replay the workflow and inspect assumptions at each round.

For real-world adaptation, this method should be treated as a learning framework, not a deployment-ready system. Safer deployment would require uncertainty calibration, stronger robustness testing, and governance checks.

## Reflection: does your current model card structure support decision-making clearly? What are its strengths and limitations? Would adding more detail improve it?

I think the current structure is clear for coursework review because it states purpose, method, metrics, assumptions, and risks in one place. Its main strength is traceability. Its limitation is that it is still high-level for advanced audit needs. Adding deeper per-function error analysis and uncertainty diagnostics would improve practical decision support.
