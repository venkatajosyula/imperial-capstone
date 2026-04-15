# Black-Box Optimisation (BBO) Capstone

## Section 1: Project Overview

This project tackles a **Black-Box Optimisation (BBO)** challenge: eight unknown functions (F1–F8) must be maximised without access to their source code, gradients, or analytical form. Each function can only be queried by submitting an input vector and observing the returned scalar. The challenge mirrors a broad class of real-world problems — hyperparameter tuning, experimental design, and engineering optimisation — where the objective is expensive to evaluate and its structure is entirely unknown.

From a career perspective, BBO directly underpins AutoML and model selection workflows. Developing disciplined query strategies under uncertainty, with traceable rationale and reproducible code, is a transferable skill for any data science or MLOps role where evaluation budgets are finite and feedback is delayed.

---

## Section 2: Inputs and Outputs

**Input format:** A real-valued vector with all dimensions drawn from the continuous range `[0, 1)`. Each function has a fixed dimensionality:

| Function | Dimensions |
|----------|-----------|
| F1, F2   | 2D        |
| F3       | 3D        |
| F4, F5   | 4D        |
| F6       | 5D        |
| F7       | 6D        |
| F8       | 8D        |

Example query (F8): `0.139785-0.239023-0.160745-0.238163-0.505938-0.701437-0.392239-0.696102`

**Output:** A single scalar response value returned by the evaluation system after each submission. Values range widely across functions — from near-zero (F1) to large positives (F5: ~1088) and persistent negatives (F4, F6). The output signal is the sole feedback available for the next query decision.

---

## Section 3: Challenge Objectives

The goal is to **maximise** each function's output over a fixed horizon of weekly queries (one query per function per round). Key constraints:

- **Limited budget:** One query per function per week; wasted queries cannot be recovered.
- **No gradient information:** The function structure, smoothness, number of local optima, and feature interactions are all unknown.
- **Response delay:** Each query result is only available after submission, so strategy must be planned one round ahead.
- **Unknown function structure:** Functions may be non-linear, multimodal, or nearly flat in observed regions (e.g. F1 returned ≈ 0 across three consecutive rounds).

The challenge therefore requires explicitly trading off **exploitation** (concentrating queries on observed high-output regions) against **exploration** (sampling untested areas to avoid local optima).

---

## Section 4: Technical Approach

### Round 1 — Weighted Blend Heuristic
With only 10 seed observations per function, no evaluation feedback had yet been received. A fixed exploitation heuristic was applied:

```
query = 0.6 × best_input + 0.3 × second_best_input + 0.1 × third_best_input
```

All dimensions were clipped to `[0, 0.999999]`. This produced new bests for F7 and F8 but underperformed on F2–F6, where uniform averaging displaced the query from the seed maximum.

### Round 2 — Feedback-Adaptive Blend
Week 1 evaluation outputs were incorporated before re-ranking, expanding each dataset to 11 points. The same 60/30/10 blend was applied to the updated top-3, making it **adaptive to feedback** rather than purely seed-driven. F2 (+16%) and F4 (+13%) exceeded their seed maxima for the first time. F1 returned an identical near-zero result for the second consecutive round, flagging it as a degenerate or flat-region problem.

### Round 3 — Dual Strategy: Heuristic + SVM Classifier
A parallel SVM-based query generator was introduced alongside the heuristic baseline (Module 14.1):

- Observations are binary-labelled: **high** (above 50th percentile) vs **low**.
- A **soft-margin RBF SVC** (C=1.0) is fitted on this labelled dataset.
- 10,000 random candidate points are scored by decision function; the argmax becomes the SVM query.

The heuristic remained the submitted query at 12 points per function (too sparse in 4D–8D to trust the SVM boundary over objectively observed maxima), but SVM queries were documented as parallel alternatives. The SVM diverged most strongly for F1 — where the heuristic has produced no usable signal — making it the primary candidate for a strategy switch in Round 4.

### Round 4 — Neural Network Surrogate + Gradient-Guided Querying
Round 4 was upgraded to a Module 15-aligned approach using neural networks, gradient descent ideas, and backpropagation-style sensitivity analysis.

Core process:

- Train one MLP surrogate per function on cumulative data (Week 3 datasets as primary source).
- Sample 30,000 random candidate points in the valid domain `[0, 1)`.
- Score each candidate by predicted output plus a distance term to avoid re-submitting old points.
- Apply gradient-ascent-style refinement (numerical input gradients) from top candidate starts.
- Select one final query per function, clipped to `[0, 0.999999]` and formatted to six decimals.

Compared with earlier rounds, Round 4 moved from fixed blending and classification-style region selection to a single neural-network surrogate workflow for all functions. The submission remained one query per function, and all generated queries passed schema checks (function-wise dimensions, value bounds in `[0, 1)`, and six-decimal formatting).

**Planned evolution:** Continue the NN surrogate pipeline while increasing robustness as more data arrives: tune architecture/regularization per function, improve gradient-step scheduling, and maintain explicit exploration controls so high-dimensional search does not collapse to narrow local regions.
