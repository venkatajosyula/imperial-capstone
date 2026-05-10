# Black-Box Optimisation (BBO) Capstone

## Section 1: Project Overview

This project tackles a **Black-Box Optimisation (BBO)** challenge: eight unknown functions (F1–F8) must be maximised without access to their source code, gradients, or analytical form. Each function can only be queried by submitting an input vector and observing the returned scalar. The challenge mirrors a broad class of real-world problems, including hyperparameter tuning, experimental design, and engineering optimisation, where the objective is expensive to evaluate and its structure is entirely unknown.

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

**Output:** A single scalar response value returned by the evaluation system after each submission. Values range widely across functions, from near-zero (F1) to large positives (F5: ~1088) and persistent negatives (F4, F6). The output signal is the sole feedback available for the next query decision.

---

## Section 3: Challenge Objectives

The goal is to **maximise** each function's output over a fixed horizon of weekly queries (one query per function per round). Key constraints:

1. **Limited budget:** One query per function per week; wasted queries cannot be recovered.
2. **No gradient information:** The function structure, smoothness, number of local optima, and feature interactions are all unknown.
3. **Response delay:** Each query result is only available after submission, so strategy must be planned one round ahead.
4. **Unknown function structure:** Functions may be non-linear, multimodal, or nearly flat in observed regions (e.g. F1 returned ≈ 0 across three consecutive rounds).

The challenge therefore requires explicitly trading off **exploitation** (concentrating queries on observed high-output regions) against **exploration** (sampling untested areas to avoid local optima).

---

## Section 4: Technical Approach

### Round 1: Weighted Blend Heuristic
With only 10 seed observations per function, no evaluation feedback had yet been received. A fixed exploitation heuristic was applied:

```
query = 0.6 × best_input + 0.3 × second_best_input + 0.1 × third_best_input
```

All dimensions were clipped to `[0, 0.999999]`. This produced new bests for F7 and F8 but underperformed on F2–F6, where uniform averaging displaced the query from the seed maximum.

### Round 2: Feedback-Adaptive Blend
Week 1 evaluation outputs were incorporated before re-ranking, expanding each dataset to 11 points. The same 60/30/10 blend was applied to the updated top-3, making it **adaptive to feedback** rather than purely seed-driven. F2 (+16%) and F4 (+13%) exceeded their seed maxima for the first time. F1 returned an identical near-zero result for the second consecutive round, flagging it as a degenerate or flat-region problem.

### Round 3: Dual Strategy: Heuristic + SVM Classifier
A parallel SVM-based query generator was introduced alongside the heuristic baseline (Module 14.1):

1. Observations are binary-labelled: **high** (above 50th percentile) vs **low**.
2. A **soft-margin RBF SVC** (C=1.0) is fitted on this labelled dataset.
3. 10,000 random candidate points are scored by decision function; the argmax becomes the SVM query.

The heuristic remained the submitted query at 12 points per function (too sparse in 4D–8D to trust the SVM boundary over objectively observed maxima), but SVM queries were documented as parallel alternatives. The SVM diverged most strongly for F1, where the heuristic has produced no usable signal, making it the primary candidate for a strategy switch in Round 4.

### Round 4: Neural Network Surrogate + Gradient-Guided Querying
Round 4 was upgraded to a Module 15-aligned approach using neural networks, gradient descent ideas, and backpropagation-style sensitivity analysis.

Core process:

1. Train one neural network surrogate per function on cumulative data (Week 3 datasets as primary source).
2. Sample 30,000 random candidate points in the valid domain `[0, 1)`.
3. Score each candidate by predicted output plus a distance term to avoid re-submitting old points.
4. Apply gradient-ascent-style refinement (numerical input gradients) from top candidate starts.
5. Select one final query per function, clipped to `[0, 0.999999]` and formatted to six decimals.

Compared with earlier rounds, Round 4 moved from fixed blending and classification-style region selection to a single neural-network surrogate workflow for all functions. The submission remained one query per function, and all generated queries passed schema checks (function-wise dimensions, value bounds in `[0, 1)`, and six-decimal formatting).

**Planned evolution:** Continue the NN surrogate pipeline while increasing robustness as more data arrives: tune architecture/regularization per function, improve gradient-step scheduling, and maintain explicit exploration controls so high-dimensional search does not collapse to narrow local regions.

### Round 5: Neural Network Surrogate + Hierarchical Feature Awareness
Round 5 continued the neural network surrogate pipeline from Round 4, now applied to 14 cumulative data points for F1/F2 and proportionally more for higher-dimensional functions (up to 44 for F8). The core process was identical to Round 4 but with two refinements informed by Module 16 concepts:

1. **Hierarchical feature awareness:** The two-hidden-layer architecture was retained to preserve capacity for capturing dimension interactions, not just individual correlations. Gradient analysis confirmed that F5 is primarily driven by dimensions 3 and 4 (magnitudes ~1027 and ~927), which guided query placement.
2. **Reduced candidate set:** Candidate sampling was reduced from 30,000 to 5,000 and top starts from 30 to 10 to keep runtime practical at this data scale, without materially affecting query quality.

Data lineage for Round 5: `week-3/data/function_X/` cumulative `.npy` files (rounds 1–3, 12 points for 2D functions) were extended with rounds 3 and 4 results parsed from `week-5/inputs.txt` and `week-5/outputs.txt`, then saved to `week-4/data/function_X/` as the Round 5 training source.

**Round 5 submitted queries:**

| Function | Query | Predicted output |
|----------|-------|-----------------|
| F1 | `0.999999-0.716687` | 0.00115 |
| F2 | `0.670109-0.283738` | 0.642 |
| F3 | `0.999999-0.668856-0.999999` | 0.071 |
| F4 | `0.553885-0.455830-0.393990-0.235527` | −3.605 |
| F5 | `0.375803-0.678335-0.999999-0.999999` | 2628 |
| F6 | `0.584100-0.000000-0.532219-0.673288-0.094311` | −0.514 |
| F7 | `0.353550-0.351452-0.319921-0.000000-0.324359-0.931710` | 2.887 |
| F8 | `0.265410-0.000000-0.000000-0.374231-0.968181-0.081225-0.253762-0.840590` | 10.195 |

**Planned evolution:** As data grows beyond ~20 points per function, explore ensemble surrogates or Gaussian process alternatives to improve uncertainty quantification, and consider a per-function exploration budget that adapts based on observed output variance.

### Round 6: Neural Network Surrogate — 15 Data Points
Round 6 continues the neural network surrogate pipeline, now trained on 15 cumulative data points for F1/F2 and proportionally more for higher-dimensional functions (up to 45 for F8). The strategy and architecture are unchanged from Round 5, reflecting the Module 17 parallel with CNNs: the same depth (two hidden layers, tanh activation) is retained to maintain capacity for dimension interactions without overfitting on the small dataset. Regularisation (L2 weight decay, α=1e-3) provides the same role as dropout in a CNN — reducing reliance on individual inputs and improving generalisation.

Data lineage for Round 6: `week-4/data/function_X/` cumulative `.npy` files (rounds 1–4, 14 points for 2D functions) were extended with the round-5 result parsed from `week-6/inputs.txt` and `week-6/outputs.txt`, then saved to `week-5/data/function_X/` as the Round 6 training source.

**Round 6 submitted queries:**

| Function | Query | Predicted output |
|----------|-------|-----------------|
| F1 | `0.000000-0.062404` | 0.000342 |
| F2 | `0.954448-0.000000` | 0.628 |
| F3 | `0.591243-0.000000-0.703210` | 0.089 |
| F4 | `0.577846-0.442866-0.377803-0.239652` | −3.803 |
| F5 | `0.347755-0.826999-0.999999-0.999999` | 2468 |
| F6 | `0.627991-0.307931-0.538595-0.855392-0.166437` | −0.545 |
| F7 | `0.175912-0.292417-0.399772-0.190708-0.383330-0.593055` | 2.670 |
| F8 | `0.000000-0.092187-0.000000-0.313546-0.351845-0.999999-0.074649-0.367430` | 10.167 |

**Planned evolution:** With 15+ points per function, the surrogate is becoming more reliable for lower-dimensional functions. Next steps: investigate per-function architecture tuning (deeper networks for higher-dimensional functions), and introduce an explicit uncertainty measure to more systematically balance exploration and exploitation.

### Round 7: Neural Network Surrogate — 16 Data Points
Round 7 extends the neural network surrogate pipeline to 16 cumulative data points for F1/F2 and proportionally more for higher-dimensional functions (46 for F8). The strategy remains identical to Rounds 5 and 6, with no architectural changes. The incremental data addition continues to refine the surrogate on the growing dataset while maintaining the same two-layer tanh network and L2 regularization.

Data lineage for Round 7: `week-5/data/function_X/` cumulative `.npy` files (rounds 1–5, 15 points for 2D functions) were extended with the round-6 result parsed from `week-7/inputs.txt` and `week-7/outputs.txt`, then saved to `week-6/data/function_X/` as the Round 7 training source.

**Round 7 submitted queries:**

| Function | Query | Predicted output |
|----------|-------|-----------------|
| F1 | `0.000000-0.997451` | 0.000232 |
| F2 | `0.632322-0.999999` | 0.624502 |
| F3 | `0.440601-0.635553-0.618984` | 0.020438 |
| F4 | `0.552008-0.484773-0.419586-0.254089` | −4.00784 |
| F5 | `0.388378-0.892873-0.999999-0.999999` | 2885.3 |
| F6 | `0.596418-0.290466-0.535810-0.999999-0.260912` | −0.4425 |
| F7 | `0.000000-0.314012-0.431251-0.161961-0.434593-0.698083` | 2.7511 |
| F8 | `0.000000-0.446318-0.000000-0.000000-0.546491-0.000000-0.595139-0.999999` | 10.176 |

**Planned evolution:** As the dataset approaches 20+ points per function, the neural network surrogate should benefit from more stable cross-validation estimates and clearer landscape inference. Future refinements include adaptive regularization per function based on data variance, and consideration of ensemble methods to better estimate prediction uncertainty for improved exploration guidance.
