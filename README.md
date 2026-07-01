# Black-Box Optimisation (BBO) Capstone

## Final Submission Snapshot

This repository is the final deliverable for the BBO capstone project. It is organised to make the full project story easy to follow: weekly data and code evolution, technical governance documents, and a concise explanation for non-technical readers.

### Non-Technical Project Summary

This project studies how to make good decisions when the true formula is unknown. I worked with eight hidden functions and could submit only one new input per function each week. I started with simple rules, then moved to a model-guided strategy that learned from previous outcomes and balanced trying strong regions with exploring new ones. Over 13 rounds, the process became more stable and better documented. The final result is not just a set of scores, but a reproducible optimisation workflow that explains what was tried, why strategy changed, and how decisions were made under limited feedback.

### Quick Navigation

- Final method walkthrough notebook: `src/bbo/notebooks/bbo_final_method_and_results.ipynb`
- Dataset datasheet: `src/bbo/docs/datasheets/bbo_capstone_dataset_datasheet.md`
- Model card: `src/bbo/docs/model-cards/bbo_optimisation_strategy_model_card.md`
- Presentation material: `src/bbo/docs/presentation/BBO capstone project presentation.pdf`
- Weekly round folders and source files: `src/bbo/week-1/` to `src/bbo/week-13/`
- Weekly execution helper: `src/bbo/scripts/run_week_pipeline.sh`
- Dependency + execution wrapper: `src/bbo/scripts/setup_and_run_week.sh`

Canonical submission files are stored at each week root using `submission_roundN.txt` (for example, `src/bbo/week-4/submission_round4.txt`).

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

Data lineage for Round 5: `src/bbo/week-3/data/function_X/` cumulative `.npy` files (rounds 1–3, 12 points for 2D functions) were extended with rounds 3 and 4 results parsed from `src/bbo/week-5/inputs.txt` and `src/bbo/week-5/outputs.txt`, then saved to `src/bbo/week-4/data/function_X/` as the Round 5 training source.

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

Data lineage for Round 6: `src/bbo/week-4/data/function_X/` cumulative `.npy` files (rounds 1–4, 14 points for 2D functions) were extended with the round-5 result parsed from `src/bbo/week-6/inputs.txt` and `src/bbo/week-6/outputs.txt`, then saved to `src/bbo/week-5/data/function_X/` as the Round 6 training source.

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

Data lineage for Round 7: `src/bbo/week-5/data/function_X/` cumulative `.npy` files (rounds 1–5, 15 points for 2D functions) were extended with the round-6 result parsed from `src/bbo/week-7/inputs.txt` and `src/bbo/week-7/outputs.txt`, then saved to `src/bbo/week-6/data/function_X/` as the Round 7 training source.

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

### Round 8: Neural Network Surrogate — 17 Data Points
Round 8 extends the neural network surrogate pipeline to 17 cumulative data points for F1/F2 and proportionally more for higher-dimensional functions (47 for F8). The strategy remains unchanged from Rounds 5–7: two hidden layers with tanh activation, L2 regularisation (alpha=1e-3), candidate exploration over 5,000 random points, and gradient-ascent-style refinement of top starts.

Data lineage for Round 8: `src/bbo/week-6/data/function_X/` cumulative `.npy` files (rounds 1–6, 16 points for 2D functions) were extended with the round-7 result parsed from `src/bbo/week-8/inputs.txt` and `src/bbo/week-8/outputs.txt`, then saved to `src/bbo/week-7/data/function_X/` as the Round 8 training source.

**Round 8 submitted queries:**

| Function | Query | Predicted output |
|----------|-------|-----------------|
| F1 | `0.999999-0.773695` | 0.000282 |
| F2 | `0.650633-0.637990` | 0.686180 |
| F3 | `0.990819-0.999999-0.000000` | 0.024726 |
| F4 | `0.577435-0.496284-0.366459-0.204129` | −3.92715 |
| F5 | `0.439929-0.956670-0.999999-0.999999` | 3354.48 |
| F6 | `0.651139-0.262567-0.599355-0.798422-0.091835` | −0.542635 |
| F7 | `0.307563-0.319620-0.265684-0.216323-0.357029-0.533625` | 2.83472 |
| F8 | `0.130847-0.000000-0.000000-0.350108-0.451876-0.074684-0.555836-0.938167` | 10.1524 |

**Planned evolution:** With 17+ points now available for low-dimensional functions, the next priority is function-specific calibration of exploration weight and regularisation rather than one global setting. As rounds continue, adding uncertainty-aware selection (for example via ensembles) should help reduce over-exploitation in noisy or high-dimensional regions.

### Round 9: Neural Network Surrogate — 18 Data Points
Round 9 extends the neural network surrogate pipeline to 18 cumulative data points for F1/F2 and proportionally more for higher-dimensional functions (48 for F8). The strategy remains the same as Rounds 5–8: two hidden layers with tanh activation, L2 regularisation (alpha=1e-3), 5,000 random candidates, and gradient-ascent-style refinement of the best starts.

Data lineage for Round 9: `src/bbo/week-7/data/function_X/` cumulative `.npy` files (rounds 1–7, 17 points for 2D functions) were extended with the round-8 result parsed from `src/bbo/week-9/inputs.txt` and `src/bbo/week-9/outputs.txt`, then saved to `src/bbo/week-8/data/function_X/` as the Round 9 training source.

**Round 9 submitted queries:**

| Function | Query | Predicted output |
|----------|-------|-----------------|
| F1 | `0.944470-0.000000` | 0.000201 |
| F2 | `0.999999-0.862726` | 0.685679 |
| F3 | `0.585362-0.982031-0.423506` | 0.015385 |
| F4 | `0.512202-0.487812-0.471215-0.362323` | −3.70303 |
| F5 | `0.570179-0.999999-0.999999-0.999999` | 4051.72 |
| F6 | `0.633574-0.221343-0.676984-0.853845-0.000000` | −0.476486 |
| F7 | `0.348781-0.360456-0.389134-0.281641-0.334543-0.598031` | 2.67495 |
| F8 | `0.000000-0.038573-0.000000-0.000000-0.542498-0.960513-0.612752-0.473576` | 10.033 |

**Planned evolution:** At 18 points, the surrogate is still strongest on the lower-dimensional functions, while the higher-dimensional cases remain more sensitive to search noise and model fit. The next refinement would be function-specific search settings and a more explicit uncertainty signal so the query strategy can balance exploitation and exploration more consistently.

### Round 10: Neural Network Surrogate — 19 Data Points
Round 10 extends the neural network surrogate pipeline to 19 cumulative data points for F1/F2 and proportionally more for higher-dimensional functions (49 for F8). The strategy remains consistent with Rounds 5-9: two hidden layers with tanh activation, L2 regularisation (alpha=1e-3), 5,000 random candidates, and gradient-ascent-style refinement of the top starts.

Data lineage for Round 10: `src/bbo/week-8/data/function_X/` cumulative `.npy` files (rounds 1-8, 18 points for 2D functions) were extended with the round-9 result parsed from `src/bbo/week-10/inputs.txt` and `src/bbo/week-10/outputs.txt`, then saved to `src/bbo/week-9/data/function_X/` as the Round 10 training source.

**Round 10 submitted queries:**

| Function | Query | Predicted output |
|----------|-------|-----------------|
| F1 | `0.999999-0.387219` | 0.000526 |
| F2 | `0.778580-0.325437` | 0.281417 |
| F3 | `0.593991-0.466307-0.438282` | -0.003492 |
| F4 | `0.468386-0.475520-0.442150-0.459084` | -2.65078 |
| F5 | `0.758271-0.999999-0.999999-0.999999` | 4890.55 |
| F6 | `0.629244-0.246822-0.625220-0.818379-0.040015` | -0.497866 |
| F7 | `0.281416-0.288326-0.368833-0.178561-0.459782-0.585532` | 2.61727 |
| F8 | `0.004538-0.285212-0.000000-0.344580-0.255946-0.954305-0.000000-0.995167` | 10.1869 |

**Planned evolution:** With 19 points available for low-dimensional functions, incremental gains are becoming more function-specific. The next step is to adapt exploration weight and regularisation per function and add a lightweight uncertainty signal so query selection stays robust when model confidence and generalisation quality diverge.

### Round 11: Cluster-Aware Neural Surrogate — 20 Data Points
Round 11 extends the optimisation pipeline to 20 cumulative data points for F1/F2 and proportionally more for higher-dimensional functions (50 for F8). The strategy remains neural-surrogate-guided, but now includes a clustering lens to identify local groups and use centroid-distance cues in query scoring.

Data lineage for Round 11: `src/bbo/week-9/data/function_X/` cumulative `.npy` files (rounds 1-9, 19 points for 2D functions) were extended with the round-10 result parsed from `src/bbo/week-11/inputs.txt` and `src/bbo/week-11/outputs.txt`, then saved to `src/bbo/week-10/data/function_X/` as the Round 11 training source.

**Round 11 submitted queries:**

| Function | Query | Predicted output |
|----------|-------|-----------------|
| F1 | `0.999999-0.436755` | 0.000202 |
| F2 | `0.772318-0.750760` | 0.519329 |
| F3 | `0.693560-0.417123-0.405668` | 0.008725 |
| F4 | `0.440312-0.475570-0.443247-0.487267` | -2.16175 |
| F5 | `0.999999-0.999999-0.999999-0.999999` | 5767.3 |
| F6 | `0.594263-0.257321-0.640626-0.831718-0.000000` | -0.441893 |
| F7 | `0.249603-0.337927-0.252680-0.270248-0.269419-0.636759` | 2.65593 |
| F8 | `0.455746-0.229401-0.000000-0.492928-0.998826-0.541156-0.000000-0.799043` | 10.0458 |

**Planned evolution:** For the remaining rounds, the immediate priority is to keep the cluster-aware cue but calibrate cluster influence per function so it improves exploration without overpowering strong surrogate evidence in stable regions.

### Round 12: Structure-Guided Neural Surrogate — 21 Data Points
Round 12 extends the optimisation pipeline to 21 cumulative data points for F1/F2 and proportionally more for higher-dimensional functions (51 for F8). The strategy remains neural-surrogate-guided with a structure lens: local clustering identifies recurring neighborhoods while distance-aware exploration keeps query diversity and reduces purely random search behavior.

Data lineage for Round 12: `src/bbo/week-10/data/function_X/` cumulative `.npy` files (rounds 1-10, 20 points for 2D functions) were extended with the round-11 result parsed from `src/bbo/week-12/inputs.txt` and `src/bbo/week-12/outputs.txt`, then saved to `src/bbo/week-11/data/function_X/` as the Round 12 training source.

**Round 12 submitted queries:**

| Function | Query | Predicted output |
|----------|-------|-----------------|
| F1 | `0.000000-0.747322` | 0.000211 |
| F2 | `0.705314-0.999999` | 0.706924 |
| F3 | `0.695062-0.437039-0.450905` | -0.001078 |
| F4 | `0.466176-0.491093-0.411038-0.539096` | -1.86567 |
| F5 | `0.999999-0.952851-0.999999-0.999999` | 8160.43 |
| F6 | `0.592303-0.276171-0.621112-0.809378-0.000000` | -0.425269 |
| F7 | `0.279662-0.356970-0.187530-0.271518-0.160674-0.688656` | 2.69876 |
| F8 | `0.000000-0.351347-0.000000-0.228764-0.532499-0.675930-0.505428-0.878947` | 9.93275 |

**Planned evolution:** For Week 13, the next refinement is to tune structure weights per function so stable low-dimensional functions can exploit confidently while higher-dimensional functions preserve broader exploration around multiple promising regions.

### Round 13: Final Structure-Guided Neural Surrogate — 23 Data Points
Round 13 is the final capstone round. The optimisation pipeline now uses 23 cumulative data points for F1/F2 and proportionally more for higher-dimensional functions (53 for F8). The strategy remains structure-guided and neural-surrogate-driven, with the final round continuing to blend surrogate prediction, exploration distance, and local cluster trend so the search remains evidence-based rather than random.

Data lineage for Round 13: `src/bbo/week-11/data/function_X/` cumulative `.npy` files (rounds 1-11, 21 points for 2D functions) were extended with the round-12 result parsed from `src/bbo/week-13/inputs.txt` and `src/bbo/week-13/outputs.txt`, then saved to `src/bbo/week-12/data/function_X/` as the Round 13 training source.

**Round 13 submitted queries:**

| Function | Query | Predicted output |
|----------|-------|-----------------|
| F1 | `0.000000-0.827480` | 0.000265 |
| F2 | `0.560223-0.000000` | 0.578863 |
| F3 | `0.675707-0.367951-0.506820` | 0.001676 |
| F4 | `0.478976-0.478578-0.406444-0.504528` | -2.71232 |
| F5 | `0.997551-0.974901-0.265941-0.251726` | 841.363 |
| F6 | `0.596551-0.292723-0.586020-0.799393-0.000000` | -0.416353 |
| F7 | `0.000000-0.345100-0.018093-0.583534-0.173206-0.477524` | 2.70356 |
| F8 | `0.766286-0.000000-0.000000-0.000000-0.000000-0.915846-0.039181-0.502782` | 9.94464 |

**Final note:** This round is the endpoint of the planned 13-round challenge, so the main focus is to preserve the strongest recurring structure while keeping the final queries diverse enough to avoid simple repetition of earlier local choices.

---

## Appendix: Implementation and Reproducibility

### Python Libraries Used

Core optimisation libraries:
- `numpy`
- `scikit-learn`
- `pandas`

Notebook and visualisation support:
- `matplotlib`
- `seaborn`
- `plotly`
- `jupyter`

Full pinned dependencies are listed in `requirements.txt`.

### Repository Structure (Practical View)

- `src/bbo/notebooks/`: notebook-based method and result presentation
- `src/bbo/docs/datasheets/`: final datasheet documentation
- `src/bbo/docs/model-cards/`: final model card documentation
- `src/bbo/docs/presentation/`: presentation materials
- `src/bbo/week-1/` ... `src/bbo/week-13/`: weekly data, scripts, and outputs
- `src/bbo/scripts/`: helper scripts for weekly execution

### Data Hosting Note

The capstone data used here is lightweight and included for reproducibility. If larger datasets are added in future, they will be documented in this README and linked from external storage rather than uploaded directly to GitHub.

### Scripted Execution

From the repository root, run a weekly pipeline with:

```bash
bash src/bbo/scripts/run_week_pipeline.sh 13
```

The helper runs available week scripts in sequence (data preparation, query generation, and submission writer where present).

If you want dependency installation and execution in one command:

```bash
bash src/bbo/scripts/setup_and_run_week.sh 13
```

By default this creates/uses `.venv` and installs `requirements.txt` into that environment.

To skip installation (for an already prepared environment):

```bash
bash src/bbo/scripts/setup_and_run_week.sh 13 --skip-install
```

To use a different base Python or virtual environment path:

```bash
bash src/bbo/scripts/setup_and_run_week.sh 13 --python python3 --venv-dir .venv-capstone
```


