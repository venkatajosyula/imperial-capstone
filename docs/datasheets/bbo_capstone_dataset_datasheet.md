# BBO Capstone Dataset Datasheet (Final Submission)

## Motivation

This dataset supports the Black-Box Optimisation (BBO) capstone project, where the goal is to maximise eight unknown objective functions under strict query limits. The dataset exists to support sequential decision-making, surrogate modeling, and transparent reflection on exploration versus exploitation trade-offs.

## Composition

The dataset contains function-wise query histories and returned objective values.

- Functions: F1 to F8
- Input dimensionality:
  - F1, F2: 2D
  - F3: 3D
  - F4, F5: 4D
  - F6: 5D
  - F7: 6D
  - F8: 8D

Format:
- Raw weekly records: `inputs.txt`, `outputs.txt`
- Processed cumulative arrays:
  - `inputs.npy` with shape `(n_samples, d)`
  - `outputs.npy` with shape `(n_samples,)`

Coverage note:
- The dataset is complete for observed rounds but sparse relative to the full search space, especially in higher-dimensional functions.

## Collection Process

Data was collected over 13 weekly rounds.

- One query per function was submitted each round.
- Returned outputs were appended to cumulative arrays.
- Validation checks were applied for dimensions, bounds, and formatting.

Strategy evolution across rounds:
- Rounds 1-2: weighted heuristic blend
- Round 3: heuristic with parallel SVM experimentation
- Rounds 4-13: neural surrogate-guided optimisation with structured exploration

## Preprocessing and Intended Use

Applied transformations:
- Parsed portal input/output text into numeric arrays
- Appended only the newest row each round
- Verified schema and bounds (`[0, 0.999999]`)
- Used scaling for surrogate model stability

Intended uses:
- Training and updating surrogate models
- Reproducible optimisation workflow analysis
- Strategy comparison and reflection

Uses to avoid:
- Safety-critical deployment decisions
- Claims of guaranteed global optimum
- Transfer to unrelated tasks without validation

## Distribution and Maintenance

Location in repository:
- Raw records: `week-N/inputs.txt`, `week-N/outputs.txt`
- Processed arrays: `week-N/data/function_X/inputs.npy`, `outputs.npy`

Maintenance:
- Maintained by repository owner
- Updated each capstone round

Terms:
- Academic and portfolio use with attribution
- Reuse should include caveats about sparse coverage and uncertainty

## Biases, Limitations, and Risks

- High-dimensional functions remain under-sampled.
- Sampling decisions can reinforce early assumptions if exploration is weak.
- Surrogate confidence may not equal true confidence without calibrated uncertainty.

## Reflection

Writing this datasheet improved traceability by making assumptions explicit: local structure can guide useful decisions, but uncertainty remains substantial in sparse regions. It also helped distinguish what this dataset supports (learning and reproducibility) from what it does not support (high-stakes automation without further validation).
