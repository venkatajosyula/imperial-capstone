# BBO Capstone Dataset Datasheet

## Motivation: why did you create this data set? What tasks does it support?

I created this data set to support my Black-Box Optimisation capstone workflow. The goal is to choose better queries each week for eight unknown objective functions.

The data set supports sequential optimisation under a strict budget of one query per function per round. I use it to train surrogate models and to compare exploration and exploitation choices over time.

## Composition: what does it contain? What is the size and format and are there any gaps?

The data set contains function-wise query histories and returned objective values.

- Functions: F1 to F8
- Input dimensions:
  - F1, F2: 2
  - F3: 3
  - F4, F5: 4
  - F6: 5
  - F7: 6
  - F8: 8
- Current cumulative size after Round 10:
  - F1: 19 rows
  - F2: 19 rows
  - F3: 24 rows
  - F4: 39 rows
  - F5: 29 rows
  - F6: 29 rows
  - F7: 39 rows
  - F8: 49 rows

Format:
- Raw weekly files: `inputs.txt`, `outputs.txt`
- Processed files: NumPy arrays
  - `inputs.npy` with shape `(n_samples, d)`
  - `outputs.npy` with shape `(n_samples,)`

Gaps:
The data set is complete for observed rounds but sparse relative to the full search space, especially for high-dimensional functions. This means many regions are still unexplored.

## Collection process: how were the queries generated? What strategy did you use? Over what time frame?

Queries were generated once per week and submitted through the capstone portal. Returned outputs were copied into weekly files and appended into cumulative arrays.

Strategy by period:
- Rounds 1 to 2: weighted blend heuristics
- Round 3: heuristic plus SVM experimentation
- Rounds 4 to 10: neural network surrogate with candidate sampling and gradient-style refinement

Time frame:
The full capstone horizon is 13 weekly rounds. This datasheet is a Week 10 snapshot and currently documents collected data through Round 10.

## Preprocessing and uses: have you applied any transformations? What are the intended and inappropriate uses?

Applied transformations:
- Parsed portal text into numeric arrays
- Joined wrapped lines where needed
- Extracted scalar outputs from portal format
- Appended only the newest row each round
- Validated dimension count, six-decimal schema, and [0, 0.999999] bounds

Intended uses:
- Train and update surrogate models
- Generate next-round candidate queries
- Provide reproducible optimisation trace and diagnostics

Inappropriate uses:
- Safety-critical decisions
- Claims of guaranteed global optimum
- Unvalidated transfer to unrelated optimisation problems

## Distribution and maintenance: where is the data set available? What are the terms of use? Who maintains it?

Availability:
The data set is in this repository under weekly folders.
- Raw weekly records: `week-N/inputs.txt`, `week-N/outputs.txt`
- Processed cumulative arrays: `week-N/data/function_X/inputs.npy`, `outputs.npy`

Terms of use:
This is an academic capstone data set for learning, reproducibility, and peer review. External reuse should include attribution and clear caveats about sparse coverage and uncertainty.

Maintenance:
- Maintained by: repository owner
- Update schedule: once per weekly challenge round

## Reflection: how did writing this datasheet affect how you describe your data, decisions, and assumptions?

Writing this datasheet made my assumptions more explicit. The main assumption is that local surrogate patterns from sparse observations are useful for choosing future queries. I now describe this as a practical but limited assumption, especially in high-dimensional functions. The datasheet also improved clarity in how I explain data lineage, preprocessing steps, and responsible use boundaries.
