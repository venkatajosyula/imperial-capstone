# Week 8 Reflection

## What Changed This Week

The biggest change was moving from narrative-heavy notes to a tighter execution workflow.
I kept the process simple: generate candidates, score them with the surrogate, refine the
best starts, then validate format and bounds before submission. This made Week 8 faster
to run and easier to audit.

## Reliability vs Exploration

The submitted queries came from deterministic scripts, so reproducibility improved.
Exploration still came from two places in the code path:

- random candidate sampling (5,000 candidates)
- gradient-style refinement from top-ranked candidates

This gave enough variation to avoid repeating the same region blindly, while keeping
output formatting stable for submission.

## Failure Checks I Used

I explicitly checked for the common failure modes before writing final outputs:

- exact number of dimensions per function
- numeric bounds inside [0, 0.999999]
- six-decimal formatting for every coordinate

If any line failed validation, it was rejected immediately. That guardrail reduced
last-minute submission risk.

## Limits At 17 Data Points

At this stage, gains were uneven across functions. Lower-dimensional functions were
more stable, but higher-dimensional ones were still sensitive to small modeling errors.
Adding more descriptive commentary did not improve outcomes; better data and better
function-specific tuning mattered more.

## What I Would Do Next

For later rounds, I would keep this workflow but tune it per function instead of using
one global setting. In particular, I would adjust exploration weight and regularisation
based on each function's observed variance, while keeping the same strict validation
before accepting any candidate.

## Practitioner Takeaway

Week 8 reinforced a practical lesson: in black-box optimisation, disciplined process
beats flashy one-off ideas. Make assumptions explicit, validate outputs aggressively,
and keep the pipeline reproducible so each round is comparable to the last.
