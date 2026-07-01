# Week 11 Section A: Clustering Lens and Model Card Check

## 1) Real-life applications and limitations of clustering in this BBO context

In this round, I used clustering as a practical way to group similar historical query points and observe local behavior around those groups. This is similar to real-world optimisation settings where teams segment operating conditions before selecting the next experiment, for example in manufacturing parameter tuning or hyperparameter search.

How clustering was useful here:
- It gave me local neighborhoods instead of treating all historical points as one cloud.
- It gave a distance cue (distance to nearest centroid) to describe whether a new query stayed near a known region or moved away.
- It provided a simple local trend signal (mean output per cluster) that helped prioritise clusters with better historical behavior.

Main limitations I observed:
- Cluster quality depends on sparse data coverage, especially in high dimensions.
- Cluster assignments can shift when only one new point is added, so conclusions are not fully stable.
- Good cluster averages do not guarantee local smoothness or generalisation.
- A centroid can represent geometry well but still miss sharp local peaks.

So, clustering helped with structured exploration, but it is not a guarantee of finding the true best region.

## 2) Model card transparency and potential bias check

I reviewed the model card at `docs/model-cards/bbo_optimisation_strategy_model_card.md` for transparency and potential bias.

What is transparent:
- The strategy evolution is clearly documented (heuristic -> SVM -> neural surrogate).
- Intended uses and out-of-scope uses are stated.
- Assumptions and failure modes are explicit.
- Reproducibility factors (scripts, lineage, fixed seeds) are described.

Potential bias or risk areas:
- Search bias: repeated optimisation can over-focus on already promising zones, reducing exploration of unexplored regions.
- Data sparsity bias: higher-dimensional functions have weaker coverage, which can make model confidence uneven.
- Model-form bias: using one surrogate family for all functions may fit some functions better than others.

How I accounted for this in Week 11:
- I kept an exploration term based on distance from observed points.
- I added a clustering cue as an additional local structure signal, not a single decision rule.
- I reported diagnostics and assumptions in the Week 11 justification file for auditability.

Overall, the model card remains useful for decision transparency, but I should extend it in later rounds with stronger uncertainty reporting and clearer per-function robustness notes.
