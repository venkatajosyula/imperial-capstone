# Week 3 Results Analysis

## Results

| Function | Submitted Query | Returned Output | Best to Date | Model Note | Observation |
|----------|----------------|-----------------|-------------|------------|-------------|
| F1 (2D) | 0.701073-0.786107 | Pending | -2.40e-21 | SVM alt: 0.075948-0.561940 | Heuristic stalled; SVM explores a new region |
| F2 (2D) | 0.702236-0.806848 | Pending | 0.6416 | SVM alt: 0.803481-0.792750 | Consistent improvement trend before Week 3 submission |
| F3 (3D) | 0.497633-0.614279-0.258288 | Pending | -0.0348 | SVM alt: 0.638409-0.748591-0.252167 | Unchanged heuristic query; weak directional signal |
| F4 (4D) | 0.533245-0.437211-0.432699-0.224122 | Pending | -3.5878 | SVM alt: 0.224407-0.392157-0.296398-0.408290 | Heuristic shifts toward current best |
| F5 (4D) | 0.219194-0.846502-0.849414-0.877066 | Pending | 1088.86 | SVM alt: 0.723745-0.066398-0.862547-0.835529 | SVM proposes opposite dim-2 region |
| F6 (5D) | 0.703577-0.225575-0.572782-0.720991-0.123648 | Pending | -0.5228 | SVM alt: 0.524346-0.572718-0.456764-0.840844-0.155797 | Stable negative region; little heuristic movement |
| F7 (6D) | 0.282444-0.380143-0.329072-0.279157-0.348277-0.714192 | Pending | 2.5703 | SVM alt: 0.360362-0.210653-0.396583-0.260497-0.064207-0.633819 | Near local maximum from Week 1 |
| F8 (8D) | 0.125536-0.217890-0.142241-0.207853-0.495262-0.718959-0.407012-0.721511 | Pending | 9.8197 | SVM alt: 0.095842-0.233274-0.144613-0.339838-0.344193-0.373939-0.465824-0.436333 | Stable region; refinement step |

## Reflection

The SVM and heuristic diverged most strongly for F1 (entirely different region) and F5 (opposite dim-2, 0.066 vs 0.847). With 12 points, the RBF boundary is too sparse in 4D–8D to be trusted over the heuristic, but F1 is the one case where SVM exploration is justified — the heuristic has failed to find any non-zero signal across three rounds. Key limitations at this stage: the blend is non-adaptive (no uncertainty estimate), assumes uniform feature relevance, and cannot detect local maxima. A surrogate model would provide more principled guidance as data grows.
