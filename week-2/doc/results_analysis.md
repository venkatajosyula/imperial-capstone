# Week 2 Results Analysis

## Results

| Function | Submitted Query | Returned Output | Best to Date | Model Note | Observation |
|----------|----------------|-----------------|-------------|------------|-------------|
| F1 (2D) | 0.701073-0.786107 | -2.40e-21 | -2.40e-21 | Heuristic only | Two consecutive near-zero returns; region is unresponsive |
| F2 (2D) | 0.700892-0.769633 | 0.6416 | 0.6416 | Heuristic only | Consistent improvement; exceeds seed max (0.611) |
| F3 (3D) | 0.497633-0.614279-0.258288 | -0.1064 | -0.0348 | Heuristic only | Slight regression; landscape appears shallow |
| F4 (4D) | 0.521076-0.439371-0.434167-0.217233 | -3.5878 | -3.5878 | Heuristic only | Improving trajectory across two rounds |
| F5 (4D) | 0.210813-0.847360-0.819576-0.874716 | 836.08 | 1088.86 | Heuristic only | Strong; still below seed max but confirms high-output region |
| F6 (5D) | 0.700843-0.227182-0.562522-0.721366-0.129918 | -0.5228 | -0.5228 | Heuristic only | Matches seed max; marginal improvement from W1 |
| F7 (6D) | 0.293991-0.383786-0.323080-0.280630-0.352965-0.718040 | 2.4063 | 2.5703 | Heuristic only | Slight regression from W1 high; near local maximum |
| F8 (8D) | 0.120069-0.226268-0.145005-0.203587-0.504352-0.726636-0.409460-0.715138 | 9.8057 | 9.8197 | Heuristic only | Essentially stable; approaching regional ceiling |

## Reflection

F2 and F4 both exceeded the seed maximum for the first time, confirming that incorporating evaluation feedback before re-ranking produces directional improvement. F1 remains unresponsive across two rounds and should be treated differently in Week 3. F7 and F8 appear near local maxima. The module context (linear/logistic regression) is most directly relevant for F2, whose smooth 2D landscape is amenable to a linear approximation, and F8, where fitting a linear model with 8 predictors on 12 observations immediately violates standard sample-size requirements.
