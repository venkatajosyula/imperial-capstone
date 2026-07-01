# Week 1 Results Analysis

## Results

| Function | Submitted Query | Returned Output | Best to Date | Model Note | Observation |
|----------|----------------|-----------------|-------------|------------|-------------|
| F1 (2D) | 0.701073-0.786107 | -2.40e-21 | 7.71e-16 | Heuristic only | Near-zero; entire region appears flat |
| F2 (2D) | 0.709101-0.670992 | 0.5520 | 0.6112 | Heuristic only | Below seed max; blend averaged away from peak |
| F3 (3D) | 0.497633-0.614279-0.258288 | -0.0985 | -0.0348 | Heuristic only | Slight regression; shallow, noisy landscape |
| F4 (4D) | 0.472696-0.449572-0.444506-0.190801 | -4.1161 | -4.0255 | Heuristic only | Marginal regression; all values negative |
| F5 (4D) | 0.214371-0.844060-0.758507-0.875420 | 655.21 | 1088.86 | Heuristic only | Strong result; confirms high-output region |
| F6 (5D) | 0.700843-0.245990-0.540046-0.729239-0.133524 | -0.6498 | -0.5228 | Heuristic only | Worst result; blend moved away from best region |
| F7 (6D) | 0.314096-0.359733-0.345676-0.288593-0.333908-0.709472 | 2.5703 | 2.5703 | Heuristic only | New best; blend found better region |
| F8 (8D) | 0.139785-0.239023-0.160745-0.238163-0.505938-0.701437-0.392239-0.696102 | 9.8197 | 9.8197 | Heuristic only | New best; high dim-6/dim-8 region confirmed |

## Reflection

The heuristic improved on the seed for F7 and F8 (higher-dimensional functions with more uniform seed coverage) but underperformed for all 2D–5D functions, where averaging across top-3 inputs created a systematic pull away from the seed maximum. F1 returned near-zero for the second consecutive observation, suggesting the function is flat or degenerate in this region. F5 and F7/F8 provide the clearest positive signals to exploit in future rounds.
