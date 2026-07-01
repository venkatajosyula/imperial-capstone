# Week 2 Query Derivation Report

## Method

Same 60/30/10 blend, applied to 11-point cumulative datasets (seed + Week 1 evaluation system result).
Change from Week 1: Week 1 evaluated points incorporated before ranking, so the blend now draws on a real observation.
Approach: exploitation-dominant; one query/week budget makes broad exploration costly.

## Submitted Queries

| Function | Dims | Submitted Query | W1 evaluation system Result | Strategy |
|----------|------|----------------|-----------------|----------|
| F1 | 2D | 0.701073-0.786107 | -2.40e-21 | W1 = near-zero; heuristic unchanged, no signal to redirect |
| F2 | 2D | 0.700892-0.769633 | 0.5520 | W1 ranked 2nd; blend shifts closer to seed best (0.611) |
| F3 | 3D | 0.497633-0.614279-0.258288 | -0.0985 | W1 below seed best; ranking unchanged, query identical |
| F4 | 4D | 0.521076-0.439371-0.434167-0.217233 | -4.1161 | W1 ranked; dim-1 shifted higher toward seed best |
| F5 | 4D | 0.210813-0.847360-0.819576-0.874716 | 655.21 | W1 ranked 2nd; retaining high dim-2/dim-4 region |
| F6 | 5D | 0.700843-0.227182-0.562522-0.721366-0.129918 | -0.6498 | W1 was worst result; blend reweights to better seed points |
| F7 | 6D | 0.293991-0.383786-0.323080-0.280630-0.352965-0.718040 | 2.5703 | W1 = new best; blend centres on W1 coordinates |
| F8 | 8D | 0.120069-0.226268-0.145005-0.203587-0.504352-0.726636-0.409460-0.715138 | 9.8197 | W1 = new best; blend centres on W1 coordinates |

## Week 1 evaluation system Outputs (Basis for Week 2 Strategy)

| Function | Week 1 Query | evaluation system Result | Assessment |
|----------|-------------|---------------|------------|
| F1 (2D) | 0.701073-0.786107 | -2.40e-21 ≈ 0 | Near-zero; function may be essentially flat in this region |
| F2 (2D) | 0.709101-0.670992 | 0.5520 | Reasonable; blend pulled slightly below historical best (0.611) |
| F3 (3D) | 0.497633-0.614279-0.258288 | -0.0985 | Negative; all outputs negative so this is relative improvement |
| F4 (4D) | 0.472696-0.449572-0.444506-0.190801 | -4.1161 | Negative; marginally worse than historical best (-4.026) |
| F5 (4D) | 0.214371-0.844060-0.758507-0.875420 | 655.21 | Strong; confirms high-value region around dim-2 near 0.84 |
| F6 (5D) | 0.700843-0.245990-0.540046-0.729239-0.133524 | -0.6498 | Negative; blend moved away from best region |
| F7 (6D) | 0.314096-0.359733-0.345676-0.288593-0.333908-0.709472 | 2.5703 | Positive; above best initial value |
| F8 (8D) | 0.139785-0.239023-0.160745-0.238163-0.505938-0.701437-0.392239-0.696102 | 9.8197 | Strong; above best initial value (9.598) |

## Per-Function Notes

### Function 1
- **Query:** 0.701073-0.786107
- **Data source:** src/bbo/week-2/data/function_1 (11 points)
- **Reasoning:** The Week 1 output was -2.40e-21, which is effectively zero. The three highest-ranked observations in the initial dataset also had near-zero outputs (7.71e-16, 2.54e-40, 1.03e-46). Because all known outputs are indistinguishably close to zero, the weighted blend returns to the same region. No meaningful gradient signal exists to justify moving elsewhere.

### Function 2
- **Query:** 0.700892-0.769633
- **Data source:** src/bbo/week-2/data/function_2 (11 points)
- **Reasoning:** Week 1 returned 0.552 against a historical best of 0.611. The Week 1 evaluated point became the second-best in the expanded dataset, so the 60/30/10 blend shifted the query slightly toward the historical best (row 9: 0.703, 0.927) and away from the evaluated point, targeting a region slightly closer to the original maximum.

### Function 3
- **Query:** 0.497633-0.614279-0.258288
- **Data source:** src/bbo/week-2/data/function_3 (11 points)
- **Reasoning:** Because all outputs for Function 3 are negative and closely clustered, the Week 1 evaluated point (-0.0985) ranked below the top historical point (-0.0348). The blend does not shift meaningfully so the query is identical to Week 1. The function appears to have a shallow landscape with limited directional signal.

### Function 4
- **Query:** 0.521076-0.439371-0.434167-0.217233
- **Data source:** src/bbo/week-2/data/function_4 (11 points)
- **Reasoning:** Week 1 returned -4.116, which ranked below the historical best (-4.026). After incorporating this result, the best historical point re-ranked first and the blend shifted closer to it, moving dim-1 higher (0.472 → 0.521) and reducing weight on the evaluated point. The query attempts to approach the apparent local maximum in this predominantly negative function.

### Function 5
- **Query:** 0.210813-0.847360-0.819576-0.874716
- **Data source:** src/bbo/week-2/data/function_5 (11 points)
- **Reasoning:** Week 1 returned 655.21, which ranked second behind the historical maximum (1088.86 from the initial dataset). The blend stays strongly exploitation-focused in this high-value region. Dim-2 and Dim-4 remain high (~0.847 and ~0.875), consistent with observed correlation between these dimensions and strong outputs.

### Function 6
- **Query:** 0.700843-0.227182-0.562522-0.721366-0.129918
- **Data source:** src/bbo/week-2/data/function_6 (11 points)
- **Reasoning:** Week 1 output was -0.650, the worst in the expanded dataset. The blend weights shifted to favour historical observations less negative than the Week 1 result, moving dim-2 slightly lower and dim-3 slightly higher. This represents constrained exploitation within a region where all outputs are negative.

### Function 7
- **Query:** 0.293991-0.383786-0.323080-0.280630-0.352965-0.718040
- **Data source:** src/bbo/week-2/data/function_7 (11 points)
- **Reasoning:** Week 1 returned 2.5703, a new high above the best initial value (1.365). After re-ranking with the evaluated point at top, the blend shifts to place more weight on that point's coordinates. The query moves dim-6 slightly down (0.709 → 0.718 is actually up) and rebalances other dimensions around the new best.

### Function 8
- **Query:** 0.120069-0.226268-0.145005-0.203587-0.504352-0.726636-0.409460-0.715138
- **Data source:** src/bbo/week-2/data/function_8 (11 points)
- **Reasoning:** Week 1 returned 9.8197, a new high above initial best (9.598). The Week 1 evaluated point became the top-ranked observation. The blend now centres on those coordinates, reducing dim-1 (0.140 → 0.120) and dim-5 (0.506 → 0.504) marginally while staying close to the known high-value region in this 8D space.
