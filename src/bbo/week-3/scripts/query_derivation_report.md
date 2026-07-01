# Week 3 Query Derivation Report

## Method

Two parallel generators run this week:
1. **Heuristic (submitted):** 60/30/10 blend on 12-point cumulative datasets (seed + W1 + W2 evaluation results).
2. **SVM (Module 14.1):** RBF soft-margin SVC (C=1.0) fitted on binary labels (above/below 50th-percentile), scoring 10,000 random candidates by decision function.

Heuristic preferred for submission at 12 points: SVM boundary is too sparse to trust over objectively observed high-output regions.

## Submitted Queries (Heuristic) vs SVM

| Function | Heuristic (submitted) | SVM query | W2 Result | Divergence |
|----------|-----------------------|-----------|-----------|------------|
| F1 | 0.701073-0.786107 | 0.075948-0.561940 | -2.40e-21 | Large — SVM explores new region (heuristic stalled) |
| F2 | 0.702236-0.806848 | 0.803481-0.792750 | 0.6416 | Small — SVM shifts dim-1 higher |
| F3 | 0.497633-0.614279-0.258288 | 0.638409-0.748591-0.252167 | -0.1064 | Moderate — SVM raises dim-1/2 |
| F4 | 0.533245-0.437211-0.432699-0.224122 | 0.224407-0.392157-0.296398-0.408290 | -3.5878 | Large — SVM inverts dim-1 direction |
| F5 | 0.219194-0.846502-0.849414-0.877066 | 0.723745-0.066398-0.862547-0.835529 | 836.08 | Large — SVM proposes opposite dim-2 region |
| F6 | 0.703577-0.225575-0.572782-0.720991-0.123648 | 0.524346-0.572718-0.456764-0.840844-0.155797 | -0.5228 | Large — SVM raises dim-2 and dim-4 |
| F7 | 0.282444-0.380143-0.329072-0.279157-0.348277-0.714192 | 0.360362-0.210653-0.396583-0.260497-0.064207-0.633819 | 2.4063 | Large — SVM lowers dim-2/5/6 |
| F8 | 0.125536-0.217890-0.142241-0.207853-0.495262-0.718959-0.407012-0.721511 | 0.095842-0.233274-0.144613-0.339838-0.344193-0.373939-0.465824-0.436333 | 9.8057 | Moderate — SVM reduces dim-6/8 |

## Week 2 evaluation system Outputs (Basis for Week 3 Strategy)

| Function | Week 2 Query | evaluation system Result | vs Week 1 | Assessment |
|----------|-------------|---------------|-----------|------------|
| F1 (2D) | 0.701073-0.786107 | -2.40e-21 | No change | Persistently near-zero; two rounds with identical result |
| F2 (2D) | 0.700892-0.769633 | 0.6416 | +16.2% | Continued improvement; approaching apparent regional maximum |
| F3 (3D) | 0.497633-0.614279-0.258288 | -0.1064 | −8.0% | Slight regression; function landscape appears shallow and noisy |
| F4 (4D) | 0.521076-0.439371-0.434167-0.217233 | -3.5878 | +12.8% | Improving trajectory; all outputs remain negative |
| F5 (4D) | 0.210813-0.847360-0.819576-0.874716 | 836.08 | +27.6% | Best single result across all functions and rounds |
| F6 (5D) | 0.700843-0.227182-0.562522-0.721366-0.129918 | -0.5228 | +19.5% | Marginal improvement; region remains below zero |
| F7 (6D) | 0.293991-0.383786-0.323080-0.280630-0.352965-0.718040 | 2.4063 | −6.4% | Slight regression from Week 1 high of 2.570 |
| F8 (8D) | 0.120069-0.226268-0.145005-0.203587-0.504352-0.726636-0.409460-0.715138 | 9.8057 | −0.1% | Essentially stable; near apparent regional peak |

## Per-Function Notes

### Function 1
- **Heuristic query:** 0.701073-0.786107
- **SVM query:** 0.075948-0.561940
- **Data source:** src/bbo/week-3/data/function_1 (12 points)
- **Reasoning:** Two rounds of -2.40e-21 suggest the visited region produces no meaningful signal. The heuristic is forced to return here because all 12 observed outputs are near-zero. The SVM query (0.076-0.562) proposes a completely different region — the classifier has labelled the top-6 observations as "high" relative to the bottom-6, even though the absolute values are all near zero, and identified a different area of the unit square as the likely boundary. This function is a strong candidate to switch to the SVM query in a future round.

### Function 2
- **Heuristic query:** 0.702236-0.806848
- **SVM query:** 0.803481-0.792750
- **Data source:** src/bbo/week-3/data/function_2 (12 points)
- **Reasoning:** Consistent improvement across two rounds (0.552 → 0.642). The heuristic moves dim-2 slightly higher (0.770 → 0.807). The SVM query moves dim-1 higher (0.703 → 0.803), reflecting an alternative view of the high-performance region boundary.

### Function 3
- **Heuristic query:** 0.497633-0.614279-0.258288
- **SVM query:** 0.638409-0.748591-0.252167
- **Data source:** src/bbo/week-3/data/function_3 (12 points)
- **Reasoning:** The slight regression in Week 2 (-0.1064 vs -0.0985) did not change the ranking, so the heuristic query is identical to Weeks 1 and 2. The SVM suggests a higher dim-1 and dim-2 with similar dim-3, proposing an unexplored sub-region of the cube.

### Function 4
- **Heuristic query:** 0.533245-0.437211-0.432699-0.224122
- **SVM query:** 0.224407-0.392157-0.296398-0.408290
- **Data source:** src/bbo/week-3/data/function_4 (12 points)
- **Reasoning:** Week 2 result (-3.588) is now the best observed, so it enters the blend at 60% weight, shifting the query (dim-1: 0.521 → 0.533). The SVM proposes a notably different region with low dim-1 and higher dim-4.

### Function 5
- **Heuristic query:** 0.219194-0.846502-0.849414-0.877066
- **SVM query:** 0.723745-0.066398-0.862547-0.835529
- **Data source:** src/bbo/week-3/data/function_5 (12 points)
- **Reasoning:** Best observed result (836.08 at Week 2) is now ranked second behind the initial maximum (1088.86). Heuristic stays in the high dim-2/dim-4 region. The SVM proposes a striking alternative: low dim-2 (0.066), very different from all historical high-output points, suggesting the boundary classifier found a different competitive region.

### Function 6
- **Heuristic query:** 0.703577-0.225575-0.572782-0.720991-0.123648
- **SVM query:** 0.524346-0.572718-0.456764-0.840844-0.155797
- **Data source:** src/bbo/week-3/data/function_6 (12 points)
- **Reasoning:** Steady marginal improvement across two rounds. Heuristic adjusts dim-3 slightly upward (0.563 → 0.573). SVM proposes a substantially different configuration with higher dim-2 and much higher dim-4 (0.841).

### Function 7
- **Heuristic query:** 0.282444-0.380143-0.329072-0.279157-0.348277-0.714192
- **SVM query:** 0.360362-0.210653-0.396583-0.260497-0.064207-0.633819
- **Data source:** src/bbo/week-3/data/function_7 (12 points)
- **Reasoning:** Week 1 (2.570) remains the best result; Week 2 declined slightly to 2.406. The heuristic adjusts toward Week 1 coordinates. The SVM proposes substantially lower dim-2 and dim-5, with dim-6 reduced (0.714 → 0.634).

### Function 8
- **Heuristic query:** 0.125536-0.217890-0.142241-0.207853-0.495262-0.718959-0.407012-0.721511
- **SVM query:** 0.095842-0.233274-0.144613-0.339838-0.344193-0.373939-0.465824-0.436333
- **Data source:** src/bbo/week-3/data/function_8 (12 points)
- **Reasoning:** Week 2 (9.806) and Week 1 (9.820) are both near the initial best (9.820). The function appears near a local maximum in the visited region. The heuristic refines local coordinates; the SVM proposes lower dim-6 and dim-8, suggesting the boundary classifier sees a different region as competitive.
