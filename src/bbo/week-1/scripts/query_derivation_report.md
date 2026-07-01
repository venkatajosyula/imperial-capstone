# Week 1 Query Derivation Report

## Method

`query = 0.6 * best_input + 0.3 * second_best_input + 0.1 * third_best_input`, clipped to [0, 0.999999].
Data: 10 seed observations per function. Approach: exploitation-first, no prior evaluation feedback available.

## Submitted Queries

| Function | Dims | Submitted Query | Est. Output Proxy | Strategy |
|----------|------|----------------|-------------------|----------|
| F1 | 2D | 0.701073-0.786107 | ≈ 0 | Blend of near-zero seed outputs; no usable gradient |
| F2 | 2D | 0.709101-0.670992 | 0.5705 | Exploiting top seed region (best: 0.611) |
| F3 | 3D | 0.497633-0.614279-0.258288 | -0.0365 | Least-negative region; all outputs negative |
| F4 | 4D | 0.472696-0.449572-0.444506-0.190801 | -5.2226 | Least-negative region; all outputs negative |
| F5 | 4D | 0.214371-0.844060-0.758507-0.875420 | 893.04 | High dim-2/dim-4 region confirmed by seed |
| F6 | 5D | 0.700843-0.245990-0.540046-0.729239-0.133524 | -0.6871 | Blend of least-negative seed points |
| F7 | 6D | 0.314096-0.359733-0.345676-0.288593-0.333908-0.709472 | 2.019 | Exploiting top seed cluster; high dim-6 |
| F8 | 8D | 0.139785-0.239023-0.160745-0.238163-0.505938-0.701437-0.392239-0.696102 | 9.706 | Exploiting top seed cluster; high dim-6/dim-8 |

## Note

Estimated output proxy is a weighted average of known top-3 outputs — not the true black-box result.
True outputs are returned by the evaluation system after submission.

<!-- original per-function detail removed for brevity; full derivation is reproducible by running src/bbo/week-1/scripts/run_all_queries.py -->

`query = 0.6 * best_input + 0.3 * second_best_input + 0.1 * third_best_input`

After computing the weighted combination, each dimension was clipped to the valid input range `[0.000000, 0.999999]`.

The estimated output shown below is only a proxy based on the weighted top-3 known outputs. It is not the true black-box output, which can only be obtained after submitting the query and receiving the next evaluation.

Note: in the evaluation system submission format, the `-` character is a separator between dimensions, not a negative sign.

## Function 1

- Top-3 source row indices: `[2, 7, 1]`
- Top-3 source input points:
  - `[0.731024, 0.733000]`
  - `[0.683418, 0.861057]`
  - `[0.574329, 0.879898]`
- Top-3 known outputs: `[7.710875114502849e-16, 2.5350011535584046e-40, 1.0330782375230975e-46]`
- Derived query: `0.701073-0.786107`
- Estimated output proxy: `4.62652506870171e-16`
- Explanation: the best observed point dominated the blend because it was much stronger than the rest, while the second and third points nudged the query slightly to avoid repeating an existing sample exactly.

## Function 2

- Top-3 source row indices: `[9, 0, 1]`
- Top-3 source input points:
  - `[0.702637, 0.926564]`
  - `[0.665800, 0.123969]`
  - `[0.877791, 0.778628]`
- Top-3 known outputs: `[0.6112052157614438, 0.5389961189269181, 0.42058623962798264]`
- Derived query: `0.709101-0.670992`
- Estimated output proxy: `0.57048058909774`
- Explanation: this query stays close to the strongest 2D region while averaging across the next best points to reduce the chance of overcommitting to a single local optimum.

## Function 3

- Top-3 source row indices: `[3, 13, 10]`
- Top-3 source input points:
  - `[0.492581, 0.611593, 0.340176]`
  - `[0.600097, 0.725136, 0.066089]`
  - `[0.220549, 0.297825, 0.343555]`
- Top-3 known outputs: `[-0.034835313350078584, -0.036377828071632486, -0.04694740582651916]`
- Derived query: `0.497633-0.614279-0.258288`
- Estimated output proxy: `-0.03650927701418881`
- Explanation: because this function is still a maximization task even with negative values, the query was placed near the least-negative region found so far, with some moderation from the next two best samples.

## Function 4

- Top-3 source row indices: `[27, 24, 23]`
- Top-3 source input points:
  - `[0.577766, 0.428772, 0.425826, 0.249007]`
  - `[0.326076, 0.472367, 0.453192, 0.105887]`
  - `[0.282138, 0.505987, 0.530531, 0.096302]`
- Top-3 known outputs: `[-4.025542281908162, -6.702089254839066, -7.9667753510303925]`
- Derived query: `0.472696-0.449572-0.444506-0.190801`
- Estimated output proxy: `-5.222629680699656`
- Explanation: the query concentrates around the strongest 4D cluster seen so far and slightly smooths across neighboring good points to stay exploratory without moving too far away.

## Function 5

- Top-3 source row indices: `[15, 18, 14]`
- Top-3 source input points:
  - `[0.224189, 0.846480, 0.879484, 0.878516]`
  - `[0.119879, 0.862540, 0.643331, 0.849804]`
  - `[0.438933, 0.774092, 0.378167, 0.933696]`
- Top-3 known outputs: `[1088.8596181962705, 431.6127567592104, 355.8068177560159]`
- Derived query: `0.214371-0.844060-0.758507-0.875420`
- Estimated output proxy: `818.380279721127`
- Explanation: this function showed a very strong best-performing point, so the query leans heavily toward it while using the other two points to keep the proposal within a broader promising region.

## Function 6

- Top-3 source row indices: `[0, 4, 17]`
- Top-3 source input points:
  - `[0.728186, 0.154693, 0.732552, 0.693997, 0.056401]`
  - `[0.618812, 0.331802, 0.187288, 0.756238, 0.328835]`
  - `[0.782880, 0.536336, 0.443284, 0.859700, 0.010326]`
- Top-3 known outputs: `[-0.7142649478202404, -0.8292365522578722, -0.9357565553342914]`
- Derived query: `0.700843-0.245990-0.540046-0.729239-0.133524`
- Estimated output proxy: `-0.770905589902935`
- Explanation: because the best values are still negative, the aim was to stay near the least-negative region and average in nearby candidates that may improve the score toward zero.

## Function 7

- Top-3 source row indices: `[6, 24, 14]`
- Top-3 source input points:
  - `[0.057896, 0.491672, 0.247422, 0.218118, 0.420428, 0.730970]`
  - `[0.881647, 0.204450, 0.414474, 0.420385, 0.264915, 0.730660]`
  - `[0.148647, 0.033943, 0.728806, 0.316066, 0.021769, 0.516918]`
- Top-3 known outputs: `[1.3649683044991994, 0.6751416308956351, 0.6115255284647864]`
- Derived query: `0.314096-0.359733-0.345676-0.288593-0.333908-0.709472`
- Estimated output proxy: `1.0826760248146887`
- Explanation: in the 6D setting, the blend keeps the query anchored to the best observed region while still using two additional strong samples to reduce the risk of chasing one point too aggressively.

## Function 8

- Top-3 source row indices: `[14, 26, 39]`
- Top-3 source input points:
  - `[0.056447, 0.065956, 0.022929, 0.038786, 0.403935, 0.801055, 0.488307, 0.893085]`
  - `[0.192640, 0.630677, 0.416796, 0.490529, 0.796086, 0.654567, 0.276241, 0.295518]`
  - `[0.481245, 0.102461, 0.219486, 0.677322, 0.247509, 0.244341, 0.163825, 0.715962]`
- Top-3 known outputs: `[9.598482002566342, 9.34427428080805, 9.1830052453254]`
- Derived query: `0.139785-0.239023-0.160745-0.238163-0.505938-0.701437-0.392239-0.696102`
- Estimated output proxy: `9.48067201031476`
- Explanation: the 8D search space is sparse, so this query favors stability by staying in the neighborhood of the best known samples instead of making a high-variance exploratory jump.