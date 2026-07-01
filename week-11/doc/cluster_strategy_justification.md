# Week 11 Cluster-Aware Query Justification

This note documents the local cluster targeted for each function,
the distance/similarity cues used, and how the Week 11 choice
builds on the first ten rounds.

## Function 1
- Data source: week-10/data/function_1 (inputs.npy, outputs.npy)
- Samples: 20 | Dimension: 2
- Suggested query: 0.999999-0.436755
- Predicted output at query: 0.0002020883572
- Local cluster targeted: cluster 0 of 3 (mean historical output=1.11392e-124)
- Similarity cue used: nearest-centroid distance=0.230863; nearest-observed-point distance=0.049536
- Cluster trend cue: normalized cluster quality=1.0000; score blended surrogate prediction + exploration distance + cluster trend
- Build on first ten rounds: query is selected from cumulative history and steered toward high-performing neighborhoods while preserving distance from exact repeats
- Surrogate diagnostics: train_mse=1.96623e-09, cv_mse=8.75663e-07

## Function 2
- Data source: week-10/data/function_2 (inputs.npy, outputs.npy)
- Samples: 20 | Dimension: 2
- Suggested query: 0.772318-0.750760
- Predicted output at query: 0.5193289007
- Local cluster targeted: cluster 0 of 3 (mean historical output=0.33489)
- Similarity cue used: nearest-centroid distance=0.071794; nearest-observed-point distance=0.073878
- Cluster trend cue: normalized cluster quality=1.0000; score blended surrogate prediction + exploration distance + cluster trend
- Build on first ten rounds: query is selected from cumulative history and steered toward high-performing neighborhoods while preserving distance from exact repeats
- Surrogate diagnostics: train_mse=0.00175318, cv_mse=0.0688433

## Function 3
- Data source: week-10/data/function_3 (inputs.npy, outputs.npy)
- Samples: 25 | Dimension: 3
- Suggested query: 0.693560-0.417123-0.405668
- Predicted output at query: 0.008724593499
- Local cluster targeted: cluster 0 of 3 (mean historical output=-0.088755)
- Similarity cue used: nearest-centroid distance=0.343277; nearest-observed-point distance=0.115744
- Cluster trend cue: normalized cluster quality=1.0000; score blended surrogate prediction + exploration distance + cluster trend
- Build on first ten rounds: query is selected from cumulative history and steered toward high-performing neighborhoods while preserving distance from exact repeats
- Surrogate diagnostics: train_mse=3.24135e-05, cv_mse=0.012108

## Function 4
- Data source: week-10/data/function_4 (inputs.npy, outputs.npy)
- Samples: 40 | Dimension: 4
- Suggested query: 0.440312-0.475570-0.443247-0.487267
- Predicted output at query: -2.161747795
- Local cluster targeted: cluster 1 of 4 (mean historical output=-8.65001)
- Similarity cue used: nearest-centroid distance=0.270144; nearest-observed-point distance=0.039795
- Cluster trend cue: normalized cluster quality=1.0000; score blended surrogate prediction + exploration distance + cluster trend
- Build on first ten rounds: query is selected from cumulative history and steered toward high-performing neighborhoods while preserving distance from exact repeats
- Surrogate diagnostics: train_mse=0.105398, cv_mse=23.881

## Function 5
- Data source: week-10/data/function_5 (inputs.npy, outputs.npy)
- Samples: 30 | Dimension: 4
- Suggested query: 0.999999-0.999999-0.999999-0.999999
- Predicted output at query: 5767.302447
- Local cluster targeted: cluster 1 of 3 (mean historical output=2450.81)
- Similarity cue used: nearest-centroid distance=0.673328; nearest-observed-point distance=0.241728
- Cluster trend cue: normalized cluster quality=1.0000; score blended surrogate prediction + exploration distance + cluster trend
- Build on first ten rounds: query is selected from cumulative history and steered toward high-performing neighborhoods while preserving distance from exact repeats
- Surrogate diagnostics: train_mse=5749.3, cv_mse=80052.2

## Function 6
- Data source: week-10/data/function_6 (inputs.npy, outputs.npy)
- Samples: 30 | Dimension: 5
- Suggested query: 0.594263-0.257321-0.640626-0.831718-0.000000
- Predicted output at query: -0.4418925094
- Local cluster targeted: cluster 1 of 3 (mean historical output=-0.752939)
- Similarity cue used: nearest-centroid distance=0.147736; nearest-observed-point distance=0.057882
- Cluster trend cue: normalized cluster quality=1.0000; score blended surrogate prediction + exploration distance + cluster trend
- Build on first ten rounds: query is selected from cumulative history and steered toward high-performing neighborhoods while preserving distance from exact repeats
- Surrogate diagnostics: train_mse=0.000976158, cv_mse=0.103239

## Function 7
- Data source: week-10/data/function_7 (inputs.npy, outputs.npy)
- Samples: 40 | Dimension: 6
- Suggested query: 0.249603-0.337927-0.252680-0.270248-0.269419-0.636759
- Predicted output at query: 2.65592785
- Local cluster targeted: cluster 2 of 4 (mean historical output=1.57242)
- Similarity cue used: nearest-centroid distance=0.154147; nearest-observed-point distance=0.144880
- Cluster trend cue: normalized cluster quality=1.0000; score blended surrogate prediction + exploration distance + cluster trend
- Build on first ten rounds: query is selected from cumulative history and steered toward high-performing neighborhoods while preserving distance from exact repeats
- Surrogate diagnostics: train_mse=0.000284244, cv_mse=0.663504

## Function 8
- Data source: week-10/data/function_8 (inputs.npy, outputs.npy)
- Samples: 50 | Dimension: 8
- Suggested query: 0.455746-0.229401-0.000000-0.492928-0.998826-0.541156-0.000000-0.799043
- Predicted output at query: 10.04581263
- Local cluster targeted: cluster 2 of 5 (mean historical output=9.51337)
- Similarity cue used: nearest-centroid distance=0.809423; nearest-observed-point distance=0.617688
- Cluster trend cue: normalized cluster quality=1.0000; score blended surrogate prediction + exploration distance + cluster trend
- Build on first ten rounds: query is selected from cumulative history and steered toward high-performing neighborhoods while preserving distance from exact repeats
- Surrogate diagnostics: train_mse=0.000189167, cv_mse=0.379515
