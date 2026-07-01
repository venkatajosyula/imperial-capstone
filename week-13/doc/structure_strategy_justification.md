# Week 13 Structure-Guided Query Justification

This note records the targeted local structure per function,
the distance/similarity cues used, and how Week 13 choices
build on the first twelve rounds.

## Function 1
- Data source: week-12/data/function_1 (inputs.npy, outputs.npy)
- Samples: 24 | Dimension: 2
- Suggested query: 0.000000-0.815572
- Predicted output at query: 0.0001618812698
- Local cluster targeted: cluster 2 of 3 (mean historical output=2.26457e-80)
- Similarity cue used: nearest-centroid distance=0.105311; nearest-observed-point distance=0.068250
- Structure cue: normalized cluster quality=1.0000; score blended surrogate prediction + exploration distance + local cluster trend
- Build on first twelve rounds: query uses cumulative history and targets recurring strong regions while keeping the search space diversified
- Surrogate diagnostics: train_mse=1.38894e-09, cv_mse=6.45619e-07

## Function 2
- Data source: week-12/data/function_2 (inputs.npy, outputs.npy)
- Samples: 24 | Dimension: 2
- Suggested query: 0.865359-0.664718
- Predicted output at query: 0.3109816417
- Local cluster targeted: cluster 1 of 3 (mean historical output=0.357363)
- Similarity cue used: nearest-centroid distance=0.223257; nearest-observed-point distance=0.050562
- Structure cue: normalized cluster quality=1.0000; score blended surrogate prediction + exploration distance + local cluster trend
- Build on first twelve rounds: query uses cumulative history and targets recurring strong regions while keeping the search space diversified
- Surrogate diagnostics: train_mse=0.00171872, cv_mse=0.0439838

## Function 3
- Data source: week-12/data/function_3 (inputs.npy, outputs.npy)
- Samples: 29 | Dimension: 3
- Suggested query: 0.689939-0.354692-0.519848
- Predicted output at query: 0.002551484148
- Local cluster targeted: cluster 2 of 3 (mean historical output=-0.0853833)
- Similarity cue used: nearest-centroid distance=0.155455; nearest-observed-point distance=0.107519
- Structure cue: normalized cluster quality=1.0000; score blended surrogate prediction + exploration distance + local cluster trend
- Build on first twelve rounds: query uses cumulative history and targets recurring strong regions while keeping the search space diversified
- Surrogate diagnostics: train_mse=6.97974e-05, cv_mse=0.0127955

## Function 4
- Data source: week-12/data/function_4 (inputs.npy, outputs.npy)
- Samples: 44 | Dimension: 4
- Suggested query: 0.482750-0.470761-0.406235-0.489839
- Predicted output at query: -2.884917174
- Local cluster targeted: cluster 0 of 4 (mean historical output=-7.468)
- Similarity cue used: nearest-centroid distance=0.213667; nearest-observed-point distance=0.049646
- Structure cue: normalized cluster quality=1.0000; score blended surrogate prediction + exploration distance + local cluster trend
- Build on first twelve rounds: query uses cumulative history and targets recurring strong regions while keeping the search space diversified
- Surrogate diagnostics: train_mse=0.191911, cv_mse=26.5014

## Function 5
- Data source: week-12/data/function_5 (inputs.npy, outputs.npy)
- Samples: 34 | Dimension: 4
- Suggested query: 0.997551-0.974901-0.265941-0.251726
- Predicted output at query: 915.6114392
- Local cluster targeted: cluster 2 of 4 (mean historical output=149.66)
- Similarity cue used: nearest-centroid distance=0.791656; nearest-observed-point distance=0.576425
- Structure cue: normalized cluster quality=0.0144; score blended surrogate prediction + exploration distance + local cluster trend
- Build on first twelve rounds: query uses cumulative history and targets recurring strong regions while keeping the search space diversified
- Surrogate diagnostics: train_mse=33545.3, cv_mse=118748

## Function 6
- Data source: week-12/data/function_6 (inputs.npy, outputs.npy)
- Samples: 34 | Dimension: 5
- Suggested query: 0.596824-0.313887-0.575189-0.821951-0.000000
- Predicted output at query: -0.4191510005
- Local cluster targeted: cluster 3 of 4 (mean historical output=-0.64749)
- Similarity cue used: nearest-centroid distance=0.140430; nearest-observed-point distance=0.060909
- Structure cue: normalized cluster quality=1.0000; score blended surrogate prediction + exploration distance + local cluster trend
- Build on first twelve rounds: query uses cumulative history and targets recurring strong regions while keeping the search space diversified
- Surrogate diagnostics: train_mse=0.000870501, cv_mse=0.119552

## Function 7
- Data source: week-12/data/function_7 (inputs.npy, outputs.npy)
- Samples: 44 | Dimension: 6
- Suggested query: 0.000000-0.351977-0.017611-0.548162-0.170167-0.459973
- Predicted output at query: 2.770370895
- Local cluster targeted: cluster 1 of 4 (mean historical output=1.65382)
- Similarity cue used: nearest-centroid distance=0.556226; nearest-observed-point distance=0.485825
- Structure cue: normalized cluster quality=1.0000; score blended surrogate prediction + exploration distance + local cluster trend
- Build on first twelve rounds: query uses cumulative history and targets recurring strong regions while keeping the search space diversified
- Surrogate diagnostics: train_mse=0.000494276, cv_mse=0.462929

## Function 8
- Data source: week-12/data/function_8 (inputs.npy, outputs.npy)
- Samples: 54 | Dimension: 8
- Suggested query: 0.803674-0.000000-0.000000-0.000000-0.000000-0.932867-0.021660-0.491959
- Predicted output at query: 9.975643371
- Local cluster targeted: cluster 1 of 5 (mean historical output=9.53074)
- Similarity cue used: nearest-centroid distance=1.097492; nearest-observed-point distance=0.948324
- Structure cue: normalized cluster quality=1.0000; score blended surrogate prediction + exploration distance + local cluster trend
- Build on first twelve rounds: query uses cumulative history and targets recurring strong regions while keeping the search space diversified
- Surrogate diagnostics: train_mse=0.00111851, cv_mse=0.289698
