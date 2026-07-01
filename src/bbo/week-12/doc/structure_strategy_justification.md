# Week 12 Structure-Guided Query Justification

This note records the targeted local structure per function,
the distance/similarity cues used, and how Week 12 choices
build on the first eleven rounds.

## Function 1
- Data source: src/bbo/week-11/data/function_1 (inputs.npy, outputs.npy)
- Samples: 21 | Dimension: 2
- Suggested query: 0.000000-0.747322
- Predicted output at query: 0.000211321101
- Local cluster targeted: cluster 2 of 3 (mean historical output=-5.39812e-55)
- Similarity cue used: nearest-centroid distance=0.608574; nearest-observed-point distance=0.250129
- Structure cue: normalized cluster quality=1.0000; score blended surrogate prediction + exploration distance + local cluster trend
- Build on first eleven rounds: query uses cumulative history and targets recurring strong regions while controlling randomness through distance-aware exploration
- Surrogate diagnostics: train_mse=1.84704e-09, cv_mse=1.51384e-06

## Function 2
- Data source: src/bbo/week-11/data/function_2 (inputs.npy, outputs.npy)
- Samples: 21 | Dimension: 2
- Suggested query: 0.705314-0.999999
- Predicted output at query: 0.7069242389
- Local cluster targeted: cluster 0 of 3 (mean historical output=0.321926)
- Similarity cue used: nearest-centroid distance=0.203159; nearest-observed-point distance=0.072992
- Structure cue: normalized cluster quality=1.0000; score blended surrogate prediction + exploration distance + local cluster trend
- Build on first eleven rounds: query uses cumulative history and targets recurring strong regions while controlling randomness through distance-aware exploration
- Surrogate diagnostics: train_mse=0.000186285, cv_mse=0.0626511

## Function 3
- Data source: src/bbo/week-11/data/function_3 (inputs.npy, outputs.npy)
- Samples: 26 | Dimension: 3
- Suggested query: 0.695062-0.437039-0.450905
- Predicted output at query: -0.001078179229
- Local cluster targeted: cluster 1 of 3 (mean historical output=-0.0848116)
- Similarity cue used: nearest-centroid distance=0.333361; nearest-observed-point distance=0.049450
- Structure cue: normalized cluster quality=1.0000; score blended surrogate prediction + exploration distance + local cluster trend
- Build on first eleven rounds: query uses cumulative history and targets recurring strong regions while controlling randomness through distance-aware exploration
- Surrogate diagnostics: train_mse=4.11345e-05, cv_mse=0.0142874

## Function 4
- Data source: src/bbo/week-11/data/function_4 (inputs.npy, outputs.npy)
- Samples: 41 | Dimension: 4
- Suggested query: 0.466176-0.491093-0.411038-0.539096
- Predicted output at query: -1.86567211
- Local cluster targeted: cluster 3 of 4 (mean historical output=-8.07213)
- Similarity cue used: nearest-centroid distance=0.295101; nearest-observed-point distance=0.068070
- Structure cue: normalized cluster quality=1.0000; score blended surrogate prediction + exploration distance + local cluster trend
- Build on first eleven rounds: query uses cumulative history and targets recurring strong regions while controlling randomness through distance-aware exploration
- Surrogate diagnostics: train_mse=0.134385, cv_mse=26.6766

## Function 5
- Data source: src/bbo/week-11/data/function_5 (inputs.npy, outputs.npy)
- Samples: 31 | Dimension: 4
- Suggested query: 0.999999-0.952851-0.999999-0.999999
- Predicted output at query: 8160.434982
- Local cluster targeted: cluster 1 of 3 (mean historical output=2928.63)
- Similarity cue used: nearest-centroid distance=0.613816; nearest-observed-point distance=0.047148
- Structure cue: normalized cluster quality=1.0000; score blended surrogate prediction + exploration distance + local cluster trend
- Build on first eleven rounds: query uses cumulative history and targets recurring strong regions while controlling randomness through distance-aware exploration
- Surrogate diagnostics: train_mse=26614.1, cv_mse=399924

## Function 6
- Data source: src/bbo/week-11/data/function_6 (inputs.npy, outputs.npy)
- Samples: 31 | Dimension: 5
- Suggested query: 0.592303-0.276171-0.621112-0.809378-0.000000
- Predicted output at query: -0.4252690625
- Local cluster targeted: cluster 1 of 3 (mean historical output=-0.730982)
- Similarity cue used: nearest-centroid distance=0.137551; nearest-observed-point distance=0.035201
- Structure cue: normalized cluster quality=1.0000; score blended surrogate prediction + exploration distance + local cluster trend
- Build on first eleven rounds: query uses cumulative history and targets recurring strong regions while controlling randomness through distance-aware exploration
- Surrogate diagnostics: train_mse=0.000923163, cv_mse=0.0953461

## Function 7
- Data source: src/bbo/week-11/data/function_7 (inputs.npy, outputs.npy)
- Samples: 41 | Dimension: 6
- Suggested query: 0.279662-0.356970-0.187530-0.271518-0.160674-0.688656
- Predicted output at query: 2.698762868
- Local cluster targeted: cluster 2 of 4 (mean historical output=1.64002)
- Similarity cue used: nearest-centroid distance=0.251177; nearest-observed-point distance=0.141531
- Structure cue: normalized cluster quality=1.0000; score blended surrogate prediction + exploration distance + local cluster trend
- Build on first eleven rounds: query uses cumulative history and targets recurring strong regions while controlling randomness through distance-aware exploration
- Surrogate diagnostics: train_mse=0.000452417, cv_mse=0.54231

## Function 8
- Data source: src/bbo/week-11/data/function_8 (inputs.npy, outputs.npy)
- Samples: 51 | Dimension: 8
- Suggested query: 0.000000-0.351347-0.000000-0.228764-0.532499-0.675930-0.505428-0.878947
- Predicted output at query: 9.932749518
- Local cluster targeted: cluster 2 of 5 (mean historical output=9.55028)
- Similarity cue used: nearest-centroid distance=0.362281; nearest-observed-point distance=0.301874
- Structure cue: normalized cluster quality=1.0000; score blended surrogate prediction + exploration distance + local cluster trend
- Build on first eleven rounds: query uses cumulative history and targets recurring strong regions while controlling randomness through distance-aware exploration
- Surrogate diagnostics: train_mse=0.000104553, cv_mse=0.342367
