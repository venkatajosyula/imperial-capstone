# Week 10 NN Strategy Diagnostics

## Function 1
- Data source: week-9/data/function_1 (inputs.npy, outputs.npy)
- Samples: 19 | Dimension: 2
- Suggested query: 0.999999-0.387219
- NN predicted output at query: 0.000526392658
- Distance from nearest observed point: 0.201221
- Steepest gradient dimensions (1-based): 1, 2
- Gradient magnitudes on those dims: 0.00776742, 0.00166427
- NN error diagnostics (lower is better): train_mse=2.01712e-09, cv_mse=1.42584e-06

## Function 2
- Data source: week-9/data/function_2 (inputs.npy, outputs.npy)
- Samples: 19 | Dimension: 2
- Suggested query: 0.778580-0.325437
- NN predicted output at query: 0.2814165032
- Distance from nearest observed point: 0.116210
- Steepest gradient dimensions (1-based): 1, 2
- Gradient magnitudes on those dims: 5.35946, 0.389125
- NN error diagnostics (lower is better): train_mse=0.000715162, cv_mse=0.0920312

## Function 3
- Data source: week-9/data/function_3 (inputs.npy, outputs.npy)
- Samples: 24 | Dimension: 3
- Suggested query: 0.593991-0.466307-0.438282
- NN predicted output at query: -0.003492149998
- Distance from nearest observed point: 0.202526
- Steepest gradient dimensions (1-based): 1, 2, 3
- Gradient magnitudes on those dims: 0.25884, 0.111633, 0.000622979
- NN error diagnostics (lower is better): train_mse=3.3927e-05, cv_mse=0.0138487

## Function 4
- Data source: week-9/data/function_4 (inputs.npy, outputs.npy)
- Samples: 39 | Dimension: 4
- Suggested query: 0.468386-0.475520-0.442150-0.459084
- NN predicted output at query: -2.650781741
- Distance from nearest observed point: 0.110808
- Steepest gradient dimensions (1-based): 3, 2, 1
- Gradient magnitudes on those dims: 8.61086, 5.23179, 1.66532
- NN error diagnostics (lower is better): train_mse=0.0832534, cv_mse=21.0514

## Function 5
- Data source: week-9/data/function_5 (inputs.npy, outputs.npy)
- Samples: 29 | Dimension: 4
- Suggested query: 0.758271-0.999999-0.999999-0.999999
- NN predicted output at query: 4890.548685
- Distance from nearest observed point: 0.188092
- Steepest gradient dimensions (1-based): 4, 2, 3
- Gradient magnitudes on those dims: 985.522, 838.414, 698.831
- NN error diagnostics (lower is better): train_mse=5142.66, cv_mse=212956

## Function 6
- Data source: week-9/data/function_6 (inputs.npy, outputs.npy)
- Samples: 29 | Dimension: 5
- Suggested query: 0.629244-0.246822-0.625220-0.818379-0.040015
- NN predicted output at query: -0.4978657299
- Distance from nearest observed point: 0.066932
- Steepest gradient dimensions (1-based): 1, 2, 4
- Gradient magnitudes on those dims: 0.2137, 0.194694, 0.166629
- NN error diagnostics (lower is better): train_mse=0.00084217, cv_mse=0.1068

## Function 7
- Data source: week-9/data/function_7 (inputs.npy, outputs.npy)
- Samples: 39 | Dimension: 6
- Suggested query: 0.281416-0.288326-0.368833-0.178561-0.459782-0.585532
- NN predicted output at query: 2.617265637
- Distance from nearest observed point: 0.134737
- Steepest gradient dimensions (1-based): 2, 3, 1
- Gradient magnitudes on those dims: 3.18323, 3.00043, 0.878478
- NN error diagnostics (lower is better): train_mse=0.000183463, cv_mse=0.470158

## Function 8
- Data source: week-9/data/function_8 (inputs.npy, outputs.npy)
- Samples: 49 | Dimension: 8
- Suggested query: 0.004538-0.285212-0.000000-0.344580-0.255946-0.954305-0.000000-0.995167
- NN predicted output at query: 10.18693521
- Distance from nearest observed point: 0.641331
- Steepest gradient dimensions (1-based): 3, 7, 5
- Gradient magnitudes on those dims: 0.67605, 0.0482178, 0.0308994
- NN error diagnostics (lower is better): train_mse=4.97944e-05, cv_mse=0.308495
