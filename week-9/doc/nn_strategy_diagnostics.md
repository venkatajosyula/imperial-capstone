# Week 9 NN Strategy Diagnostics

## Function 1
- Data source: week-8/data/function_1 (inputs.npy, outputs.npy)
- Samples: 18 | Dimension: 2
- Suggested query: 0.944470-0.000000
- NN predicted output at query: 0.000201018356
- Distance from nearest observed point: 0.284470
- Steepest gradient dimensions (1-based): 2, 1
- Gradient magnitudes on those dims: 0.000203394, 1.80855e-06
- NN error diagnostics (lower is better): train_mse=1.4402e-09, cv_mse=1.95447e-06

## Function 2
- Data source: week-8/data/function_2 (inputs.npy, outputs.npy)
- Samples: 18 | Dimension: 2
- Suggested query: 0.999999-0.862726
- NN predicted output at query: 0.6856788753
- Distance from nearest observed point: 0.148348
- Steepest gradient dimensions (1-based): 1, 2
- Gradient magnitudes on those dims: 2.13717, 1.8978e-05
- NN error diagnostics (lower is better): train_mse=0.000995534, cv_mse=0.0979147

## Function 3
- Data source: week-8/data/function_3 (inputs.npy, outputs.npy)
- Samples: 23 | Dimension: 3
- Suggested query: 0.585362-0.982031-0.423506
- NN predicted output at query: 0.01538498537
- Distance from nearest observed point: 0.287988
- Steepest gradient dimensions (1-based): 1, 3, 2
- Gradient magnitudes on those dims: 0.0186793, 0.0140673, 0.0048667
- NN error diagnostics (lower is better): train_mse=2.58088e-05, cv_mse=0.0176066

## Function 4
- Data source: week-8/data/function_4 (inputs.npy, outputs.npy)
- Samples: 38 | Dimension: 4
- Suggested query: 0.512202-0.487812-0.471215-0.362323
- NN predicted output at query: -3.703027997
- Distance from nearest observed point: 0.115975
- Steepest gradient dimensions (1-based): 3, 2, 1
- Gradient magnitudes on those dims: 10.9573, 2.21087, 1.51373
- NN error diagnostics (lower is better): train_mse=0.0824693, cv_mse=29.6645

## Function 5
- Data source: week-8/data/function_5 (inputs.npy, outputs.npy)
- Samples: 28 | Dimension: 4
- Suggested query: 0.570179-0.999999-0.999999-0.999999
- NN predicted output at query: 4051.716613
- Distance from nearest observed point: 0.137268
- Steepest gradient dimensions (1-based): 4, 3, 2
- Gradient magnitudes on those dims: 945.491, 833.286, 663.149
- NN error diagnostics (lower is better): train_mse=4728.39, cv_mse=130113

## Function 6
- Data source: week-8/data/function_6 (inputs.npy, outputs.npy)
- Samples: 28 | Dimension: 5
- Suggested query: 0.633574-0.221343-0.676984-0.853845-0.000000
- NN predicted output at query: -0.4764855823
- Distance from nearest observed point: 0.139784
- Steepest gradient dimensions (1-based): 1, 2, 4
- Gradient magnitudes on those dims: 0.269501, 0.261467, 0.234071
- NN error diagnostics (lower is better): train_mse=0.000745387, cv_mse=0.105973

## Function 7
- Data source: week-8/data/function_7 (inputs.npy, outputs.npy)
- Samples: 38 | Dimension: 6
- Suggested query: 0.348781-0.360456-0.389134-0.281641-0.334543-0.598031
- NN predicted output at query: 2.674952508
- Distance from nearest observed point: 0.124740
- Steepest gradient dimensions (1-based): 3, 2, 5
- Gradient magnitudes on those dims: 3.46385, 1.78933, 1.77154
- NN error diagnostics (lower is better): train_mse=6.6306e-05, cv_mse=0.509268

## Function 8
- Data source: week-8/data/function_8 (inputs.npy, outputs.npy)
- Samples: 48 | Dimension: 8
- Suggested query: 0.000000-0.038573-0.000000-0.000000-0.542498-0.960513-0.612752-0.473576
- NN predicted output at query: 10.03297984
- Distance from nearest observed point: 0.492003
- Steepest gradient dimensions (1-based): 1, 3, 4
- Gradient magnitudes on those dims: 0.610798, 0.369937, 0.26236
- NN error diagnostics (lower is better): train_mse=5.92014e-05, cv_mse=0.299684
