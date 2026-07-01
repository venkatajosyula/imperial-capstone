# Week 5 NN Strategy Diagnostics

## Function 1
- Data source: src/bbo/week-4/data/function_1 (inputs.npy, outputs.npy)
- Samples: 14 | Dimension: 2
- Suggested query: 0.999999-0.716687
- NN predicted output at query: 0.001147379667
- Distance from nearest observed point: 0.177633
- Steepest gradient dimensions (1-based): 2, 1
- Gradient magnitudes on those dims: 0.00137307, 0.000570307
- NN error diagnostics (lower is better): train_mse=7.83416e-10, cv_mse=3.48588e-06

## Function 2
- Data source: src/bbo/week-4/data/function_2 (inputs.npy, outputs.npy)
- Samples: 14 | Dimension: 2
- Suggested query: 0.670109-0.283738
- NN predicted output at query: 0.6419891862
- Distance from nearest observed point: 0.159827
- Steepest gradient dimensions (1-based): 1, 2
- Gradient magnitudes on those dims: 4.71902, 0.609949
- NN error diagnostics (lower is better): train_mse=0.00152477, cv_mse=0.110715

## Function 3
- Data source: src/bbo/week-4/data/function_3 (inputs.npy, outputs.npy)
- Samples: 19 | Dimension: 3
- Suggested query: 0.999999-0.668856-0.999999
- NN predicted output at query: 0.07069557831
- Distance from nearest observed point: 0.453797
- Steepest gradient dimensions (1-based): 1, 2, 3
- Gradient magnitudes on those dims: 0.0910655, 0.073416, 0.0509881
- NN error diagnostics (lower is better): train_mse=4.54905e-06, cv_mse=0.0123665

## Function 4
- Data source: src/bbo/week-4/data/function_4 (inputs.npy, outputs.npy)
- Samples: 34 | Dimension: 4
- Suggested query: 0.553885-0.455830-0.393990-0.235527
- NN predicted output at query: -3.605000245
- Distance from nearest observed point: 0.049001
- Steepest gradient dimensions (1-based): 3, 2, 1
- Gradient magnitudes on those dims: 3.06056, 1.2952, 0.779016
- NN error diagnostics (lower is better): train_mse=0.041812, cv_mse=32.4481

## Function 5
- Data source: src/bbo/week-4/data/function_5 (inputs.npy, outputs.npy)
- Samples: 24 | Dimension: 4
- Suggested query: 0.375803-0.678335-0.999999-0.999999
- NN predicted output at query: 2628.234931
- Distance from nearest observed point: 0.117891
- Steepest gradient dimensions (1-based): 4, 3, 2
- Gradient magnitudes on those dims: 1027.83, 927.53, 2.58751e-07
- NN error diagnostics (lower is better): train_mse=149.92, cv_mse=117767

## Function 6
- Data source: src/bbo/week-4/data/function_6 (inputs.npy, outputs.npy)
- Samples: 24 | Dimension: 5
- Suggested query: 0.584100-0.000000-0.532219-0.673288-0.094311
- NN predicted output at query: -0.5138895178
- Distance from nearest observed point: 0.264080
- Steepest gradient dimensions (1-based): 1, 5, 4
- Gradient magnitudes on those dims: 0.0570368, 0.0119646, 0.0117166
- NN error diagnostics (lower is better): train_mse=0.000461833, cv_mse=0.121361

## Function 7
- Data source: src/bbo/week-4/data/function_7 (inputs.npy, outputs.npy)
- Samples: 34 | Dimension: 6
- Suggested query: 0.353550-0.351452-0.319921-0.000000-0.324359-0.931710
- NN predicted output at query: 2.886911657
- Distance from nearest observed point: 0.360318
- Steepest gradient dimensions (1-based): 3, 5, 2
- Gradient magnitudes on those dims: 0.629562, 0.555689, 0.398836
- NN error diagnostics (lower is better): train_mse=6.31654e-05, cv_mse=0.681292

## Function 8
- Data source: src/bbo/week-4/data/function_8 (inputs.npy, outputs.npy)
- Samples: 44 | Dimension: 8
- Suggested query: 0.265410-0.000000-0.000000-0.374231-0.968181-0.081225-0.253762-0.840590
- NN predicted output at query: 10.19526554
- Distance from nearest observed point: 0.810514
- Steepest gradient dimensions (1-based): 3, 2, 5
- Gradient magnitudes on those dims: 0.77605, 0.144657, 0.0208033
- NN error diagnostics (lower is better): train_mse=3.55528e-05, cv_mse=0.270567
