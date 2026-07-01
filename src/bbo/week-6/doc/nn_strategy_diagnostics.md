# Week 6 NN Strategy Diagnostics

## Function 1
- Data source: src/bbo/week-5/data/function_1 (inputs.npy, outputs.npy)
- Samples: 15 | Dimension: 2
- Suggested query: 0.000000-0.062404
- NN predicted output at query: 0.0003415789669
- Distance from nearest observed point: 0.313117
- Steepest gradient dimensions (1-based): 1, 2
- Gradient magnitudes on those dims: 0.000493917, 1.6263e-16
- NN error diagnostics (lower is better): train_mse=8.16287e-10, cv_mse=2.72504e-06

## Function 2
- Data source: src/bbo/week-5/data/function_2 (inputs.npy, outputs.npy)
- Samples: 15 | Dimension: 2
- Suggested query: 0.954448-0.000000
- NN predicted output at query: 0.6276844965
- Distance from nearest observed point: 0.314144
- Steepest gradient dimensions (1-based): 2, 1
- Gradient magnitudes on those dims: 0.0415697, 0.00301547
- NN error diagnostics (lower is better): train_mse=0.00127825, cv_mse=0.0956876

## Function 3
- Data source: src/bbo/week-5/data/function_3 (inputs.npy, outputs.npy)
- Samples: 20 | Dimension: 3
- Suggested query: 0.591243-0.000000-0.703210
- NN predicted output at query: 0.08865033114
- Distance from nearest observed point: 0.455593
- Steepest gradient dimensions (1-based): 2, 3, 1
- Gradient magnitudes on those dims: 0.133875, 7.41734e-06, 1.09575e-06
- NN error diagnostics (lower is better): train_mse=1.55766e-05, cv_mse=0.0161495

## Function 4
- Data source: src/bbo/week-5/data/function_4 (inputs.npy, outputs.npy)
- Samples: 35 | Dimension: 4
- Suggested query: 0.577846-0.442866-0.377803-0.239652
- NN predicted output at query: -3.802528361
- Distance from nearest observed point: 0.031956
- Steepest gradient dimensions (1-based): 2, 4, 1
- Gradient magnitudes on those dims: 3.34335, 1.4809, 0.949733
- NN error diagnostics (lower is better): train_mse=0.0592903, cv_mse=28.2111

## Function 5
- Data source: src/bbo/week-5/data/function_5 (inputs.npy, outputs.npy)
- Samples: 25 | Dimension: 4
- Suggested query: 0.347755-0.826999-0.999999-0.999999
- NN predicted output at query: 2468.208608
- Distance from nearest observed point: 0.061158
- Steepest gradient dimensions (1-based): 4, 3, 1
- Gradient magnitudes on those dims: 1690.25, 1275.17, 6.82121e-09
- NN error diagnostics (lower is better): train_mse=687.188, cv_mse=61634.5

## Function 6
- Data source: src/bbo/week-5/data/function_6 (inputs.npy, outputs.npy)
- Samples: 25 | Dimension: 5
- Suggested query: 0.627991-0.307931-0.538595-0.855392-0.166437
- NN predicted output at query: -0.5449709232
- Distance from nearest observed point: 0.161692
- Steepest gradient dimensions (1-based): 2, 1, 4
- Gradient magnitudes on those dims: 0.1873, 0.140778, 0.0976449
- NN error diagnostics (lower is better): train_mse=0.000506216, cv_mse=0.12227

## Function 7
- Data source: src/bbo/week-5/data/function_7 (inputs.npy, outputs.npy)
- Samples: 35 | Dimension: 6
- Suggested query: 0.175912-0.292417-0.399772-0.190708-0.383330-0.593055
- NN predicted output at query: 2.670016294
- Distance from nearest observed point: 0.218563
- Steepest gradient dimensions (1-based): 3, 5, 4
- Gradient magnitudes on those dims: 3.46539, 2.73783, 2.04547
- NN error diagnostics (lower is better): train_mse=2.86763e-05, cv_mse=0.561973

## Function 8
- Data source: src/bbo/week-5/data/function_8 (inputs.npy, outputs.npy)
- Samples: 45 | Dimension: 8
- Suggested query: 0.000000-0.092187-0.000000-0.313546-0.351845-0.999999-0.074649-0.367430
- NN predicted output at query: 10.16734616
- Distance from nearest observed point: 0.628009
- Steepest gradient dimensions (1-based): 3, 6, 1
- Gradient magnitudes on those dims: 0.623454, 0.239238, 0.169184
- NN error diagnostics (lower is better): train_mse=2.2107e-05, cv_mse=0.281197
