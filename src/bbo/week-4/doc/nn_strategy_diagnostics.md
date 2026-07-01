# Week 4 NN Strategy Diagnostics

## Function 1
- Data source: src/bbo/week-3/data/function_1 (inputs.npy, outputs.npy)
- Samples: 12 | Dimension: 2
- Suggested query: 0.999999-0.940024
- NN predicted output at query: 0.001069176394
- Distance from nearest observed point: 0.326281
- Steepest gradient dimensions (1-based): 2, 1
- Gradient magnitudes on those dims: 0.00261777, 0.00187562
- NN error diagnostics (lower is better): train_mse=3.2682e-10, cv_mse=4.8771e-06

## Function 2
- Data source: src/bbo/week-3/data/function_2 (inputs.npy, outputs.npy)
- Samples: 12 | Dimension: 2
- Suggested query: 0.825049-0.999999
- NN predicted output at query: 0.8866159581
- Distance from nearest observed point: 0.142749
- Steepest gradient dimensions (1-based): 2, 1
- Gradient magnitudes on those dims: 0.469133, 0.0773974
- NN error diagnostics (lower is better): train_mse=2.31136e-06, cv_mse=0.0423769

## Function 3
- Data source: src/bbo/week-3/data/function_3 (inputs.npy, outputs.npy)
- Samples: 17 | Dimension: 3
- Suggested query: 0.421806-0.806832-0.000000
- NN predicted output at query: 0.006753829024
- Distance from nearest observed point: 0.206954
- Steepest gradient dimensions (1-based): 3, 1, 2
- Gradient magnitudes on those dims: 0.234766, 4.16334e-14, 0
- NN error diagnostics (lower is better): train_mse=0.000243634, cv_mse=0.0115644

## Function 4
- Data source: src/bbo/week-3/data/function_4 (inputs.npy, outputs.npy)
- Samples: 32 | Dimension: 4
- Suggested query: 0.576927-0.504025-0.384598-0.323654
- NN predicted output at query: -2.964363716
- Distance from nearest observed point: 0.113735
- Steepest gradient dimensions (1-based): 3, 1, 2
- Gradient magnitudes on those dims: 7.60824, 2.66456, 2.25235
- NN error diagnostics (lower is better): train_mse=0.063455, cv_mse=32.4974

## Function 5
- Data source: src/bbo/week-3/data/function_5 (inputs.npy, outputs.npy)
- Samples: 22 | Dimension: 4
- Suggested query: 0.311936-0.777428-0.999999-0.999999
- NN predicted output at query: 1324.253165
- Distance from nearest observed point: 0.204328
- Steepest gradient dimensions (1-based): 4, 3, 1
- Gradient magnitudes on those dims: 241.99, 234.05, 0.000118375
- NN error diagnostics (lower is better): train_mse=193.008, cv_mse=17215.5

## Function 6
- Data source: src/bbo/week-3/data/function_6 (inputs.npy, outputs.npy)
- Samples: 22 | Dimension: 5
- Suggested query: 0.338046-0.000000-0.659400-0.952214-0.000000
- NN predicted output at query: -0.272886496
- Distance from nearest observed point: 0.501346
- Steepest gradient dimensions (1-based): 2, 1, 5
- Gradient magnitudes on those dims: 0.0471866, 0.0422863, 0.0301449
- NN error diagnostics (lower is better): train_mse=0.000431702, cv_mse=0.147925

## Function 7
- Data source: src/bbo/week-3/data/function_7 (inputs.npy, outputs.npy)
- Samples: 32 | Dimension: 6
- Suggested query: 0.000000-0.046064-0.406126-0.000000-0.196875-0.999999
- NN predicted output at query: 3.149348301
- Distance from nearest observed point: 0.622229
- Steepest gradient dimensions (1-based): 3, 5, 6
- Gradient magnitudes on those dims: 0.58987, 0.539477, 0.438355
- NN error diagnostics (lower is better): train_mse=1.17554e-05, cv_mse=0.484704

## Function 8
- Data source: src/bbo/week-3/data/function_8 (inputs.npy, outputs.npy)
- Samples: 42 | Dimension: 8
- Suggested query: 0.000000-0.260373-0.000000-0.460567-0.561143-0.621366-0.548445-0.596871
- NN predicted output at query: 10.5793926
- Distance from nearest observed point: 0.372833
- Steepest gradient dimensions (1-based): 3, 1, 5
- Gradient magnitudes on those dims: 1.34523, 0.395356, 0.0228998
- NN error diagnostics (lower is better): train_mse=1.60497e-05, cv_mse=0.241812
