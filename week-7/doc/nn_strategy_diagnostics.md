# Week 7 NN Strategy Diagnostics

## Function 1
- Data source: week-6/data/function_1 (inputs.npy, outputs.npy)
- Samples: 16 | Dimension: 2
- Suggested query: 0.000000-0.997451
- NN predicted output at query: 0.0002323856671
- Distance from nearest observed point: 0.396239
- Steepest gradient dimensions (1-based): 1, 2
- Gradient magnitudes on those dims: 0.000105702, 6.48234e-05
- NN error diagnostics (lower is better): train_mse=1.24103e-09, cv_mse=3.37089e-06

## Function 2
- Data source: week-6/data/function_2 (inputs.npy, outputs.npy)
- Samples: 16 | Dimension: 2
- Suggested query: 0.632322-0.999999
- NN predicted output at query: 0.6245018148
- Distance from nearest observed point: 0.101670
- Steepest gradient dimensions (1-based): 1, 2
- Gradient magnitudes on those dims: 4.77849, 0.406046
- NN error diagnostics (lower is better): train_mse=0.00135242, cv_mse=0.108944

## Function 3
- Data source: week-6/data/function_3 (inputs.npy, outputs.npy)
- Samples: 21 | Dimension: 3
- Suggested query: 0.440601-0.635553-0.618984
- NN predicted output at query: 0.02043826056
- Distance from nearest observed point: 0.284622
- Steepest gradient dimensions (1-based): 1, 2, 3
- Gradient magnitudes on those dims: 0.209648, 0.149929, 0.0833458
- NN error diagnostics (lower is better): train_mse=0.000127246, cv_mse=0.0156301

## Function 4
- Data source: week-6/data/function_4 (inputs.npy, outputs.npy)
- Samples: 36 | Dimension: 4
- Suggested query: 0.552008-0.484773-0.419586-0.254089
- NN predicted output at query: -4.007839985
- Distance from nearest observed point: 0.042906
- Steepest gradient dimensions (1-based): 3, 2, 4
- Gradient magnitudes on those dims: 3.68474, 2.47171, 1.63697
- NN error diagnostics (lower is better): train_mse=0.0653235, cv_mse=20.9636

## Function 5
- Data source: week-6/data/function_5 (inputs.npy, outputs.npy)
- Samples: 26 | Dimension: 4
- Suggested query: 0.388378-0.892873-0.999999-0.999999
- NN predicted output at query: 2885.302992
- Distance from nearest observed point: 0.077393
- Steepest gradient dimensions (1-based): 4, 3, 2
- Gradient magnitudes on those dims: 1368.47, 938.421, 0
- NN error diagnostics (lower is better): train_mse=639.426, cv_mse=44826.3

## Function 6
- Data source: week-6/data/function_6 (inputs.npy, outputs.npy)
- Samples: 26 | Dimension: 5
- Suggested query: 0.596418-0.290466-0.535810-0.999999-0.260912
- NN predicted output at query: -0.4424949038
- Distance from nearest observed point: 0.176483
- Steepest gradient dimensions (1-based): 1, 2, 5
- Gradient magnitudes on those dims: 0.152041, 0.11072, 0.0681063
- NN error diagnostics (lower is better): train_mse=0.000477838, cv_mse=0.120602

## Function 7
- Data source: week-6/data/function_7 (inputs.npy, outputs.npy)
- Samples: 36 | Dimension: 6
- Suggested query: 0.000000-0.314012-0.431251-0.161961-0.434593-0.698083
- NN predicted output at query: 2.75110529
- Distance from nearest observed point: 0.216535
- Steepest gradient dimensions (1-based): 3, 5, 4
- Gradient magnitudes on those dims: 1.1072, 0.947641, 0.480979
- NN error diagnostics (lower is better): train_mse=0.000293181, cv_mse=0.554489

## Function 8
- Data source: week-6/data/function_8 (inputs.npy, outputs.npy)
- Samples: 46 | Dimension: 8
- Suggested query: 0.000000-0.446318-0.000000-0.000000-0.546491-0.000000-0.595139-0.999999
- NN predicted output at query: 10.1764104
- Distance from nearest observed point: 0.673768
- Steepest gradient dimensions (1-based): 3, 1, 4
- Gradient magnitudes on those dims: 0.665807, 0.449999, 0.151783
- NN error diagnostics (lower is better): train_mse=4.68976e-05, cv_mse=0.258061
