# Week 8 NN Strategy Diagnostics

## Function 1
- Data source: week-7/data/function_1 (inputs.npy, outputs.npy)
- Samples: 17 | Dimension: 2
- Suggested query: 0.999999-0.773695
- NN predicted output at query: 0.0002821986749
- Distance from nearest observed point: 0.057008
- Steepest gradient dimensions (1-based): 2, 1
- Gradient magnitudes on those dims: 0.000335075, 0.000112154
- NN error diagnostics (lower is better): train_mse=5.37896e-10, cv_mse=2.46886e-06

## Function 2
- Data source: week-7/data/function_2 (inputs.npy, outputs.npy)
- Samples: 17 | Dimension: 2
- Suggested query: 0.650633-0.637990
- NN predicted output at query: 0.686180304
- Distance from nearest observed point: 0.067140
- Steepest gradient dimensions (1-based): 1, 2
- Gradient magnitudes on those dims: 3.14214, 0.00968574
- NN error diagnostics (lower is better): train_mse=0.000585885, cv_mse=0.119709

## Function 3
- Data source: week-7/data/function_3 (inputs.npy, outputs.npy)
- Samples: 22 | Dimension: 3
- Suggested query: 0.990819-0.999999-0.000000
- NN predicted output at query: 0.02472639519
- Distance from nearest observed point: 0.482267
- Steepest gradient dimensions (1-based): 3, 2, 1
- Gradient magnitudes on those dims: 0.0201579, 0.0159268, 9.71445e-14
- NN error diagnostics (lower is better): train_mse=0.000221372, cv_mse=0.0160442

## Function 4
- Data source: week-7/data/function_4 (inputs.npy, outputs.npy)
- Samples: 37 | Dimension: 4
- Suggested query: 0.577435-0.496284-0.366459-0.204129
- NN predicted output at query: -3.927147657
- Distance from nearest observed point: 0.062729
- Steepest gradient dimensions (1-based): 2, 4, 1
- Gradient magnitudes on those dims: 6.02381, 2.4568, 1.69495
- NN error diagnostics (lower is better): train_mse=0.0802328, cv_mse=40.9888

## Function 5
- Data source: week-7/data/function_5 (inputs.npy, outputs.npy)
- Samples: 27 | Dimension: 4
- Suggested query: 0.439929-0.956670-0.999999-0.999999
- NN predicted output at query: 3354.48446
- Distance from nearest observed point: 0.082021
- Steepest gradient dimensions (1-based): 4, 3, 1
- Gradient magnitudes on those dims: 956.043, 927.539, 2.27374e-10
- NN error diagnostics (lower is better): train_mse=1955.23, cv_mse=146922

## Function 6
- Data source: week-7/data/function_6 (inputs.npy, outputs.npy)
- Samples: 27 | Dimension: 5
- Suggested query: 0.651139-0.262567-0.599355-0.798422-0.091835
- NN predicted output at query: -0.5426353495
- Distance from nearest observed point: 0.108774
- Steepest gradient dimensions (1-based): 2, 1, 4
- Gradient magnitudes on those dims: 0.172522, 0.143118, 0.12338
- NN error diagnostics (lower is better): train_mse=0.000909215, cv_mse=0.148005

## Function 7
- Data source: week-7/data/function_7 (inputs.npy, outputs.npy)
- Samples: 37 | Dimension: 6
- Suggested query: 0.307563-0.319620-0.265684-0.216323-0.357029-0.533625
- NN predicted output at query: 2.834719373
- Distance from nearest observed point: 0.202315
- Steepest gradient dimensions (1-based): 3, 2, 5
- Gradient magnitudes on those dims: 3.23809, 2.0353, 1.17893
- NN error diagnostics (lower is better): train_mse=0.000142574, cv_mse=0.591063

## Function 8
- Data source: week-7/data/function_8 (inputs.npy, outputs.npy)
- Samples: 47 | Dimension: 8
- Suggested query: 0.130847-0.000000-0.000000-0.350108-0.451876-0.074684-0.555836-0.938167
- NN predicted output at query: 10.15238457
- Distance from nearest observed point: 0.598994
- Steepest gradient dimensions (1-based): 3, 2, 6
- Gradient magnitudes on those dims: 0.9859, 0.189529, 0.0134225
- NN error diagnostics (lower is better): train_mse=7.73203e-05, cv_mse=0.29482
