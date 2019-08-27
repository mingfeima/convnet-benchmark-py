# Convnet Performance Benchmark for PyTorch

## usage
```bash
### the script runs on GPU when GPU is available, otherwise on CPU
### for training performance (batch mode)
./run.sh


### for inference performance
### to achieve best performance on CPU, tensor needs to be stored in mkldnn layout
###   --mkldnn will allow input and output tensor to be in mkldnn layout
###   --cache-weight will allow weight to be cached in mkldnn layout
###   (beneficial for single batch size inference; typically won't help that much for batch mode)

### batch mode for throughput (input/output/weight in native layout)
./run.sh --inference
### batch mode for throughput (input/output in mkldnn layout; weight uncached)
./run.sh --inference --mkldnn
### batch mode for throughput (input/output in mkldnn layout; weight cached)
./run.sh --inference --mkldnn --cache-weight

### single batch for latency (input/output/weight in native layout)
./run.sh --inference --single
### single batch for latency (input/output in mkldnn layout; weight uncached)
./run.sh --inference --single --mkldnn
### single batch for latency (input/output in mkldnn layout; weight cached)
./run.sh --inference --single --mkldnn --cache-weight
```
## performance
1. inference with single batch:
```bash
Running on device:  Intel(R) Xeon(R) Platinum 8180 CPU @ 2.50GHz
Running on torch: 1.2.0
Running on torchvision: 0.4.0

ModelType: resnext101, Kernels: nn Input shape: 1x3x224x224
nn                              :forward:      92.52 (ms)      10.81 (imgs/s)
nn                             :backward:       0.00 (ms)
nn                               :update:       0.00 (ms)
nn                                :total:      92.52 (ms)      10.81 (imgs/s)
```

2. inference with single batch (input and output in mkldnn layout; weight uncached)
```bash
Running on device:  Intel(R) Xeon(R) Platinum 8180 CPU @ 2.50GHz
Running on torch: 1.2.0
Running on torchvision: 0.4.0

ModelType: resnext101, Kernels: nn Input shape: 1x3x224x224
nn                              :forward:      43.27 (ms)      23.11 (imgs/s)
nn                             :backward:       0.00 (ms)
nn                               :update:       0.00 (ms)
nn                                :total:      43.27 (ms)      23.11 (imgs/s)
```

3. inference with single batch (input and output in mkldnn layout; weight cached)
```bash
Running on device:  Intel(R) Xeon(R) Platinum 8180 CPU @ 2.50GHz
Running on torch: 1.2.0
Running on torchvision: 0.4.0

ModelType: resnext101, Kernels: nn Input shape: 1x3x224x224
### load script module from resnext101.script.pt, weight reordered in mkldnn format
nn                              :forward:      32.35 (ms)      30.91 (imgs/s)
nn                             :backward:       0.00 (ms)
nn                               :update:       0.00 (ms)
nn                                :total:      32.35 (ms)      30.91 (imgs/s)
```
