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
### batch mode for throughput (input/output in mkldnn layout)
./run.sh --inference --mkldnn
### batch mode for throughput (input/output in mkldnn layout; jitted)
./run.sh --inference --mkldnn --cache-weight

### single batch for latency (input/output/weight in native layout)
./run.sh --inference --single
### single batch for latency (input/output in mkldnn layout)
./run.sh --inference --single --mkldnn
### single batch for latency (input/output in mkldnn layout; jitted)
./run.sh --inference --single --mkldnn --jit
```
## performance
1. inference with single batch:
```bash
./run.sh --inference --single
### inference only
### using single batch size
### using OMP_NUM_THREADS=28
### using KMP_AFFINITY=granularity=fine,compact,1,0
### using KMP_BLOCKTIME=1
### using numactl --physcpubind=0-27 --membind=0

Running on device:  Intel(R) Xeon(R) Platinum 8280 CPU @ 2.70GHz
Running on torch: 1.7.0a0+7cc6540
Running on torchvision: 0.7.0+cpu

ModelType: resnext101, Kernels: nn Input shape: 1x3x224x224
nn                              :forward:      54.51 (ms)      18.35 (imgs/s)
nn                             :backward:       0.00 (ms)
nn                               :update:       0.00 (ms)
nn                                :total:      54.51 (ms)      18.35 (imgs/s)
```

2. inference with single batch (input and output in mkldnn layout; weight uncached)
```bash
./run.sh --inference --single --mkldnn
### inference only
### using single batch size
### cache input/output in mkldnn format
### using OMP_NUM_THREADS=28
### using KMP_AFFINITY=granularity=fine,compact,1,0
### using KMP_BLOCKTIME=1
### using numactl --physcpubind=0-27 --membind=0

Running on device:  Intel(R) Xeon(R) Platinum 8280 CPU @ 2.70GHz
Running on torch: 1.7.0a0+7cc6540
Running on torchvision: 0.7.0+cpu

ModelType: resnext101, Kernels: nn Input shape: 1x3x224x224
nn                              :forward:      38.19 (ms)      26.18 (imgs/s)
nn                             :backward:       0.00 (ms)
nn                               :update:       0.00 (ms)
nn                                :total:      38.19 (ms)      26.18 (imgs/s)
```

3. inference with single batch (input and output in mkldnn layout; weight cached)
```bash
### inference only
### using single batch size
### cache input/output in mkldnn format
### jitted in mkldnn format
### using OMP_NUM_THREADS=28
### using KMP_AFFINITY=granularity=fine,compact,1,0
### using KMP_BLOCKTIME=1
### using numactl --physcpubind=0-27 --membind=0

Running on device:  Intel(R) Xeon(R) Platinum 8280 CPU @ 2.70GHz
Running on torch: 1.7.0a0+7cc6540
Running on torchvision: 0.7.0+cpu

ModelType: resnext101, Kernels: nn Input shape: 1x3x224x224
### load script module from resnext101.script.pt, weight reordered in mkldnn format
nn                              :forward:      31.61 (ms)      31.64 (imgs/s)
nn                             :backward:       0.00 (ms)
nn                               :update:       0.00 (ms)
nn                                :total:      31.61 (ms)      31.64 (imgs/s)
```
