# Convnet Performance Benchmark for PyTorch

## usage
```bash
### the script runs on GPU when GPU is available, otherwise on CPU
### for training performance (batch mode)
./run.sh


### for inference performance
### you may choose to use 3 types of memory formats on PyTorch CPU:
### 1. NCHW memory format
###   ./run_inference.sh
###
### 2. NHWC memory format
###   ./run_inference.sh --channels_last
###
### 3. MKLDNN blocked memory format (with weight prepacking)
###   ./run_inference.sh --mkldnn
```
## performance
Results on Intel(R) Xeon(R) Gold 6248 CPU @ 2.50GHz, single soskcet 20 cores, [jemalloc](https://github.com/jemalloc/jemalloc/wiki/Getting-Started) enabled.
* **NCHW (org)**: PyTorch 1.8.1+cpu
* **NHWC (opt)**: branch [test_channels_last_with_bfloat16_support](https://github.com/mingfeima/pytorch/tree/test_channels_last_with_bfloat16_support), upstreaming to public pytorch is ongoing, for more details on channels last plan, please refer to [PyTorch Channels Last Memory Format Performance Optimization on CPU Path](https://gist.github.com/mingfeima/595f63e5dd2ac6f87fdb47df4ffe4772).


Unit: (imgs/second)
Model | NCHW (org) | NHWC (opt)
-- | -- | --
alexnet | 142.38 | 260.16
vgg11 | 24.14 | 75.37
inception_v3 | 14.58 | 45.74
resnet50 | 17.89 | 71.31
resnext101 | 3.5 | 24.12
squeezenet1_0 | 39.75 | 224.99
densenet121 | 16.03 | 36.2
mobilenet_v2 | 7.36 | 129.5
shufflenet | 18.89 | 134.43
unet | 27.15 | 51.59
