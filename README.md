# Convnet Performance Benchmark for PyTorch

## usage
```
### the script runs on GPU when GPU is available, otherwise on CPU
### for training performance (batch mode)
./run.sh


### for inference performance
### to achieve best performance on CPU, tensor needs to be stored in mkldnn layout
###   --mkldnn will allow input and output tensor to be in mkldnn layout
###   --cache-weight will allow weight to be cached in mkldnn layout
###   (beneficial for single batch size inference)

### batch mode (input/output/weight in native layout)
./run.sh --inference
### single batch (input/output/weight in native layout)
./run.sh --inference --single
### single batch (input/output in mkldnn layout; weight uncached)
./run.sh --inference --single --mkldnn
### single batch (input/output in mkldnn layout; weight cached)
./run.sh --inference --single --mkldnn --cache-weight
```
