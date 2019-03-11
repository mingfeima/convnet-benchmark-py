# Convnet Performance Benchmark for PyTorch

## usage
```
### the script runs on GPU when GPU is available, otherwise on CPU
### for training performance (batched mode)
./run.sh
### for inference performance (batched mode)
./run.sh --inference
### for inference latency (single batch)
./run.sh --inference --single
```
