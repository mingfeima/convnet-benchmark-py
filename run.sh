#!/bin/sh

ARGS=""
if [[ "$1" == "--inference" ]]; then
    ARGS="$ARGS --inference"
    echo "### inference only"
    shift
fi

if [[ "$1" == "--single" ]]; then
    ARGS="$ARGS --single-batch-size"
    echo "### using single batch size"
    shift
fi

if [[ "$1" == "--mkldnn" ]]; then
    ARGS="$ARGS --mkldnn"
    echo "### cache input/output in mkldnn format"
    shift
fi

if [[ "$1" == "--cache-weight" ]]; then
    ARGS="$ARGS --cache-weight"
    echo "### cache weight in mkldnn format"
    shift
fi

CORES=`lscpu | grep Core | awk '{print $4}'`
SOCKETS=`lscpu | grep Socket | awk '{print $2}'`
TOTAL_CORES=`expr $CORES \* $SOCKETS`

KMP_SETTING="KMP_AFFINITY=granularity=fine,compact,1,0"
KMP_BLOCKTIME=1

export OMP_NUM_THREADS=$TOTAL_CORES
export $KMP_SETTING
export KMP_BLOCKTIME=$KMP_BLOCKTIME

echo -e "### using OMP_NUM_THREADS=$TOTAL_CORES"
echo -e "### using $KMP_SETTING"
echo -e "### using KMP_BLOCKTIME=$KMP_BLOCKTIME\n"

python -u benchmark.py $ARGS
