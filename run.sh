#!/bin/sh

### [tcmalloc]: compile tcmalloc according to
###   http://goog-perftools.sourceforge.net/doc/tcmalloc.html
export LD_PRELOAD=/home/mingfeim/packages/gperftools-2.8/install/lib/libtcmalloc.so


CORES=`lscpu | grep Core | awk '{print $4}'`
SOCKETS=`lscpu | grep Socket | awk '{print $2}'`
TOTAL_CORES=`expr $CORES \* $SOCKETS`

PREFIX=""

ARGS=""
if [[ "$1" == "--inference" ]]; then
    ARGS="$ARGS --inference"
    echo "### inference only"
    ### using single socket for inference to allow numactrl to work
    TOTAL_CORES=$CORES
    LAST_CORE=`expr $CORES - 1`
    PREFIX="numactl --physcpubind=0-$LAST_CORE --membind=0"
    shift
fi

if [[ "$1" == "--single" ]]; then
    ARGS="$ARGS --single-batch-size"
    echo "### using single batch size"
    shift
fi

if [[ "$1" == "--mkldnn" ]]; then
    ARGS="$ARGS --mkldnn"
    echo "### use mkldnn blocking format"
    shift
fi

if [[ "$1" == "--channels_last" ]]; then
    ARGS="$ARGS --channels_last"
    echo "### use channels last format"
    shift
fi

KMP_SETTING="KMP_AFFINITY=granularity=fine,compact,1,0"
KMP_BLOCKTIME=1

export OMP_NUM_THREADS=$TOTAL_CORES
export $KMP_SETTING
export KMP_BLOCKTIME=$KMP_BLOCKTIME

echo -e "### using OMP_NUM_THREADS=$TOTAL_CORES"
echo -e "### using $KMP_SETTING"
echo -e "### using KMP_BLOCKTIME=$KMP_BLOCKTIME"
echo -e "### using $PREFIX\n\n"

$PREFIX python -u benchmark.py $ARGS
