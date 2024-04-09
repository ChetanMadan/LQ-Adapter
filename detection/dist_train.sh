#!/usr/bin/env bash

CONFIG=$1
GPUS=$2
WORK_DIR=$3
PORT=${PORT:-29511}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/train.py $CONFIG --work-dir=${WORK_DIR} --auto-resume --deterministic --launcher pytorch ${@:4}
