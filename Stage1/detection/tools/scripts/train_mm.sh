
#!/usr/bin/env bash

set -x
NGPUSLIST=$1
NGPUS=$2
PY_ARGS=${@:3}



CUDA_VISIBLE_DEVICES=${NGPUSLIST} python train_mm.py --launcher pytorch ${PY_ARGS}