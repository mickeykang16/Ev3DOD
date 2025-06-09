
#!/usr/bin/env bash

# 0 1 --cfg_file cfgs/det_model_cfgs/waymo/LoGoNet-1f.yaml --extra_tag logo_front

set -x
# NGPUSLIST=0,1
# NGPUS=$2
PY_ARGS=${@:3}

while true
do
    PORT=$(( ((RANDOM<<15)|RANDOM) % 49152 + 10000 ))
    status="$(nc -z 127.0.0.1 $PORT < /dev/null &>/dev/null; echo $?)"
    if [ "${status}" != "0" ]; then
        break;
    fi
done
echo $PORT

CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 python -m torch.distributed.launch --nproc_per_node=${NGPUS}  --rdzv_endpoint=localhost:${PORT} train_mm.py --launcher pytorch ${PY_ARGS}