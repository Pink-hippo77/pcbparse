#!/bin/bash
# opt90 wire training launch (GPU1, serial queue; run ONLY with Lead slot approval).
#
# Usage: run_training.sh timing   -> GPU timing smoke (~minutes, sizes the job)
#        run_training.sh formal   -> formal 20-epoch training -> work/opt90-wire/train-v1
#
# Contract: GPU1 must be idle (no compute processes, <1GB used) before launch;
# the caller owns the queue slot (ocr -> pin -> wire).
set -euo pipefail
cd /home/data/yinjx/pcbparse-competition/project

STAGE=${1:?usage: run_training.sh timing|formal}
export CUDA_VISIBLE_DEVICES=1
export PYTHONPATH=src
PY=/home/data/yinjx/pcbparse-competition/runtime/ocr-paddle33-cu118-py312/bin/python
export PCB_WIRE_JUNCTION_MATERIALIZED_ROOT=$PWD/work/opt90-wire/materialized

# serial-queue guard
PROCS=$(nvidia-smi --id=1 --query-compute-apps=pid --format=csv,noheader | sed '/^\s*$/d' | wc -l)
USED=$(nvidia-smi --id=1 --query-gpu=memory.used --format=csv,noheader,nounits)
if [ "$PROCS" -gt 0 ] || [ "$USED" -ge 1024 ]; then
  echo "GPU1 not idle: processes=$PROCS used_mib=$USED" >&2
  exit 1
fi

case "$STAGE" in
  timing)
    exec $PY -m pcbparse.training.torch_train_cli \
      --config configs/train/opt90_wire_v1_negpatch.json --task wire \
      --output work/opt90-wire/train-timing --workspace-root . \
      --device cuda --precision fp16 \
      --smoke --smoke-epochs 1 --smoke-train-batches 40 --smoke-val-batches 10 \
      --smoke-export-samples 4
    ;;
  formal)
    exec $PY -m pcbparse.training.torch_train_cli \
      --config configs/train/opt90_wire_v1_negpatch.json --task wire \
      --output work/opt90-wire/train-v1 --workspace-root . \
      --device cuda --precision fp16
    ;;
  *)
    echo "unknown stage: $STAGE" >&2
    exit 2
    ;;
esac
