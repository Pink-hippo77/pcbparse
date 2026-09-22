# Legacy training archive

This directory contains the former `E:\EDA比赛测试` workspace.  It is kept as
historical evidence and is not the active implementation.

## Layout

- `scripts/`: early training, smoke-test, and evaluation entry points.  Their
  live paths are resolved by `scripts/_paths.py` after consolidation.
- `runs/`: model checkpoints, exports, metrics, and logs from the early
  training experiments.
- `results/inference/`: historical inference and evaluation outputs.
- `documents/`: early design and deployment documents plus the official task
  brief copy that was stored with the experiments.

The JSON status files under `runs/` intentionally retain the absolute paths
recorded at run time.  Those strings are provenance, not current path settings.
