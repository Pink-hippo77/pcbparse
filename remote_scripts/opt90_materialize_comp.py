"""opt90-comp: materialize the Component/Pin training index with DEFAULT config.

GT-ACCESS: reads only the derived train/val/test manifests and source images;
no GT enters inference. Training uses `train` (train240) only; `val` is the
early-stopping/gate monitor; `test` is never read.

Every ComponentPinMaterializeConfig field keeps its default, i.e. the same
contract the blessed component-v5-fullmix training used. Output is isolated
under work/opt90-comp/; the shared artifacts/training/materialized tree is
never written.

Usage: opt90_materialize_comp.py [output_root]
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path('/home/data/yinjx/pcbparse-competition/project')
sys.path.insert(0, str(ROOT / 'src'))

from pcbparse.training.component_pin_materialize import (  # noqa: E402
    ComponentPinMaterializeConfig,
    materialize_component_pin_training_data,
)

DEFAULT_OUT = 'work/opt90-comp/materialized-default/component-pin'


def main() -> int:
    output_root = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_OUT
    config = ComponentPinMaterializeConfig()
    print('materialize config:', config, flush=True)
    result = materialize_component_pin_training_data(
        workspace_root=ROOT,
        output_root=output_root,
        derived_root='data/derived/dataset300-v2',
        source_root='Dataset300/Dataset300',
        source_manifest='data/manifests/dataset300.jsonl',
        split_manifest='data/splits/dataset300_split.jsonl',
        config=config,
        sample_ids=None,
        write_crops=False,
        verify_source_hashes=True,
    )
    print('binding_sha256:', result.binding_sha256)
    print('component_index:', result.component_index)
    print('pin_index:', result.pin_index)
    print('summary:', result.summary)
    print('component_view_count:', result.component_view_count)
    print('pin_component_count:', result.pin_component_count)
    print('crop_count:', result.crop_count)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
