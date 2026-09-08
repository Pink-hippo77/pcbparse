from __future__ import annotations

import hashlib
from pathlib import Path

from PIL import Image

from pcbparse.ir.models import CircuitIR, ImageInfo, IRState
from pcbparse.pipeline.context import RunContext


class EmptyImageProvider:
    """CPU disaster-recovery provider that emits a valid empty Circuit-IR.

    It guarantees interface/schema availability when learned weights are absent;
    it is not an accuracy baseline and should never be reported as one.
    """

    def build_ir(self, input_path: str | Path, ctx: RunContext | None = None) -> CircuitIR:
        path = Path(input_path)
        with Image.open(path) as image:
            width, height = image.size
            actual_format = image.format
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        return CircuitIR(
            state=IRState.PERCEIVED,
            image=ImageInfo(
                image_id=ctx.image_id if ctx else path.stem,
                path=str(path),
                width=width,
                height=height,
                sha256=digest,
                actual_format=actual_format,
            ),
            metadata={"provider": "empty_disaster_recovery"},
        )

