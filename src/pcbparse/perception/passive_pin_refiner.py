"""Optional specialist observations for already verified passive leads."""
from __future__ import annotations

import hashlib

from pcbparse.io import load_image, make_component_view
from pcbparse.perception.composite_provider import _restore_pins
from pcbparse.perception.torch_pin_adapter import create_pin_provider


class PassivePinRefiner:
    def __init__(self, options, root):
        from pcbparse.perception.submission_ocr import _asset
        if not isinstance(options,dict) or options.get('enabled') is not True:
            raise ValueError('passive pin refiner must be explicitly enabled')
        if set(options)-{'enabled','model_path','model_sha256','split_sha256','inference'}:
            raise ValueError('unsupported passive pin refiner options')
        path=_asset(root,options['model_path'])
        self.sha256=hashlib.sha256(path.read_bytes()).hexdigest()
        if self.sha256!=options['model_sha256']:
            raise ValueError('passive pin refiner SHA256 mismatch')
        inference=options.get('inference')
        if not isinstance(inference,dict) or inference.get('component_mask_channel') is not True:
            raise ValueError('passive refiner requires its trained ownership-mask input')
        self.provider=create_pin_provider(path,{'task':'pin','sha256':self.sha256,
            'artifact_format':'torchscript','version':'passive-pin-'+self.sha256[:12],
            'inference':inference})

    def observations(self, path, ir, original_components, config, ctx=None):
        wanted={p.component_id for p in ir.pins
                if p.internal_id.startswith(('visual-cap-pin:','visual-rl-pin:'))}
        components=[c for c in original_components if c.internal_id in wanted]
        if not components:return []
        original=load_image(path)
        views=[make_component_view(original,c,
            padding=max(max(c.bbox.width,c.bbox.height)*config.component_roi_padding_factor,
                        config.resolved_pin_component_roi_padding_px),
            view_id_prefix='passive-refiner') for c in components]
        index={v.full_roi.view_id:v.full_roi for v in views}
        owners={v.full_roi.view_id:v.component_id for v in views}
        # Preserve primary branch model metadata; this model supplies only
        # observations for a separate visible-lead acceptance rule.
        primary_version=ctx.model_versions.get('pin') if ctx else None
        try:
            pins=_restore_pins(self.provider.predict(views,ctx),index,owners)
        finally:
            if ctx:
                if primary_version is None:ctx.model_versions.pop('pin',None)
                else:ctx.model_versions['pin']=primary_version
        if ctx:
            ctx.time_budget.check(stage='passive_pin_refiner')
            ctx.model_versions['passive_pin_refiner']=self.provider.version
        return [{'id':p.internal_id,'owner':p.component_id,'point':(p.point.x,p.point.y),
                 'orientation':p.orientation,'score':p.score} for p in pins]
