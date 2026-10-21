"""Explicit submission OCR bundle, separate from the baseline promotion lock."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any
import time

from pcbparse.association.ocr_observation_cleanup import clean_ocr_observations, classify_visible_ic_titles
from pcbparse.perception.onnx_ocr_adapter import ONNXOCRAdapter
from pcbparse.perception.symbol_body import readback_rgb


def resolve_submission_assets(config: dict[str, Any], root: Path) -> None:
    """Resolve explicit relative static assets within the unpacked application."""
    for section, key in (('association', 'component_model_catalog_path'), ('knowledge', 'pin_library_path')):
        value = config.get(section, {}).get(key)
        if value is not None:
            config[section][key] = str(_asset(root, value))


def _asset(root: Path, value: str) -> Path:
    root = root.resolve()
    if not isinstance(value, str) or not value.strip() or Path(value).is_absolute():
        raise ValueError('submission assets require a relative application path')
    path = (root/value).resolve(strict=True)
    path.relative_to(root)
    if not path.is_file():
        raise ValueError('submission asset must be a file')
    return path


class SubmissionOCRProvider:
    version = 'submission-ppocr-v5-visual-readback-v10'

    def __init__(self, base, descriptor_sha256: str, engine=None, ic_readback=False, rail_readback=False,
                 color_terminals=False, interior_ic=False, learned_passive_tips=False, learned_ic_tips=False, ic_outline_dedup=False,
                 monochrome_ground_ports=False, shaped_ic_leads=False, learned_color_passive_tips=False,
                 passive_pin_refiner=None, visual_ic_pin_refiner=False, triangle_grounds=False, refine_name_lanes=False,
                 local_reference_readback=False, local_reference_crop_retry=False,
                 local_reference_line_isolation=False, local_reference_selective_retry=False,
                 decorated_ic_leads=False, local_symbol_types=False,
                 occluded_passives=False, interior_number_tips=False, repeated_name_lanes=False,
                 observed_terminal_boundaries=False, observed_capacitor_extents=False):
        self.base = base
        self.descriptor_sha256 = descriptor_sha256
        self.engine = engine
        self.ic_readback = ic_readback
        self.rail_readback = rail_readback
        self.color_terminals = color_terminals
        self.interior_ic = interior_ic
        self.learned_passive_tips = learned_passive_tips
        self.learned_ic_tips = learned_ic_tips
        self.ic_outline_dedup = ic_outline_dedup
        self.monochrome_ground_ports = monochrome_ground_ports
        self.shaped_ic_leads = shaped_ic_leads
        self.learned_color_passive_tips = learned_color_passive_tips
        self.passive_pin_refiner = passive_pin_refiner
        self.visual_ic_pin_refiner = visual_ic_pin_refiner
        self.triangle_grounds = triangle_grounds
        self.refine_name_lanes = refine_name_lanes
        self.local_reference_readback = local_reference_readback
        self.local_reference_crop_retry = local_reference_crop_retry
        self.local_reference_line_isolation = local_reference_line_isolation
        self.local_reference_selective_retry = local_reference_selective_retry
        self.decorated_ic_leads = decorated_ic_leads
        self.local_symbol_types = local_symbol_types
        self.occluded_passives = occluded_passives
        self.interior_number_tips = interior_number_tips
        self.repeated_name_lanes = repeated_name_lanes
        self.observed_terminal_boundaries = observed_terminal_boundaries
        self.observed_capacitor_extents = observed_capacitor_extents
        if color_terminals or interior_ic or learned_passive_tips or learned_ic_tips or ic_outline_dedup:
            self.version = 'submission-ppocr-v5-terminal-refinement-v1'

    def build_ir(self, path, ctx=None):
        ir = self.base.build_ir(path, ctx)
        clean_ocr_observations(ir)
        rgb = None
        type_seconds = 0.
        if self.local_symbol_types:
            from pcbparse.perception.local_symbol_types import refine_local_symbol_types
            begin = time.monotonic()
            rgb = readback_rgb(ir, path)
            refine_local_symbol_types(ir, rgb, ctx)
            type_seconds = time.monotonic()-begin
        raw_passives=[c.model_copy(deep=True) for c in ir.components if c.category.upper() in {'R','C','L'}] if self.passive_pin_refiner else []
        timings = {'local_symbol_types': round(type_seconds, 6)} if self.local_symbol_types else {}
        passive_observations=[]
        if self.learned_passive_tips or self.learned_color_passive_tips:
            components=ir.component_index()
            passive_observations=[{'id':p.internal_id,'owner':p.component_id,'point':(p.point.x,p.point.y),
                                   'orientation':p.orientation,'score':p.score}
                                  for p in ir.pins if p.component_id in components
                                  and components[p.component_id].category.upper() in {'C','R','L'}]
        if rgb is None and (self.ic_readback or self.rail_readback or self.color_terminals or self.learned_passive_tips or self.learned_color_passive_tips or self.monochrome_ground_ports or self.passive_pin_refiner or self.interior_number_tips):
            rgb = readback_rgb(ir, path)
        interior_rectangles = None
        if self.ic_readback:
            from pcbparse.perception.ic_visual_readback import measure_ic_outlines, recover_visual_ics
            begin = time.monotonic()
            shared = {}
            if self.interior_number_tips:
                from pcbparse.graph.palette_wire import hue_map
                from pcbparse.perception.ic_visual_readback import closed_rectangle_sets
                palette = hue_map(rgb)
                rectangles = closed_rectangle_sets(rgb, palette=palette,
                    check=lambda: self.engine._check(ctx, 'ic_outline_readback'))
                interior_rectangles = rectangles[38]
                shared = {'palette': palette, 'ordinary_rectangles': rectangles[12]}
            outlines = measure_ic_outlines(rgb, self.engine, ctx,
                learned_tip_fusion=self.learned_ic_tips, outline_dedup=self.ic_outline_dedup, **shared)
            from pcbparse.perception.compact_connectors import compact_connector_outlines
            compact = compact_connector_outlines(ir, rgb, outlines[0], ctx)
            outlines = (outlines[0], (*outlines[1], *compact))
            recover_visual_ics(ir, path, self.engine, ctx, rgb=rgb,learned_tip_fusion=self.learned_ic_tips,
                               outline_dedup=self.ic_outline_dedup,shaped_leads=self.shaped_ic_leads,refine_name_lanes=self.refine_name_lanes,
                               decorated_leads=self.decorated_ic_leads, outline_geometry=outlines)
            if self.interior_ic:
                recover_visual_ics(ir,path,self.engine,ctx,rgb=rgb,interior_only=True,learned_tip_fusion=self.learned_ic_tips,
                                   outline_dedup=self.ic_outline_dedup,shaped_leads=self.shaped_ic_leads,refine_name_lanes=self.refine_name_lanes,
                                   decorated_leads=self.decorated_ic_leads, outline_geometry=outlines)
            timings['ic'] = round(time.monotonic()-begin, 6)
        if self.rail_readback:
            from pcbparse.perception.passive_visual_readback import recover_visual_capacitors
            from pcbparse.perception.rail_visual_readback import recover_visual_rails
            from pcbparse.perception.two_terminal_readback import recover_two_terminal_symbols
            begin = time.monotonic()
            recover_visual_capacitors(ir, path, ctx, rgb=rgb,occluded_plates=self.occluded_passives)
            timings['capacitor'] = round(time.monotonic()-begin, 6)
            begin = time.monotonic()
            recover_two_terminal_symbols(ir, path, ctx, rgb=rgb,occluded_rectangles=self.occluded_passives)
            timings['two_terminal'] = round(time.monotonic()-begin, 6)
            from pcbparse.perception.crystal_visual_readback import recover_visual_crystals
            begin = time.monotonic()
            recover_visual_crystals(ir, rgb, ctx)
            timings['crystal'] = round(time.monotonic()-begin, 6)
            from pcbparse.perception.switch_visual_readback import recover_visual_switches
            begin = time.monotonic()
            recover_visual_switches(ir, rgb, ctx)
            timings['switch_contacts'] = round(time.monotonic()-begin, 6)
            begin = time.monotonic()
            recover_visual_rails(ir, path, self.engine, ctx, rgb=rgb,triangle_grounds=self.triangle_grounds)
            timings['rail'] = round(time.monotonic()-begin, 6)
        if self.monochrome_ground_ports:
            from pcbparse.perception.rail_visual_readback import refine_monochrome_ground_ports
            begin=time.monotonic()
            refine_monochrome_ground_ports(ir,rgb)
            timings['ground_ports']=round(time.monotonic()-begin,6)
        if self.learned_passive_tips or self.learned_color_passive_tips:
            from pcbparse.perception.color_terminals import restore_monochrome_passive_tips
            begin=time.monotonic()
            restore_monochrome_passive_tips(ir,rgb,passive_observations,allow_color=self.learned_color_passive_tips)
            timings['monochrome_terminals']=round(time.monotonic()-begin,6)
        if self.passive_pin_refiner:
            from pcbparse.perception.color_terminals import restore_monochrome_passive_tips
            begin=time.monotonic()
            targets=raw_passives
            if self.occluded_passives:
                original_ids={c.internal_id for c in raw_passives}
                # The specialist was not validated for newly synthesized
                # rectangular resistor views or cross-class detector bodies.
                # Supplement only newly recovered, visibly verified capacitors.
                targets=[*raw_passives,*[c for c in ir.components if c.internal_id not in original_ids
                         and c.internal_id.startswith('visual-cap:') and c.category.upper()=='C']]
            observations=self.passive_pin_refiner.observations(path,ir,targets,self.base.config,ctx)
            primary_report=ir.metadata.get('monochrome_terminal_fusion')
            report=restore_monochrome_passive_tips(ir,rgb,observations,allow_color=True,
                observation_source='specialist:'+self.passive_pin_refiner.sha256[:12])
            report['model_sha256']=self.passive_pin_refiner.sha256
            ir.metadata['passive_specialist_terminal_fusion']=report
            ir.metadata['monochrome_terminal_fusion']=primary_report
            timings['passive_pin_refiner']=round(time.monotonic()-begin,6)
        if self.color_terminals:
            from pcbparse.perception.color_terminals import refine_color_terminals
            begin = time.monotonic()
            refine_color_terminals(ir, rgb)
            timings['color_terminals'] = round(time.monotonic()-begin, 6)
        if self.visual_ic_pin_refiner:
            from pcbparse.perception.ic_visual_readback import refine_achromatic_ic_tips
            begin=time.monotonic()
            refine_achromatic_ic_tips(ir,path,rgb,self.base.pin_provider,self.base.config,ctx)
            timings['visual_ic_pin_refiner']=round(time.monotonic()-begin,6)
        if self.local_reference_readback:
            from pcbparse.perception.reference_readback import recover_local_references
            begin = time.monotonic()
            recover_local_references(ir, rgb if rgb is not None else readback_rgb(ir, path), self.engine, ctx,
                                     crop_retry=self.local_reference_crop_retry,
                                     line_isolation=self.local_reference_line_isolation,
                                     selective_retry=self.local_reference_selective_retry)
            timings['local_references'] = round(time.monotonic()-begin, 6)
        if self.repeated_name_lanes:
            from pcbparse.perception.repeated_name_lanes import refine_repeated_pin_name_lanes
            begin=time.monotonic()
            refine_repeated_pin_name_lanes(ir,ctx)
            timings['repeated_name_lanes']=round(time.monotonic()-begin,6)
        classify_visible_ic_titles(ir)
        if self.interior_number_tips:
            from pcbparse.perception.interior_number_tips import refine_interior_number_tips
            begin=time.monotonic()
            refine_interior_number_tips(ir,rgb,ctx,rectangles=interior_rectangles)
            timings['interior_number_tips']=round(time.monotonic()-begin,6)
        if self.observed_terminal_boundaries:
            from pcbparse.perception.observed_terminal_boundaries import refine_observed_terminal_boundaries
            begin=time.monotonic()
            refine_observed_terminal_boundaries(ir,rgb,ctx)
            timings['observed_terminal_boundaries']=round(time.monotonic()-begin,6)
        if self.observed_capacitor_extents:
            from pcbparse.perception.capacitor_symbol_extent import restore_capacitor_symbol_extents
            begin=time.monotonic()
            restore_capacitor_symbol_extents(ir,rgb,ctx)
            timings['capacitor_symbol_extent']=round(time.monotonic()-begin,6)
        ir.metadata['submission_ocr'] = {
            'descriptor_sha256': self.descriptor_sha256,
            'baseline_lock_modified': False, 'model_promotion': False, 'gt_access': False,
        }
        ir.metadata['supplement_timings_seconds'] = timings
        return ir


def apply_submission_ocr(base, config: dict[str, Any], root: Path):
    options = config.get('submission_ocr')
    if options is None:
        return base
    if not isinstance(options, dict) or options.get('enabled') is not True:
        raise ValueError('submission_ocr must be explicitly enabled')
    if set(options)-{'enabled', 'descriptor_path', 'descriptor_sha256', 'ic_readback_enabled', 'rail_readback_enabled', 'color_terminals_enabled','interior_ic_enabled','learned_passive_tips_enabled','learned_color_passive_tips_enabled','learned_ic_tips_enabled','ic_outline_dedup_enabled','monochrome_ground_ports_enabled','shaped_ic_leads_enabled','passive_pin_refiner','visual_ic_pin_refiner_enabled','triangle_grounds_enabled','pin_name_lane_refinement_enabled','local_reference_readback_enabled','local_reference_crop_retry_enabled','local_reference_line_isolation_enabled','local_reference_selective_retry_enabled','decorated_ic_leads_enabled','local_symbol_type_readback_enabled','occluded_passive_readback_enabled','interior_number_tip_refinement_enabled','cpu_threads','repeated_pin_name_lanes_enabled','observed_terminal_boundaries_enabled','observed_capacitor_extents_enabled'}:
        raise ValueError('unsupported submission_ocr options')
    if type(options.get('ic_readback_enabled', False)) is not bool:
        raise ValueError('ic_readback_enabled must be a boolean')
    if type(options.get('local_symbol_type_readback_enabled', False)) is not bool:
        raise ValueError('local_symbol_type_readback_enabled must be a boolean')
    for flag in ['occluded_passive_readback_enabled','interior_number_tip_refinement_enabled','repeated_pin_name_lanes_enabled','observed_terminal_boundaries_enabled','observed_capacitor_extents_enabled']:
        if type(options.get(flag,False)) is not bool:
            raise ValueError(flag+' must be a boolean')
    if options.get('repeated_pin_name_lanes_enabled') and not options.get('ic_readback_enabled'):
        raise ValueError('repeated pin name lanes require IC readback')
    threads=options.get('cpu_threads',4)
    if type(threads) is not int or not 1<=threads<=16:
        raise ValueError('OCR cpu_threads must be an integer in [1,16]')
    if options.get('occluded_passive_readback_enabled') and not options.get('rail_readback_enabled'):
        raise ValueError('occluded passive readback requires rail readback')
    if type(options.get('rail_readback_enabled', False)) is not bool:
        raise ValueError('rail_readback_enabled must be a boolean')
    if type(options.get('color_terminals_enabled', False)) is not bool:
        raise ValueError('color_terminals_enabled must be a boolean')
    if type(options.get('interior_ic_enabled', False)) is not bool:
        raise ValueError('interior_ic_enabled must be a boolean')
    if type(options.get('learned_passive_tips_enabled', False)) is not bool:
        raise ValueError('learned_passive_tips_enabled must be a boolean')
    if type(options.get('learned_color_passive_tips_enabled', False)) is not bool:
        raise ValueError('learned_color_passive_tips_enabled must be a boolean')
    if type(options.get('learned_ic_tips_enabled', False)) is not bool:
        raise ValueError('learned_ic_tips_enabled must be a boolean')
    if type(options.get('ic_outline_dedup_enabled', False)) is not bool:
        raise ValueError('ic_outline_dedup_enabled must be a boolean')
    if type(options.get('monochrome_ground_ports_enabled', False)) is not bool:
        raise ValueError('monochrome_ground_ports_enabled must be a boolean')
    if type(options.get('shaped_ic_leads_enabled', False)) is not bool:
        raise ValueError('shaped_ic_leads_enabled must be a boolean')
    if type(options.get('decorated_ic_leads_enabled', False)) is not bool:
        raise ValueError('decorated_ic_leads_enabled must be a boolean')
    if options.get('decorated_ic_leads_enabled') and not options.get('ic_readback_enabled'):
        raise ValueError('decorated IC leads require visual IC readback')
    if type(options.get('visual_ic_pin_refiner_enabled', False)) is not bool:
        raise ValueError('visual_ic_pin_refiner_enabled must be a boolean')
    if options.get('visual_ic_pin_refiner_enabled') and not options.get('ic_readback_enabled'):
        raise ValueError('visual IC pin refinement requires visual IC readback')
    if type(options.get('triangle_grounds_enabled', False)) is not bool:
        raise ValueError('triangle_grounds_enabled must be a boolean')
    if type(options.get('pin_name_lane_refinement_enabled', False)) is not bool:
        raise ValueError('pin_name_lane_refinement_enabled must be a boolean')
    if type(options.get('local_reference_readback_enabled', False)) is not bool:
        raise ValueError('local_reference_readback_enabled must be a boolean')
    if type(options.get('local_reference_crop_retry_enabled', False)) is not bool:
        raise ValueError('local_reference_crop_retry_enabled must be a boolean')
    if options.get('local_reference_crop_retry_enabled') and not options.get('local_reference_readback_enabled'):
        raise ValueError('reference crop retry requires local reference readback')
    for flag in ('local_reference_line_isolation_enabled', 'local_reference_selective_retry_enabled'):
        if type(options.get(flag, False)) is not bool:
            raise ValueError(flag + ' must be a boolean')
        if options.get(flag) and not options.get('local_reference_readback_enabled'):
            raise ValueError(flag + ' requires local reference readback')
    if options.get('local_reference_selective_retry_enabled') and not options.get('local_reference_crop_retry_enabled'):
        raise ValueError('selective reference retry requires reference crop retry')
    descriptor = _asset(root, options['descriptor_path'])
    adapter = ONNXOCRAdapter(descriptor, {'sha256': options['descriptor_sha256'],
                                        'inference': {'device': 'cpu', 'cpu_threads': threads}})
    base.config = replace(base.config, ocr_tile_size=1024, ocr_tile_overlap=160,
                          component_roi_padding_factor=0.0)
    base.ocr_provider = adapter
    passive_refiner=None
    if options.get('passive_pin_refiner') is not None:
        from pcbparse.perception.passive_pin_refiner import PassivePinRefiner
        passive_refiner=PassivePinRefiner(options['passive_pin_refiner'],root)
    return SubmissionOCRProvider(base, adapter.descriptor_sha256, adapter,
                                 options.get('ic_readback_enabled', False),options.get('rail_readback_enabled', False),
                                 options.get('color_terminals_enabled', False),options.get('interior_ic_enabled',False),
                                 options.get('learned_passive_tips_enabled',False),options.get('learned_ic_tips_enabled',False),
                                 options.get('ic_outline_dedup_enabled',False),options.get('monochrome_ground_ports_enabled',False),
                                 options.get('shaped_ic_leads_enabled',False),options.get('learned_color_passive_tips_enabled',False),
                                 passive_refiner,options.get('visual_ic_pin_refiner_enabled',False),
                                 options.get('triangle_grounds_enabled',False),
                                 options.get('pin_name_lane_refinement_enabled',False),
                                 options.get('local_reference_readback_enabled',False),
                                 options.get('local_reference_crop_retry_enabled',False),
                                 options.get('local_reference_line_isolation_enabled',False),
                                 options.get('local_reference_selective_retry_enabled',False),
                                 options.get('decorated_ic_leads_enabled',False),
                                 options.get('local_symbol_type_readback_enabled',False),
                                 options.get('occluded_passive_readback_enabled',False),
                                 options.get('interior_number_tip_refinement_enabled',False),
                                 options.get('repeated_pin_name_lanes_enabled',False),
                                 options.get('observed_terminal_boundaries_enabled',False),
                                 options.get('observed_capacitor_extents_enabled',False))
