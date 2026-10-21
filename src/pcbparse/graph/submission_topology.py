"""Optional, hash-bound raster topology for the offline submission.

The original detector/OCR observations still own pin positions, identities and
component fields. This module replaces only traced wires and net membership.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from pcbparse.ir.models import CircuitIR, ElectricalNet, Evidence, PhysicalNet, Point, WireSegment
from pcbparse.perception.submission_ocr import _asset
from pcbparse.perception.symbol_body import refine_component_body_boxes
from pcbparse.perception.wire_vectorizer import vectorize_wire_mask, WireVectorizerConfig


def observed_geometry(ir: CircuitIR) -> dict[str, Any]:
    return {'components': {c.internal_id: {
        'category': c.category, 'state': 'active', 'location': c.bbox.model_dump(),
        'realBBox': (c.body_bbox or c.bbox).model_dump(),
    } for c in ir.components}, 'texts': {t.internal_id: {
        'content': t.raw_text, 'location': t.bbox.model_dump(),
        'state': 'active', 'details': t.details, 'score':float(t.score),
    } for t in ir.texts}}


def apply_traced_graph(ir: CircuitIR, graph: dict[str, Any], labels: np.ndarray,ctx=None) -> None:
    """Keep separate physical strokes under each traced electrical net.

    In particular, equally named labels and straight-through crossings may
    unite distinct raster components electrically. They must not be encoded
    as one disconnected PhysicalNet.
    """
    valid_pins = ir.pin_index()
    check=(lambda:ctx.time_budget.check(stage='raster_vectorization')) if ctx is not None else None
    if check:check()
    identity = {p['pin_id']: p['source_pin_id'] for p in graph['pins']
                if p.get('source_pin_id') in valid_pins}
    graph_pins = {identity[p['pin_id']]: p for p in graph['pins'] if p['pin_id'] in identity}
    # Index bounds once. Scanning the entire image separately for each wire
    # label grows costly on dense sheets with hundreds of physical strokes.
    ys, xs = np.nonzero(labels)
    values = labels[ys, xs]
    count = int(values.max(initial=0))+1
    left = np.full(count, labels.shape[1], np.int64)
    top = np.full(count, labels.shape[0], np.int64)
    right = np.full(count, -1, np.int64)
    bottom = np.full(count, -1, np.int64)
    np.minimum.at(left, values, xs); np.minimum.at(top, values, ys)
    np.maximum.at(right, values, xs); np.maximum.at(bottom, values, ys)
    pixel_counts = np.bincount(values, minlength=count)
    retained_labels, skipped_nets = set(), 0
    segments, physical_nets, electrical_nets = [], [], []
    for net in graph['nets']:
        if check:check()
        pins = [identity[p] for p in net['pin_ids'] if p in identity]
        if not pins:
            skipped_nets += 1
            continue
        nid = 'raster:' + net['net_id']
        physical_ids, segment_ids = [], []
        members = set(net['segment_ids'])
        pin_labels = {}
        for pid in pins:
            attachment = graph_pins[pid].get('wire_attachment')
            label = None
            if attachment:
                x, y = int(attachment['x']), int(attachment['y'])
                if 0 <= y < labels.shape[0] and 0 <= x < labels.shape[1]:
                    candidate = int(labels[y, x])
                    if candidate in members:
                        label = candidate
            pin_labels[pid] = label
        for label in sorted(members):
            if check:check()
            if label >= count or right[label] < 0:
                continue
            x1, x2, y1, y2 = int(left[label]), int(right[label])+1, int(top[label]), int(bottom[label])+1
            mask = labels[y1:y2, x1:x2] == label
            vectors = vectorize_wire_mask(mask, origin=(x1, y1), id_namespace=f'{nid}:{label}',
                config=WireVectorizerConfig(rdp_epsilon=1., maximum_mask_pixels=max(4_194_304, mask.size)))
            owned_segments = []
            for line in vectors.polylines:
                for index, (a, b) in enumerate(zip(line.points, line.points[1:])):
                    sid = f'{line.internal_id}:{index}'
                    segments.append(WireSegment(internal_id=sid, start=Point(x=a[0], y=a[1]),
                        end=Point(x=b[0], y=b[1]), source_polyline_id=line.internal_id,
                        evidence=[Evidence(evidence_id='raster-wire:'+sid, source='detector', metadata={
                            'kind': 'raster-model-trace', 'gt_access': False, 'raster_label': label})]))
                    owned_segments.append(sid)
            owned_pins = [p for p in pins if pin_labels[p] == label]
            if owned_segments or owned_pins:
                retained_labels.add(label)
                physical_id = f'{nid}:stroke:{label}'
                physical_nets.append(PhysicalNet(internal_id=physical_id, pins=owned_pins, segments=owned_segments))
                physical_ids.append(physical_id)
                segment_ids.extend(owned_segments)
        for pid in pins:
            if pin_labels[pid] is None:
                physical_id = f'{nid}:terminal:{pid}'
                physical_nets.append(PhysicalNet(internal_id=physical_id, pins=[pid]))
                physical_ids.append(physical_id)
        names = net['names']
        label = names[0] if len(names) == 1 else None
        net_type = 'GROUND' if label and label.upper().endswith('GND') else 'SIGNAL'
        electrical_nets.append(ElectricalNet(internal_id=nid, pins=pins, segments=segment_ids,
            physical_net_ids=physical_ids, label=label, net_type=net_type,
            provenance=['image-raster-topology']))
    ir.segments, ir.junctions = segments, []
    ir.physical_nets, ir.electrical_nets = physical_nets, electrical_nets
    from pcbparse.graph.crossing_wire_geometry import restore_crossing_wires
    restore_crossing_wires(ir, graph.get('crossings', ()), ctx)
    from pcbparse.graph.terminal_wire_geometry import trim_terminal_leads
    trim_terminal_leads(ir)
    for pin in ir.pins:
        # These edge references mean local attachment, not every branch in a
        # raster component. Leave them empty rather than asserting false joins.
        pin.edge_ids = []
        pin.evidence = [e for e in pin.evidence if e.metadata.get('kind') != 'pin_net_evidence']
    for label in ir.net_labels:
        label.attached_physical_net_id = None
    for key in ('connection_evidence', 'connection_params', 'snap_decisions', 'pin_net_evidence', 'label_merge_events'):
        ir.metadata.pop(key, None)
    raster_diagnostics = dict(graph.get('raster_diagnostics', {}))
    retained_pixels = sum(int(pixel_counts[label]) for label in retained_labels)
    raster_diagnostics['wire_export_coverage'] = {
        'traced_pixels': len(values), 'retained_pixels': retained_pixels,
        'unexported_pixels': len(values)-retained_pixels,
        'skipped_nets_without_valid_source_pin': skipped_nets,
        'source_pin_ids_not_in_exportable_ir': sum(p['pin_id'] not in identity for p in graph['pins']),
        'restored_non_junction_axes': ir.metadata['crossing_wire_geometry']['restored_axes'],
        'gt_access': False,
    }
    ir.metadata['raster_topology'] = {'gt_access': False, 'parameters': graph['parameters'],
        'issues': graph['issues'], 'pin_geometry_changed': False, 'pins_created': 0,
        'raster_diagnostics': raster_diagnostics,
        'physical_nets': len(physical_nets), 'electrical_nets': len(electrical_nets)}


class SubmissionTopologyProvider:
    version = 'submission-raster-topology-v10'

    def __init__(self, base, descriptor: dict[str, Any], root: Path, digest: str, device: str,
                 trusted_source_owners=False, palette_gate=False, text_obstacles=False, signal_labels=False,
                 achromatic_wire_recovery=False, palette_stroke_recovery=False, direct_rail_contacts=False):
        import torch
        from pcbparse.perception.raster_wire_models import NetUNet, CrossingPredictor
        self.base, self.descriptor, self.descriptor_sha256 = base, descriptor, getattr(base, 'descriptor_sha256', None)
        self.topology_descriptor_sha256 = digest
        self.trusted_source_owners = trusted_source_owners
        self.palette_gate = palette_gate
        self.text_obstacles = text_obstacles
        self.signal_labels = signal_labels
        self.achromatic_wire_recovery = achromatic_wire_recovery
        self.palette_stroke_recovery = palette_stroke_recovery
        self.direct_rail_contacts = direct_rail_contacts
        if trusted_source_owners or palette_gate:
            self.version = 'submission-raster-topology-guarded-v1'
        if device not in {'auto', 'cpu', 'cuda'}:
            raise ValueError('unsupported raster topology device')
        self.device = 'cuda' if device == 'auto' and torch.cuda.is_available() else ('cpu' if device == 'auto' else device)
        paths = {}
        for name in ('wire', 'crossing'):
            row = descriptor[name]
            path = _asset(root, row['path'])
            if hashlib.sha256(path.read_bytes()).hexdigest() != row['sha256']:
                raise ValueError('raster model SHA256 mismatch: ' + name)
            paths[name] = path
        self.model = NetUNet().to(self.device)
        with paths['wire'].open('rb') as stream:
            saved = torch.load(stream, map_location='cpu', weights_only=True)
        self.model.load_state_dict(saved['model'], strict=True)
        self.model.eval()
        self.crossing = CrossingPredictor(paths['crossing'])

    def build_ir(self, path, ctx=None):
        ir = self.base.build_ir(path, ctx)
        refine_component_body_boxes(ir, path)
        ir.metadata['raster_input_geometry'] = observed_geometry(ir)
        return ir

    def rebuild_topology(self, path, ir: CircuitIR, ctx=None) -> None:
        from pcbparse.graph.raster_trace import trace_nets
        from pcbparse.perception.raster_wire_models import predict
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != ir.image.sha256:
            raise ValueError('raster topology image SHA256 mismatch')
        with Image.open(path) as bitmap:
            if bitmap.size != (ir.image.width, ir.image.height):
                raise ValueError('raster topology image dimensions mismatch')
            rgb = np.asarray(bitmap.convert('RGB'))
        geometry = ir.metadata.pop('raster_input_geometry')
        # Association/pruning may reject detector artifacts after perception.
        # A rejected proposal must not remain an insulating obstacle that
        # cuts real wires in the subsequent raster pass.
        active_components=set(ir.component_index())
        # Missing identity is not evidence that a printed IC outline is a
        # wire. Keep discarded box observations as abstention regions for
        # the new short-jog repair, without changing the baseline wire mask.
        jog_obstacles=[c.get('realBBox')for cid,c in geometry['components'].items()
                       if cid not in active_components and c.get('category','').upper()in{'BOX','IC','BOXIC','BOX_IC'}
                       and c.get('realBBox')is not None]
        geometry['components']={cid:c for cid,c in geometry['components'].items() if cid in active_components}
        from pcbparse.association.supply_names import bind_supply_names
        bind_supply_names(ir, geometry)
        inference_stats={}
        probabilities = predict(self.model, rgb, self.device, self.descriptor['patch_size'],
                                self.descriptor['stride'], ctx=ctx, output_channels=(0,),stats=inference_stats)
        ir.metadata['raster_inference']=inference_stats
        np.clip(probabilities,0,1,out=probabilities)
        np.multiply(probabilities,255,out=probabilities)
        probabilities=probabilities.astype(np.uint8).astype(np.float32)
        probabilities/=255
        points = [{'x': p.point.x, 'y': p.point.y, 'score': p.score, 'source': 'existing-detector',
                   'source_pin_id': p.internal_id, 'component_hint':p.component_id,
                   'source_side':{'L':'left','R':'right','T':'top','B':'bottom'}.get(p.orientation),
                   'owner_verified':any(e.metadata.get('kind')=='visible-stroke-and-text-lanes' for e in p.evidence)} for p in ir.pins
                  if not any(e.metadata.get('kind')=='visible-no-connect-marker' for e in p.evidence)]
        options = dict(self.descriptor['topology'])
        options['jog_body_obstacles']=jog_obstacles
        if self.trusted_source_owners:
            options['trusted_source_owners'] = True
        if self.text_obstacles:
            options['text_obstacles'] = True
        if self.achromatic_wire_recovery:
            options['achromatic_wire_recovery'] = True
        if self.palette_stroke_recovery:
            options['palette_stroke_recovery'] = True
        if self.palette_gate or self.signal_labels:
            from pcbparse.graph.palette_wire import PalettePolicy, estimate_palette, hue_map, palette_cleanup_stability
            bins,ink = hue_map(rgb)
            palette = estimate_palette(ir,bins,ink,PalettePolicy())
            ir.metadata['raster_palette'] = {k: v for k, v in palette.items() if k != 'votes'}
            if self.palette_gate:
                stability = palette_cleanup_stability(ir, bins, ink, palette, PalettePolicy())
                ir.metadata['raster_palette']['cleanup_stability'] = stability
                if palette['enabled'] and stability['stable']:
                    options['palette_wire_hue'] = palette['wire_hue_bin']
            if self.signal_labels:
                from pcbparse.graph.signal_labels import visible_signal_label_anchors, signal_palette_supported
                anchors=visible_signal_label_anchors(ir,bins,ink,palette['wire_hue_bin']) if signal_palette_supported(palette) else []
                geometry['net_labels']=anchors
                ir.metadata['visible_signal_labels']={'anchors':anchors,'gt_access':False}
        graph, debug = trace_nets(rgb, probabilities, points, geometry, **options,
                                 crossing_predictor=self.crossing, crossing_confidence=.9,ctx=ctx)
        apply_traced_graph(ir, graph, debug['wire_labels'],ctx=ctx)
        if self.direct_rail_contacts:
            from pcbparse.graph.terminal_contacts import join_touching_rail_terminals
            join_touching_rail_terminals(ir, rgb, ctx)
            ir.metadata['raster_topology']['electrical_nets'] = len(ir.electrical_nets)
        from pcbparse.graph.colored_lead_contacts import join_observed_colored_leads
        join_observed_colored_leads(ir,rgb,ctx,
            palette=(bins,ink)if self.palette_gate or self.signal_labels else None)
        ir.metadata['raster_topology'].update(physical_nets=len(ir.physical_nets),
                                               electrical_nets=len(ir.electrical_nets))
        ir.metadata['raster_topology']['descriptor_sha256'] = self.topology_descriptor_sha256
        if ctx:
            ctx.model_versions['raster_topology'] = self.version+':'+self.topology_descriptor_sha256[:16]


def apply_submission_topology(base, config: dict[str, Any], root: Path):
    options = config.get('submission_topology')
    if options is None:
        return base
    if not isinstance(options, dict) or options.get('enabled') is not True:
        raise ValueError('submission_topology must be explicitly enabled')
    if set(options)-{'enabled', 'descriptor_path', 'descriptor_sha256', 'device', 'trusted_source_owners_enabled','palette_gate_enabled','text_obstacles_enabled','signal_labels_enabled','achromatic_wire_recovery_enabled','palette_stroke_recovery_enabled','skip_preliminary_wires_enabled','direct_rail_contacts_enabled'}:
        raise ValueError('unsupported submission topology options')
    for key in ('trusted_source_owners_enabled','palette_gate_enabled','text_obstacles_enabled','signal_labels_enabled','achromatic_wire_recovery_enabled','palette_stroke_recovery_enabled','skip_preliminary_wires_enabled','direct_rail_contacts_enabled'):
        if type(options.get(key,False)) is not bool:
            raise ValueError(key+' must be a boolean')
    path = _asset(root, options['descriptor_path'])
    blob = path.read_bytes()
    digest = hashlib.sha256(blob).hexdigest()
    if digest != options['descriptor_sha256']:
        raise ValueError('raster topology descriptor SHA256 mismatch')
    descriptor = json.loads(blob)
    if descriptor['patch_size'] != 512 or descriptor['stride'] != 384:
        raise ValueError('unsupported raster wire tiling')
    if options.get('skip_preliminary_wires_enabled'):
        if config.get('graph', {}).get('raster_direct_enabled') is not True:
            raise ValueError('skipping preliminary wires requires direct raster topology')
        if 'junction.kind' in config.get('qwen', {}).get('task_types', []):
            raise ValueError('junction reasoning requires preliminary wires')
        composite = getattr(base, 'base', base)
        composite.config = replace(composite.config, preliminary_wire_enabled=False)
    return SubmissionTopologyProvider(base, descriptor, root, digest, options['device'],
                                      options.get('trusted_source_owners_enabled',False),options.get('palette_gate_enabled',False),
                                      options.get('text_obstacles_enabled',False),options.get('signal_labels_enabled',False),
                                      options.get('achromatic_wire_recovery_enabled',False),
                                      options.get('palette_stroke_recovery_enabled',False),
                                      options.get('direct_rail_contacts_enabled',False))
