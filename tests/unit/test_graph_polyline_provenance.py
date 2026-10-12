from __future__ import annotations

import itertools
import unittest

from pcbparse.graph.physical import build_physical_graph, collect_segment_connections
from pcbparse.ir.models import CircuitIR, ImageInfo, Point, WireSegment
from pcbparse.validation.graph import validate_graph
from pcbparse.validation.repair import repair_ir_once


def segment(sid, start, end, *, channel, source=None, aliases=(), score=1.0):
    return WireSegment(
        internal_id=sid, start=Point(x=start[0], y=start[1]),
        end=Point(x=end[0], y=end[1]), channel=channel,
        source_polyline_id=source, source_polyline_ids=list(aliases), score=score,
    )


def fixture(segments):
    return CircuitIR(
        image=ImageInfo(image_id='turn', path='turn.png', width=80, height=80),
        segments=list(segments),
    )


class DuplicatePolylineProvenanceTests(unittest.TestCase):
    def test_dedup_keeps_observed_turn_after_roundtrip(self):
        ir = fixture([
            segment('weak', (10, 10), (30, 10), channel='horizontal', source='corner', aliases=('other-observation',), score=0.7),
            segment('strong', (30, 10), (10, 10), channel='horizontal', source='straight', score=0.9),
            segment('turn', (30, 10), (30, 30), channel='vertical', source='corner'),
        ])
        ir.physical_nets, _ = build_physical_graph(ir, endpoint_tolerance_px=1.0)
        self.assertEqual(1, len(ir.physical_nets))
        repaired = repair_ir_once(ir)
        restored = CircuitIR.model_validate_json(repaired.model_dump_json())
        self.assertEqual(('corner', 'other-observation', 'straight'), restored.segment_index()['strong'].observed_polyline_ids)
        self.assertEqual({'strong', 'turn'}, set(restored.physical_nets[0].segments))
        self.assertEqual([], [e for e in validate_graph(restored, endpoint_tolerance_px=1.0) if e.severity == 'ERROR'])
        self.assertEqual(3, len(ir.segments))  # repair must not mutate its input
        self.assertEqual([], ir.segment_index()['strong'].source_polyline_ids)
        removed = next(r for r in restored.repairs if r.rule_id == 'REPAIR.REMOVE_DUPLICATE_SEGMENT')
        self.assertEqual('corner', removed.before['source_polyline_id'])
        self.assertEqual(['other-observation'], removed.before['source_polyline_ids'])

    def test_source_reuse_does_not_connect_a_distant_turn(self):
        ir = fixture([
            segment('a', (10, 10), (30, 10), channel='horizontal', source='straight', aliases=('corner',)),
            segment('b', (50, 10), (50, 30), channel='vertical', source='corner'),
        ])
        self.assertEqual([], collect_segment_connections(ir.segments, endpoint_tolerance_px=1.0))
        ir.physical_nets, _ = build_physical_graph(ir, endpoint_tolerance_px=1.0)
        self.assertEqual(2, len(ir.physical_nets))

    def test_nearby_sources_are_not_merged_as_exact_duplicates(self):
        ir = fixture([
            segment('a', (10, 10), (30, 10), channel='horizontal', source='straight'),
            segment('b', (10, 11), (30, 11), channel='horizontal', source='corner'),
            segment('c', (10, 10), (30, 10), channel='vertical', source='cross'),
        ])
        repaired = repair_ir_once(ir)
        self.assertEqual(3, len(repaired.segments))
        self.assertTrue(all(not s.source_polyline_ids for s in repaired.segments))
        self.assertFalse(repaired.repairs)

    def test_dedup_sources_and_repair_are_deterministic(self):
        originals = [
            segment('weak', (10, 10), (30, 10), channel='horizontal', source='corner', score=0.7),
            segment('strong', (30, 10), (10, 10), channel='horizontal', source='straight', score=0.9),
            segment('turn', (30, 10), (30, 30), channel='vertical', source='corner'),
        ]
        outcomes = set()
        for order in itertools.permutations(originals):
            repaired = repair_ir_once(fixture(order))
            again = repair_ir_once(repaired)
            self.assertEqual(repaired.model_dump_json(), again.model_dump_json())
            repaired.physical_nets, _ = build_physical_graph(repaired, endpoint_tolerance_px=1.0)
            self.assertEqual(1, len(repaired.physical_nets))
            outcomes.add(tuple(sorted((s.internal_id, s.observed_polyline_ids) for s in repaired.segments)))
        self.assertEqual(1, len(outcomes))

    def test_old_ir_has_one_primary_source_without_aliases(self):
        value = WireSegment.model_validate(dict(internal_id='old', start=dict(x=1, y=1), end=dict(x=10, y=1), source_polyline_id='primary'))
        self.assertEqual([], value.source_polyline_ids)
        self.assertEqual(('primary',), value.observed_polyline_ids)


if __name__ == '__main__':
    unittest.main()
