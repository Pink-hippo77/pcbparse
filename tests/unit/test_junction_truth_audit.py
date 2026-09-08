from __future__ import annotations

import sys
import unittest
from pathlib import Path

from PIL import Image, ImageDraw


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
for import_root in (REPOSITORY_ROOT, SOURCE_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from data_tools.junction_truth_audit import (  # noqa: E402
    analyze_geometry,
    analyze_topology,
    analyze_visual_symbol,
    parse_edif_pin_nets,
    parse_spice_edge_nodes,
    review_proposal,
)


def _edge(
    skeleton: list[list[int]],
    path: list[list[int]],
    *,
    pin: str,
    location: list[int],
    component: str = "C_1",
    netlabel: str = "",
    endpoints: list[list[int]] | None = None,
) -> dict:
    return {
        "edgeId": pin.replace("P", "E"),
        "category": "wire",
        "netlabel": netlabel,
        "connect": {
            pin: {
                "pinId": pin,
                "cpntId": component,
                "location": location,
            }
        },
        "keyPoints": {
            "skeleton": {"wire": skeleton},
            "endPoints": {"wire": endpoints or []},
            "divergePoints": {"wire": []},
            "turnPoints": {"wire": []},
        },
        "keyPointPaths": {"wire": [path]},
        # The audit must not need this dense member.
        "coordinates": "forbidden-sentinel",
    }


class JunctionTruthAuditTests(unittest.TestCase):
    def test_sparse_geometry_separates_collinear_overlap_from_cross(self) -> None:
        horizontal = [[x, 16] for x in range(8, 25)]
        vertical = [[16, y] for y in range(8, 25)]
        duplicate = [[x, 16] for x in range(8, 25)]
        edges = {
            "E_1": _edge(horizontal, [[8, 16], [24, 16]], pin="P_1", location=[8, 16]),
            "E_2": _edge(vertical, [[16, 8], [16, 24]], pin="P_2", location=[16, 8]),
            "E_3": _edge(duplicate, [[8, 16], [24, 16]], pin="P_3", location=[24, 16]),
        }

        crossing = analyze_geometry(edges, ["E_1", "E_2"], (16.0, 16.0))
        overlap = analyze_geometry(edges, ["E_1", "E_3"], (16.0, 16.0))

        self.assertEqual("orthogonal_intersection", crossing["shape"])
        self.assertEqual(1, crossing["shared_component_pixels"])
        self.assertEqual("collinear_overlap", overlap["shape"])
        self.assertTrue(overlap["identical_skeleton_sets"])
        self.assertGreater(overlap["shared_component_pixels"], 1)

    def test_edif_and_spice_topology_are_auditable_separate_channels(self) -> None:
        edif = """
        (edif X
          (net N1
            (joined
              (portRef PORT_P_1 (instanceRef INST_C_1))
              (portRef PORT_P_2 (instanceRef INST_C_2))))
          (net N2
            (joined (portRef PORT_P_3 (instanceRef INST_C_3)))))
        """
        pin_nets = parse_edif_pin_nets(edif)
        spice_nodes = parse_spice_edge_nodes(
            "* E_1 = SIGNAL : P_1(C_1)\n* E_2 = SIGNAL : P_2(C_2)\n"
        )
        edges = {
            "E_1": _edge([[16, 16]], [[8, 16], [24, 16]], pin="P_1", location=[8, 16]),
            "E_2": _edge([[16, 16]], [[16, 8], [16, 24]], pin="P_2", location=[16, 8]),
        }

        topology = analyze_topology(edges, ["E_1", "E_2"], pin_nets, spice_nodes)

        self.assertEqual("same", topology["edif_pin_net_relation"])
        self.assertEqual("same", topology["spice_edge_node_relation"])
        self.assertEqual("connected", topology["consensus"])
        self.assertFalse(topology["raw_connect"]["shared_pin_id"])

    def test_visual_classifier_distinguishes_plain_dot_and_gap(self) -> None:
        def canvas() -> tuple[Image.Image, ImageDraw.ImageDraw]:
            image = Image.new("RGB", (64, 64), "white")
            return image, ImageDraw.Draw(image)

        plain, draw = canvas()
        draw.line((8, 32, 56, 32), fill="black", width=1)
        draw.line((32, 8, 32, 56), fill="black", width=1)
        dotted, draw = canvas()
        draw.line((8, 32, 56, 32), fill="black", width=1)
        draw.line((32, 8, 32, 56), fill="black", width=1)
        draw.ellipse((27, 27, 37, 37), fill="black")
        gap, draw = canvas()
        draw.line((8, 32, 56, 32), fill="black", width=1)
        draw.line((32, 8, 32, 27), fill="black", width=1)
        draw.line((32, 37, 32, 56), fill="black", width=1)

        self.assertEqual("plain_intersection", analyze_visual_symbol(plain)["symbol"])
        self.assertEqual("filled_dot", analyze_visual_symbol(dotted)["symbol"])
        self.assertEqual("bridge_or_gap", analyze_visual_symbol(gap)["symbol"])

    def test_review_rule_never_promotes_a_proposal_to_ground_truth(self) -> None:
        geometry = {
            "shape": "orthogonal_intersection",
            "identical_skeleton_sets": False,
            "raw_keypoint_at_center": {},
        }
        topology = {
            "consensus": "connected",
            "negative_channel_count": 0,
        }
        visual = {"symbol": "filled_dot"}

        proposal = review_proposal(["cross"], geometry, topology, visual)

        self.assertEqual("connected_cross", proposal["proposed_label"])
        self.assertEqual("medium", proposal["confidence"])
        self.assertFalse(proposal["eligible_for_review_auto_accept"])
        self.assertFalse(proposal["accepted_as_ground_truth"])
        self.assertFalse(proposal["may_enter_training_without_human_review"])


if __name__ == "__main__":
    unittest.main()
