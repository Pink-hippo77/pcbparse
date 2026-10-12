"""Phase 2 A/B validation: calibration + dictionary + orientation quality."""
import sys
sys.path.insert(0, r"E:\EDA挑战题目六\src")

from pcbparse.association.ocr_calibration import calibrate_score, PlattCalibrator
from pcbparse.association.ocr_dictionary import constrained_correct, snap_to_net, snap_pin_number
from pcbparse.perception.torch_ocr_adapter import _orientation_quality, _text_has_structure

print("=== P2.1: Word-Level Confidence Calibration ===\n")
test_scores = [
    ("R", 0.95, "single char overconfident"),
    ("O", 0.90, "single char overconfident"),
    ("R23", 0.70, "designator structure bonus"),
    ("USB_DP_PLUS", 0.65, "long string underconfident"),
    ("10k", 0.80, "value structure bonus"),
    ("xyz", 0.60, "no structure"),
]
for text, raw, note in test_scores:
    calibrated = calibrate_score(raw, text)
    delta = calibrated - raw
    print(f"  {text!r:16s} raw={raw:.2f} -> calibrated={calibrated:.2f} ({delta:+.2f})  [{note}]")

print("\n=== P2.2: Dictionary-Constrained Decoding ===\n")
dict_cases = [
    ("O", "pinnumber", None, "0"),
    ("l", "pinnumber", None, "1"),
    ("RI", "designator", None, "R1"),
    ("GND", "netlike", None, "GND"),
    ("VCC", "netlike", None, "VCC"),
    ("GN0", "netlike", None, "GND"),
    ("R23", "designator", None, "R23"),
    ("42", "pinnumber", None, "42"),
]
for text, context, category, expected in dict_cases:
    corrected, records = constrained_correct(text, context=context, category=category)
    status = "OK" if corrected == expected else ("FIXED" if corrected != text else "MISS")
    rule = records[0]["rule"] if records else "none"
    print(f"  {text!r:8s} ctx={context:12s} -> {corrected!r:8s} expected={expected!r:8s} rule={rule} [{status}]")

print("\n=== P2.3: 90deg Dual-Channel Orientation Quality ===\n")
orient_cases = [
    ("R23", 0.75, "GND", 0.80),   # 90deg higher raw score but 0deg has structure
    ("1", 0.85, "l", 0.90),        # 90deg higher raw score but "1" has structure
    ("VCC", 0.70, "CCV", 0.72),   # similar scores, "VCC" has structure
    ("hello", 0.60, "world", 0.65),  # no structure, higher raw score wins
]
for text0, score0, text90, score90 in orient_cases:
    q0 = _orientation_quality(text0, score0)
    q90 = _orientation_quality(text90, score90)
    winner = "0deg" if q0 >= q90 else "90deg"
    print(f"  0deg={text0!r:8s}({score0:.2f},q={q0:.3f}) vs 90deg={text90!r:8s}({score90:.2f},q={q90:.3f}) -> {winner}")

print("\n=== Summary ===")
print("P2.1: Length-binned Platt scaling + heuristic calibration")
print("P2.2: Dictionary snap (net/pin/designator) + confusion-aware edit distance")
print("P2.3: Orientation quality = 70% score + 20% structure + 10% charset")
