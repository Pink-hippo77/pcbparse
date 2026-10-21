"""Shared conventional reference prefixes for classification and binding."""
from __future__ import annotations

import re

DESIGNATOR_PATTERN = r"^[A-Z]{1,8}[0-9]+[A-Z]?$"
BOX_REFERENCE_PATTERN = r"(?:U|IC|J|P|H|CN|CON|JP|USB|SW)\d+[A-Z]?$"

_PREFIXES = {
    "BOX": ("*",), "BOXIC": ("*",), "BOX_IC": ("*",),
    "RESISTOR": ("R",), "CAPACITOR": ("C",), "INDUCTOR": ("L",),
    "DIODE": ("D",), "LED": ("D", "LED"), "IC": ("U",),
    "CONNECTOR": ("J",), "CON": ("J",), "JACK": ("J",),
    "PLUG": ("J",), "SOCKET": ("J",), "PORT": ("J",),
    "HEADER": ("J",), "TERMINAL": ("J",),
    "SWITCH": ("SW", "S"), "RELAY": ("K",),
    "TRANSISTOR": ("Q",), "MOSFET": ("Q",), "BJT": ("Q",),
    "NPN": ("Q",), "PNP": ("Q",),
    "CRYSTAL": ("Y", "X"), "XTAL": ("Y", "X"),
    "FUSE": ("F",), "TESTPOINT": ("TP",), "PIN": ("P", "TP"),
    "VOLTAGE": ("U",), "REGULATOR": ("U",), "POWER": ("P",),
    "BATTERY": ("B",), "MOTOR": ("M",),
    "OTHER": ("O", "T"), "M3螺丝": ("SCREW", "H"),
    "BUZZER": ("B", "LS"),
}


def prefix_compatible(category: str, prefix: str) -> bool:
    normalized = category.upper()
    expected = _PREFIXES.get(normalized, (normalized[:1],))
    return expected == ("*",) or prefix.startswith(expected)


def designator_compatible(category: str, designator: str) -> bool:
    prefix = re.match(r"^[A-Z]+", designator)
    return prefix is not None and prefix_compatible(category, prefix.group(0))
