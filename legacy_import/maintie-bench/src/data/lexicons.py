"""Domain lexicons and patterns for entity-cue detection and weak pre-annotation.

Grounded in the corpus frequency profile. These seed a FIRST-DRAFT annotation
that the human corrects — they are intentionally high-precision where possible
and are NOT a substitute for gold annotation.
"""
from __future__ import annotations

import re

# ---- multi-word items first (matched before single words) -------------------
MULTIWORD_ITEMS = [
    "rocker cover gasket", "rocker box cover", "rocker cover", "intake gasket",
    "push rod tube", "push rod", "spark plug", "baffle seal", "valve cover",
    "fuel servo", "fuel injector", "oil cooler", "landing gear", "cylinder head",
    "wire harness", "hose clamp", "adel clamp", "starter generator",
]
ITEMS = [
    "gasket", "gaskets", "cylinder", "cylinders", "baffle", "baffles", "cover",
    "covers", "engine", "engines", "intake", "rocker", "seal", "seals", "screw",
    "screws", "bolt", "bolts", "nut", "nuts", "tube", "tubes", "hose", "hoses",
    "clamp", "clamps", "plug", "plugs", "valve", "valves", "rod", "rods",
    "bracket", "line", "lines", "magneto", "magnetos", "alternator", "battery",
    "exhaust", "injector", "governor", "propeller", "prop", "cowl", "cowling",
    "bearing", "filter", "fitting", "harness", "aircraft", "wire", "washer",
    "oil", "fuel", "rivet", "rivets", "baffling",
]
MATERIALS = ["rtv", "silicone", "sealant", "patch", "safety wire", "grease"]

ACTIONS = [
    "replaced", "replace", "removed", "remove", "installed", "install",
    "inspected", "inspect", "checked", "check", "cleaned", "clean", "tightened",
    "tighten", "torqued", "torque", "adjusted", "adjust", "fabricated",
    "reinstalled", "secured", "repaired", "repair", "sealed", "stop drilled",
    "riveted", "found", "ran", "performed", "timed", "lubricated", "applied",
    "serviced", "replaced", "drilled", "patched", "welded", "safetied",
]
MULTIWORD_ACTIONS = ["leak check", "ground run", "run up", "run-up", "ops check",
                     "compression check", "stop drilled", "change out"]

FAULTS = ["cracked", "crack", "loose", "broken", "break", "worn", "wear",
          "missing", "damaged", "damage", "inoperative", "inop", "stuck",
          "sheared", "bent", "corroded", "chafed", "deteriorated", "frayed"]
MULTIWORD_FAULTS = ["low compression", "no compression", "no oil pressure",
                    "free play", "low voltage"]

ABN_PROC = ["leaking", "leak", "leaks", "seeping", "vibrating", "vibration",
            "overheating", "sputtering", "chafing", "smoking", "knocking",
            "howling", "ticking", "whine", "squeal"]
MULTIWORD_PROC = ["running rough", "running very rough", "losing power",
                  "loss of power", "run rough", "rpm drop", "power loss"]

OUTCOMES = ["good", "normal", "serviceable", "operational"]
MULTIWORD_OUTCOMES = ["no leaks", "no defects noted", "no defects found",
                      "within limits", "could not duplicate", "ops check good",
                      "leak check good", "check good", "no further"]

LOC_PATTERNS = [
    (r"#\s?\d+[a-z]?", "LOC"),                      # #2, #4A
    (r"\bnumber \d+", "LOC"),                       # number 2  (post-normalization)
    (r"\br/?h\b|\bright-hand\b|\bright hand\b", "LOC"),
    (r"\bl/?h\b|\bleft-hand\b|\bleft hand\b", "LOC"),
    (r"\b(fwd|forward|aft|upper|lower|inboard|outboard|front|rear|top|bottom)\b", "LOC"),
]

TECH_PATTERNS = [
    (r"\d+\s?/\s?\d+", "TECH_OBS"),                 # 20/80, 70/80
    (r"\d+\s?(psi|rpm|volts?|degrees?|quarts?|inches?|pounds?|fahrenheit)", "TECH_OBS"),
    (r"\b(compression|oil pressure|manifold pressure|clearance|tappet|mag drop|idle speed|torque)\b", "TECH_OBS"),
]

OP_CTX_PATTERNS = [
    (r"during (climb|climbout|taxi|takeoff|run.?up|descent|approach|flight|start|cruise)", "OP_CTX"),
    (r"\bin flight\b|\bat idle\b|\bon (start|climb|takeoff|approach|ground)\b", "OP_CTX"),
    (r"\bfull (power|throttle)\b|\brun.?up\b|\bclimb out\b|\btake ?off\b|\bground run\b", "OP_CTX"),
]

REFERENCE_PATTERNS = [
    (r"\bad \d{4}-\d+-\d+\b", "REFERENCE"),
    (r"\bservice bulletin\b|\bsb \d+", "REFERENCE"),
    (r"\b(lycoming|continental|rolls royce|pratt and whitney)\b.{0,20}(manual|bulletin)?", "REFERENCE"),
    (r"\bmaintenance manual\b|\bchapter \d+\b|\biaw\b", "REFERENCE"),
]


def _word_set(words):
    return set(w.lower() for w in words)


# single-token dictionaries by entity (for cue detection + fallback tagging)
SINGLE = {
    "MAINT_ITEM": _word_set(ITEMS + MATERIALS),
    "ACTION": _word_set(ACTIONS),
    "FAULT": _word_set(FAULTS),
    "ABN_PROC": _word_set(ABN_PROC),
    "OUTCOME": _word_set(OUTCOMES),
}
MULTIWORD = {
    "MAINT_ITEM": MULTIWORD_ITEMS,
    "ACTION": MULTIWORD_ACTIONS,
    "FAULT": MULTIWORD_FAULTS,
    "ABN_PROC": MULTIWORD_PROC,
    "OUTCOME": MULTIWORD_OUTCOMES,
}
PATTERNS = {
    "LOC": LOC_PATTERNS,
    "TECH_OBS": TECH_PATTERNS,
    "OP_CTX": OP_CTX_PATTERNS,
    "REFERENCE": REFERENCE_PATTERNS,
}


def has_cue(text: str, entity: str) -> bool:
    """Cheap check: does `text` contain any cue for `entity`? Used for sampling strata."""
    low = text.lower()
    if entity in SINGLE:
        toks = set(re.findall(r"[a-z]+", low))
        if toks & SINGLE[entity]:
            return True
        return any(mw in low for mw in MULTIWORD.get(entity, []))
    if entity in PATTERNS:
        return any(re.search(p, low) for p, _ in PATTERNS[entity])
    return False

