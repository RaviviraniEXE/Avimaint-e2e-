"""Canonicalisation, action/fault families, and lexicon-based structure.

This module turns raw maintenance text into the derived comparison concepts the
dashboard needs (component, fault/issue, action family, role, outcome polarity).

Two sources of structure are supported:
  * SpERT predictions (preferred, high quality) -- consumed in extraction.py.
  * A deterministic lexicon fallback here -- so the dashboard runs on the raw
    CSV alone, before SpERT is wired in.

Nothing here changes the frozen SpERT schema; these are derived, runtime-only
concepts (exactly the "action_profile derived at runtime" the dataset config
describes).
"""
from __future__ import annotations

import re
from functools import lru_cache

# --------------------------------------------------------------------------- #
# Action families (same seven published families used by the frozen system).
# --------------------------------------------------------------------------- #
ACTION_FAMILIES = ("Replace", "Repair", "Inspect", "Adjust", "Service", "Diagnose", "Calibrate")

_FAMILY_PATTERNS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("Replace", (r"replac", r"\brenew", r"\bswap", r"\bchang(e|ed|ing)", r"\binstall", r"remov(e|ed|al).*(replac|install|new)", r"\bnew\b")),
    ("Calibrate", (r"calibrat", r"\brig\b", r"\brigg", r"\btime[d]?\b", r"\bre-?time", r"synchron")),
    ("Inspect", (r"inspect", r"borescop", r"\bvisual", r"examin", r"\bcheck(ed|ing)?\b.*(crack|wear|condition|security)", r"\ble?a?k check")),
    ("Diagnose", (r"diagnos", r"troubleshoot", r"\btest", r"\bcheck", r"measure", r"\bconfirm", r"verif", r"evaluat", r"analy[sz]", r"ground run", r"run.?up", r"\bfound\b", r"\btrace", r"isolat", r"duplicat")),
    ("Adjust", (r"adjust", r"\balign", r"\bgap", r"torque", r"tighten", r"clearance", r"\bset\b", r"reposition", r"secur")),
    ("Service", (r"servic", r"clean", r"lubricat", r"\blube", r"greas", r"drain", r"refill", r"\bfill", r"bleed", r"flush", r"safet(y|ie)", r"lock.?wire", r"seal(ed|ing)?\b", r"paint", r"apply", r"treat")),
    ("Repair", (r"repair", r"weld", r"reswedg", r"resweg", r"patch", r"overhaul", r"rebuild", r"reseat", r"restor", r"fabricat", r"\bfix", r"rework", r"reglu")),
)

# corrective vs diagnostic role for each family
_DIAGNOSTIC_FAMILIES = {"Inspect", "Diagnose"}


@lru_cache(maxsize=8192)
def action_family(text: str) -> str:
    """Return the best-matching action family for an action/verb string."""
    low = (text or "").lower()
    if not low.strip():
        return "Other"
    for family, patterns in _FAMILY_PATTERNS:
        for pat in patterns:
            if re.search(pat, low):
                return family
    return "Other"


def action_role(family: str) -> str:
    return "diagnostic" if family in _DIAGNOSTIC_FAMILIES else "corrective"


# --------------------------------------------------------------------------- #
# Issue (fault / abnormal-process) lexicon -> canonical fault family.
# --------------------------------------------------------------------------- #
_ISSUE_CANON: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("leak", (r"leak", r"seep", r"weep")),
    ("crack", (r"crack", r"fractur", r"split")),
    ("wear", (r"\bworn", r"\bwear\b", r"eroded", r"pitt", r"scor(e|ed|ing)", r"galled")),
    ("loose", (r"loose", r"\bplay\b", r"backlash", r"wobbl")),
    ("corrosion", (r"corro", r"rust", r"oxidi")),
    ("burnt", (r"burn", r"scorch", r"charred", r"overheat", r"\bhot\b")),
    ("broken", (r"broke", r"\bbroken", r"snapp", r"separat", r"fail", r"\bbent", r"deform", r"collaps")),
    ("low compression", (r"low compress", r"no compress", r"lost compress", r"weak cylinder")),
    ("rough running", (r"rough", r"misfir", r"miss(ing|es|ed)?\b", r"backfir", r"surg", r"sputter", r"stumbl", r"hesitat")),
    ("vibration", (r"vibrat", r"shak", r"shudder")),
    ("power loss", (r"power loss", r"los(e|es|ing|t) power", r"low power", r"partial power", r"won.?t make power")),
    ("will not start", (r"won.?t start", r"will not start", r"no start", r"hard start", r"fail.*start", r"crank.*no")),
    ("quit / shutdown", (r"\bquit", r"shut ?down", r"died", r"cut ?out", r"stall")),
    ("noise", (r"noise", r"nois", r"knock", r"grind", r"squeal", r"clunk", r"tick")),
    ("smoke", (r"smok",)),
    ("contamination", (r"contaminat", r"debris", r"metal", r"blocked", r"clog", r"restrict", r"plugged")),
    ("chafing", (r"chaf", r"rub", r"fray")),
    ("stuck / binding", (r"stuck", r"seiz", r"bind", r"jam", r"frozen")),
    ("inoperative", (r"inop", r"inoperativ", r"no output", r"not work", r"dead", r"intermittent", r"erratic", r"fluctuat")),
    ("missing / damaged", (r"missing", r"damag", r"\bhole", r"punctur", r"\btorn", r"\bnick", r"gouge", r"dent")),
    ("high reading", (r"\bhigh\b", r"excessive", r"over ?temp", r"over ?press")),
    ("low reading", (r"\blow\b", r"insufficient", r"\bweak\b")),
)


@lru_cache(maxsize=8192)
def issue_family(text: str) -> str | None:
    low = (text or "").lower()
    if not low.strip():
        return None
    for canon, patterns in _ISSUE_CANON:
        for pat in patterns:
            if re.search(pat, low):
                return canon
    return None


# --------------------------------------------------------------------------- #
# Component lexicon (aviation piston-maintenance vocabulary). Multi-word first.
# Used only for the no-SpERT fallback; SpERT MAINT_ITEM entities override it.
# --------------------------------------------------------------------------- #
_COMPONENT_TERMS = [
    # multiword (checked first, longest-first sorting below)
    "intake gasket", "exhaust gasket", "rocker cover gasket", "valve cover gasket",
    "fuel servo", "fuel pump", "fuel injector", "fuel line", "fuel filter", "fuel tank",
    "fuel selector", "fuel cap", "fuel nozzle", "primer line",
    "oil filter", "oil cooler", "oil line", "oil seal", "oil pump", "oil screen",
    "spark plug", "exhaust valve", "intake valve", "exhaust pipe", "exhaust stack",
    "exhaust system", "exhaust manifold", "cylinder head", "engine mount", "motor mount",
    "vacuum pump", "vacuum system", "air filter", "induction hose", "induction system",
    "landing gear", "nose gear", "main gear", "nose wheel", "brake disc", "brake pad",
    "brake line", "brake caliper", "master cylinder", "wheel bearing", "tail wheel",
    "propeller governor", "prop governor", "prop hub", "spinner bulkhead",
    "magneto drive", "ignition harness", "ignition switch", "ignition lead",
    "starter adapter", "voltage regulator", "alternator belt", "cabin heat",
    "carburetor heat", "mixture control", "throttle cable", "control cable",
    "trim tab", "wing tip", "static wick", "pitot tube", "landing light", "nav light",
    "battery box", "primer pump", "gascolator", "push rod", "push rod tube",
    "valve guide", "valve spring", "valve seat", "piston ring", "connecting rod",
    "crankcase half", "cylinder barrel", "baffle seal", "engine baffle",
    # single word
    "engine", "cylinder", "piston", "valve", "gasket", "magneto", "mag", "carburetor",
    "carburettor", "carb", "exhaust", "muffler", "intake", "manifold", "baffle", "baffles",
    "hose", "seal", "bearing", "crankshaft", "camshaft", "alternator", "generator",
    "starter", "battery", "propeller", "prop", "governor", "tire", "tyre", "brake",
    "strut", "gear", "injector", "nozzle", "filter", "line", "fitting", "clamp", "bracket",
    "mount", "tube", "cable", "switch", "sensor", "gauge", "light", "antenna", "door",
    "window", "windshield", "seat", "belt", "actuator", "pump", "tank", "cap", "screw",
    "bolt", "nut", "fastener", "rivet", "panel", "cowl", "cowling", "fairing", "flap",
    "aileron", "rudder", "elevator", "trim", "spinner", "hub", "boot", "duct", "vent",
    "thermostat", "regulator", "solenoid", "relay", "breaker", "wire", "harness",
    "connector", "terminal", "plug", "coil", "distributor", "pushrod", "rocker",
    "tappet", "lifter", "sump", "cowl flap", "primer", "throttle", "mixture", "governor",
    "spar", "rib", "skin", "bulkhead", "firewall", "engine", "prop", "wheel", "axle",
    "diaphragm", "float", "needle", "jet", "impulse coupling", "points", "condenser",
    "vacuum", "gyro", "compass", "transponder", "radio", "headset", "microphone",
    "windscreen", "canopy", "hinge", "latch", "spring", "shim", "washer", "grommet",
    "bushing", "o-ring", "packing", "retainer", "circlip", "cotter", "safety wire",
]
# sort longest first so multiword wins
_COMPONENT_TERMS_SORTED = sorted(set(_COMPONENT_TERMS), key=lambda s: (-len(s.split()), -len(s)))
_COMPONENT_RE = re.compile(
    r"\b(" + "|".join(re.escape(t) for t in _COMPONENT_TERMS_SORTED) + r")\b",
    re.IGNORECASE,
)

# canonical merges for near-synonyms
_COMPONENT_CANON = {
    "mag": "magneto", "carb": "carburetor", "carburettor": "carburetor",
    "prop": "propeller", "tyre": "tire", "baffles": "baffle", "pushrod": "push rod",
    "motor mount": "engine mount", "carburetor heat": "carburetor",
    "valve cover gasket": "rocker cover gasket",
}


def canonical_component(surface: str) -> str:
    s = re.sub(r"\s+", " ", (surface or "").strip().lower())
    return _COMPONENT_CANON.get(s, s)


def find_components(text: str) -> list[str]:
    """Ordered, de-duplicated canonical components mentioned in text."""
    seen: dict[str, None] = {}
    for m in _COMPONENT_RE.finditer(text or ""):
        c = canonical_component(m.group(0))
        seen.setdefault(c, None)
    return list(seen.keys())


# --------------------------------------------------------------------------- #
# Location / positional designators.
# --------------------------------------------------------------------------- #
_LOC_RE = re.compile(
    r"\b(#\s?\d+|no\.?\s?\d+|number\s?\d+|cyl(?:inder)?\s?\d+|"
    r"l\.?h\.?|r\.?h\.?|left|right|front|rear|fwd|forward|aft|"
    r"upper|lower|inboard|outboard|top|bottom|nose|tail|port|starboard)\b",
    re.IGNORECASE,
)
_LOC_CANON = {
    "lh": "left", "l.h.": "left", "rh": "right", "r.h.": "right",
    "fwd": "forward", "port": "left", "starboard": "right",
}


def find_locations(text: str) -> list[str]:
    out: dict[str, None] = {}
    for m in _LOC_RE.finditer(text or ""):
        s = re.sub(r"\s+", " ", m.group(0).strip().lower())
        s = _LOC_CANON.get(s.replace(" ", ""), s)
        s = re.sub(r"^(no\.?|number)\s*", "#", s)
        out.setdefault(s, None)
    return list(out.keys())


# --------------------------------------------------------------------------- #
# Outcome polarity from action/solution text.
# --------------------------------------------------------------------------- #
_POS = (r"\bgood\b", r"\bok\b", r"okay", r"satisfactor", r"serviceable", r"no defect",
        r"no leak", r"no further", r"correct(ed|ly)?\b", r"resolv", r"\brepaired\b",
        r"operational", r"within limit", r"\bnormal\b", r"complet", r"signed off",
        r"return.*service", r"ops? ?check.*good", r"ground.?check.*good", r"leak ?check.*good",
        r"check.*good", r"\bpass", r"functions? ?normal")
_NEG = (r"\bstill\b", r"could not", r"couldn.?t", r"unable", r"unresolv", r"persist",
        r"no effect", r"not duplicat", r"cannot duplicat", r"could not duplicat",
        r"\bfail", r"defer", r"inop", r"recurr", r"replace again", r"no improve", r"worse")
_POS_RE = re.compile("|".join(_POS), re.IGNORECASE)
_NEG_RE = re.compile("|".join(_NEG), re.IGNORECASE)


def outcome_polarity(action_text: str) -> str:
    low = action_text or ""
    neg = bool(_NEG_RE.search(low))
    pos = bool(_POS_RE.search(low))
    if neg and not pos:
        return "negative"
    if pos and not neg:
        return "positive"
    if pos and neg:
        return "mixed"
    return "unknown"


# --------------------------------------------------------------------------- #
# Problem-text normalisation for near-duplicate clustering.
# --------------------------------------------------------------------------- #
_WS = re.compile(r"\s+")
_PUNCT = re.compile(r"[^\w\s#]")


def normalize_problem(text: str) -> str:
    s = (text or "").lower().strip()
    s = _PUNCT.sub(" ", s)
    s = _WS.sub(" ", s).strip()
    return s

