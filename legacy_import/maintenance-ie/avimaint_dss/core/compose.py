"""Conservative, source-grounded recommendation sentence composition."""
from __future__ import annotations
import re

_FAMILY_VERB = {
    "Replace": "Replace {t}", "Repair": "Repair {t}", "Adjust": "Adjust {t}",
    "Service": "Service {t}", "Inspect": "Inspect {t}",
    "Diagnose": "Diagnose {t}", "Calibrate": "Calibrate {t}",
}
_FAMILY_MEANING = {
    "Replace": "Replace the affected component, consistent with recorded historical actions.",
    "Repair": "Repair the affected component, consistent with recorded historical actions.",
    "Adjust": "Adjust the affected component, consistent with recorded historical actions.",
    "Service": "Service the affected component, consistent with recorded historical actions.",
    "Inspect": "Inspect the affected component before selecting a corrective action.",
    "Diagnose": "Perform diagnostic work to isolate the reported issue.",
    "Calibrate": "Calibrate the affected component, consistent with recorded historical actions.",
    "Other": "Recorded historical action (family not classified).",
}
_VERIFY_RE = re.compile(r"leak\s*(ck|check)|ops\s*(ck|check)|ground\s*(ck|check)|run\s*up|checked good|check good|ops good|functional check", re.I)
_FAULT_ADJ = {"leak":"leaking","crack":"cracked","wear":"worn","loose":"loose","corrosion":"corroded","burnt":"burnt","broken":"broken","chafing":"chafed","vibration":"vibrating"}
_FAULT_NOUN = {"leak":"leak","crack":"crack","wear":"wear","loose":"looseness","corrosion":"corrosion","vibration":"vibration","noise":"noise","low compression":"low compression","rough running":"rough running","power loss":"power loss","will not start":"no-start condition","quit / shutdown":"shutdown","smoke":"smoke"}

def build_target(components:list[str], locations:list[str], fallback:str="the affected component") -> str:
    comp = components[0] if components else ""
    loc = " ".join(locations[:1]) if locations else ""
    if comp and loc: return f"the {loc} {comp}".replace("  "," ")
    if comp: return f"the {comp}"
    if loc: return f"the {loc} component"
    return fallback

def with_fault(target:str, fault:str|None) -> str:
    if not fault: return target
    adj=_FAULT_ADJ.get(fault)
    if not adj or adj in target: return target
    m=re.match(r"^(the\s+)(.*)$", target)
    return f"{m.group(1)}{adj} {m.group(2)}" if m else f"{adj} {target}"

def compose_sentence(family:str, target:str, fault:str|None=None, has_verification:bool=False) -> str:
    verb=_FAMILY_VERB.get(family)
    if not verb: return ""
    tgt=with_fault(target,fault) if family in ("Replace","Repair","Service") else target
    sentence=verb.format(t=tgt).strip()
    if family=="Inspect":
        noun=_FAULT_NOUN.get(fault or "")
        if noun: sentence += f" for evidence of {noun}"
    elif family=="Diagnose":
        noun=_FAULT_NOUN.get(fault or "")
        if noun: sentence += f" to isolate the {noun}"
    if has_verification and family in ("Replace","Repair","Adjust","Service"):
        sentence += ", then perform a verification check"
    return sentence.rstrip(".")+"."

def family_meaning(family:str)->str:
    return _FAMILY_MEANING.get(family,_FAMILY_MEANING["Other"])

def cases_have_verification(action_texts:list[str])->bool:
    return any(_VERIFY_RE.search(t or "") for t in action_texts)
