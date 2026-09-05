
from __future__ import annotations
import json, re, urllib.request
from dataclasses import dataclass
from functools import lru_cache

_NUM = re.compile(r"(?<![A-Za-z])#?\d+(?:\.\d+)?(?:/\d+(?:\.\d+)?)?(?![A-Za-z])")
_PART = re.compile(r"\b(?:P/?N|S/?N|PART\s+NO\.?|SERIAL\s+NO\.?)\s*[:#-]?\s*[A-Z0-9._/-]+", re.I)

def ascii_upper(text: str) -> str:
    return str(text).translate(str.maketrans(
        "abcdefghijklmnopqrstuvwxyz", "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    ))

def number_set(text: str) -> set[str]:
    return {m.group(0).lower().lstrip("#") for m in _NUM.finditer(text or "")}

def part_set(text: str) -> set[str]:
    return {m.group(0).lower() for m in _PART.finditer(text or "")}

def validate_candidate(original: str, candidate: str) -> tuple[bool, str]:
    candidate = str(candidate or "").strip()
    if not candidate:
        return False, "ByT5 returned empty text."
    missing_num = sorted(number_set(original) - number_set(candidate))
    if missing_num:
        return False, "Protected numeric value(s) were not preserved: " + ", ".join(missing_num)
    missing_part = sorted(part_set(original) - part_set(candidate))
    if missing_part:
        return False, "Protected part/serial identifier(s) were not preserved: " + ", ".join(missing_part)
    if len(candidate) > max(5000, len(str(original)) * 4 + 80):
        return False, "ByT5 rewrite was implausibly long."
    return True, ""

@dataclass(frozen=True)
class NormalizationResult:
    original: str
    model_input: str
    normalized: str
    method: str
    accepted_for_semantic_spert: bool
    changed: bool
    warning: str = ""
    model: str = ""

class NormalizationClient:
    def __init__(self, url="http://127.0.0.1:8766", timeout=45.0, enabled=True):
        self.url = str(url or "").rstrip("/")
        self.timeout = float(timeout)
        self.enabled = bool(enabled)

    def health(self):
        if not self.enabled:
            return None
        try:
            with urllib.request.urlopen(self.url + "/health", timeout=3.0) as r:
                return json.loads(r.read().decode("utf-8"))
        except Exception:
            return None

    @lru_cache(maxsize=512)
    def normalize(self, text: str) -> NormalizationResult:
        original = str(text or "").strip()
        model_input = ascii_upper(original)
        if not original:
            return NormalizationResult(original, model_input, original, "none", False, False)
        if not self.enabled:
            return NormalizationResult(
                original,
                model_input,
                original,
                "disabled",
                False,
                False,
                "ByT5 normalization is disabled by the runtime model lock.",
                "",
            )
        try:
            body = json.dumps({"text": original}).encode("utf-8")
            req = urllib.request.Request(
                self.url + "/normalize",
                data=body,
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                data = json.loads(r.read().decode("utf-8"))
        except Exception as exc:
            return NormalizationResult(
                original, model_input, original, "service_unavailable", False, False,
                f"Normalization service unavailable: {type(exc).__name__}", ""
            )

        candidate = str(data.get("normalized", "")).strip()
        ok, warning = validate_candidate(original, candidate)
        if not ok:
            return NormalizationResult(
                original, str(data.get("model_input", model_input)), original,
                "guard_rejected", False, False, warning,
                str(data.get("model", "") or "")
            )
        return NormalizationResult(
            original=original,
            model_input=str(data.get("model_input", model_input)),
            normalized=candidate,
            method="byt5_guarded_deployment",
            accepted_for_semantic_spert=True,
            changed=(candidate != original),
            warning=str(data.get("warning", "") or ""),
            model=str(data.get("model", "") or ""),
        )
