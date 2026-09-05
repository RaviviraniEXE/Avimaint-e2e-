from __future__ import annotations

import json
import urllib.request
from dataclasses import dataclass
from functools import lru_cache

from .hybrid_normalization import validate_hybrid_candidate


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
            with urllib.request.urlopen(self.url + "/health", timeout=3.0) as response:
                metadata = json.loads(response.read().decode("utf-8"))
            if (
                metadata.get("status") == "ready"
                and metadata.get("role") == "operational_rules_then_byt5_normalizer"
                and metadata.get("representation") == "rules_then_byt5_guarded_operational"
                and metadata.get("decoding_strategy") == "greedy_deterministic"
                and int(metadata.get("num_beams", 0)) == 1
            ):
                return metadata
        except Exception:
            pass
        return None

    @lru_cache(maxsize=512)
    def normalize(self, text: str) -> NormalizationResult:
        original = str(text or "").strip()
        if not original:
            return NormalizationResult(original, original, original, "none", False, False)
        if not self.enabled:
            return NormalizationResult(
                original, original, original, "disabled", False, False,
                "Hybrid normalization is disabled by the runtime model lock.", "",
            )
        try:
            body = json.dumps({"text": original}).encode("utf-8")
            request = urllib.request.Request(
                self.url + "/normalize", data=body, headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                data = json.loads(response.read().decode("utf-8"))
        except Exception as exc:
            return NormalizationResult(
                original, original, original, "service_unavailable", False, False,
                f"Normalization service unavailable: {type(exc).__name__}", "",
            )

        rule_text = str(data.get("rule_normalized", "") or "").strip()
        candidate = str(data.get("candidate_normalized", "") or "").strip()
        service_accept = bool(data.get("accepted_for_semantic_spert", False))
        client_accept, client_problems = validate_hybrid_candidate(original, rule_text, candidate)
        accepted = bool(service_accept and client_accept)
        normalized = str(data.get("normalized", original) or original).strip() if accepted else original
        warnings = list(data.get("warnings", []) or [])
        warnings.extend(client_problems)
        if not accepted and not warnings:
            warnings.append("Hybrid candidate was not approved for semantic SpERT.")
        return NormalizationResult(
            original=original,
            model_input=rule_text or original,
            normalized=normalized,
            method=str(data.get("method", "rules_then_byt5_guarded") or "rules_then_byt5_guarded"),
            accepted_for_semantic_spert=accepted,
            changed=bool(accepted and normalized != original),
            warning="; ".join(dict.fromkeys(str(item) for item in warnings if str(item))),
            model=str(data.get("model", "") or ""),
        )
