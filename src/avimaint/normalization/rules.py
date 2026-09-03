"""Conservative, auditable normalization-rule baseline."""

from __future__ import annotations

import re
from pathlib import Path

import yaml

from avimaint.normalization.protected import extract_protected


def load_replacements(path: str | Path | None) -> dict[str, str]:
    if path is None or not Path(path).exists():
        return {}
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    replacements = payload.get("replacements", {})
    if not isinstance(replacements, dict):
        raise ValueError("Dictionary file must contain a replacements mapping")
    return {str(key).lower(): str(value) for key, value in replacements.items()}


def normalize_rules(text: str, replacements: dict[str, str] | None = None) -> str:
    replacements = replacements or {}
    normalized = str(text)
    protected_values = sorted(set(extract_protected(normalized)), key=len, reverse=True)
    placeholders: dict[str, str] = {}
    for index, value in enumerate(protected_values):
        marker = f"PROTECTEDTOKEN{index}"
        normalized = re.sub(re.escape(value), marker, normalized, flags=re.IGNORECASE)
        placeholders[marker] = value
    # Longest-first phrase replacement supports forms such as "ref man" and
    # token forms such as L/H without substring collisions.
    for source, target in sorted(replacements.items(), key=lambda item: len(item[0]), reverse=True):
        if source in {'"', '#', '&', "'", '*', '+', '=', '>', '@'}:
            continue
        normalized = re.sub(rf"(?<!\w){re.escape(source)}(?!\w)", target, normalized,
                            flags=re.IGNORECASE)
    normalized = re.sub(r"#\s*(\d+)", r"number \1", normalized)
    for symbol in ('"', '&', "'", '*', '+', '=', '>', '@'):
        if symbol in replacements:
            normalized = normalized.replace(symbol, f" {replacements[symbol]} ")
    # Amin's reference corpus and the source work orders use uppercase text.
    # Lowercasing here made every otherwise-correct token count as an error in
    # the case-sensitive WER/CER evaluation and also shifted the hybrid input
    # away from the distribution used to fine-tune ByT5.
    normalized = normalized.upper()
    for marker, value in placeholders.items():
        normalized = normalized.replace(marker, value)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    normalized = re.sub(r"\s+([,.;:!?])", r"\1", normalized)
    return normalized
