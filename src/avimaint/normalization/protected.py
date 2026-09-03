"""Protected-token extraction and output validation."""

from __future__ import annotations

import re
from dataclasses import dataclass

DEFAULT_PATTERNS = [
    r"\b[A-Z]{1,5}[-/]?\d[A-Z0-9./-]*\b",
    r"\b\d+(?:\.\d+)?\s?(?:V|A|PSI|PSIG|IN|MM|CM|NM|LB|LBS|HR|HRS|MIN|SEC|DEG|°C|°F)\b",
    r"\b(?:LEFT|RIGHT|LH|RH|NO|NOT|NEVER)\b",
]

ORIENTATION_EQUIVALENTS = {
    "lh": ("LH", "L/H", "LEFT-HAND", "LEFT HAND"),
    "rh": ("RH", "R/H", "RIGHT-HAND", "RIGHT HAND"),
    "left": ("LEFT", "LEFT-HAND", "LEFT HAND", "LH", "L/H"),
    "right": ("RIGHT", "RIGHT-HAND", "RIGHT HAND", "RH", "R/H"),
}

NUMBER_WORDS = {
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
    "eleven": "11",
    "twelve": "12",
    "thirteen": "13",
    "fourteen": "14",
    "fifteen": "15",
    "sixteen": "16",
    "seventeen": "17",
    "eighteen": "18",
    "nineteen": "19",
    "twenty": "20",
}


@dataclass(frozen=True)
class ProtectionResult:
    accepted: bool
    missing_tokens: tuple[str, ...]


def extract_protected(text: str, patterns: list[str] | None = None) -> list[str]:
    found: list[str] = []
    for pattern in patterns or DEFAULT_PATTERNS:
        found.extend(match.group(0) for match in re.finditer(pattern, text, flags=re.IGNORECASE))
    return found


def validate_protected(
    source: str, candidate: str, patterns: list[str] | None = None
) -> ProtectionResult:
    expected = extract_protected(source, patterns)
    def contains_form(form: str) -> bool:
        return bool(
            re.search(
                rf"(?<![A-Z0-9]){re.escape(form)}(?![A-Z0-9])",
                candidate,
                flags=re.IGNORECASE,
            )
        )

    missing_values: list[str] = []
    for token in expected:
        forms = ORIENTATION_EQUIVALENTS.get(token.lower(), (token,))
        if not any(contains_form(form) for form in forms):
            missing_values.append(token)
    # A lexical normalizer may rewrite a grounded value (for example, #3 to
    # NUMBER THREE) but must never introduce a numeric value absent from the
    # source. This catches unsupported completions without rejecting benign
    # abbreviation expansion or punctuation changes.
    def numeric_values(text: str) -> set[str]:
        values = set(re.findall(r"\d+(?:\.\d+)?", text))
        for word, value in NUMBER_WORDS.items():
            if re.search(rf"\b{word}\b", text, flags=re.IGNORECASE):
                values.add(value)
        return values

    unsupported_numbers = sorted(numeric_values(candidate) - numeric_values(source))
    missing_values.extend(f"UNSUPPORTED_NUMBER:{value}" for value in unsupported_numbers)
    missing = tuple(missing_values)
    return ProtectionResult(accepted=not missing, missing_tokens=missing)
