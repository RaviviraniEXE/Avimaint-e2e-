"""Train-only most-frequent-replacement normalization baseline."""

from __future__ import annotations

from collections import Counter, defaultdict
from difflib import SequenceMatcher

import pandas as pd


def learn_replacements(train: pd.DataFrame) -> dict[str, str]:
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for source, target in zip(train["raw_text"].astype(str), train["target_text"].astype(str), strict=False):
        source_tokens = source.split()
        target_tokens = target.split()
        matcher = SequenceMatcher(None, [value.lower() for value in source_tokens], [value.lower() for value in target_tokens])
        for tag, left_start, left_end, right_start, right_end in matcher.get_opcodes():
            if tag == "replace" and left_end - left_start == right_end - right_start:
                for raw_token, target_token in zip(
                    source_tokens[left_start:left_end], target_tokens[right_start:right_end], strict=False
                ):
                    counts[raw_token.lower()][target_token] += 1
    return {source: alternatives.most_common(1)[0][0] for source, alternatives in counts.items()}


def apply_replacements(text: str, replacements: dict[str, str]) -> str:
    return " ".join(replacements.get(token.lower(), token) for token in str(text).split())
