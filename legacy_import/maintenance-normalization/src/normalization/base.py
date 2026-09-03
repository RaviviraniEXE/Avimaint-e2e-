"""Common interface for all normalization systems (A/B/C/D)."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class NormalizationResult:
    raw: str
    normalized: str
    alignment: List[Tuple[str, Tuple[int, int], List[str], Optional[str]]] = field(default_factory=list)
    stats: Dict[str, int] = field(default_factory=dict)


class Normalizer(ABC):
    name: str = "base"

    @abstractmethod
    def normalize(self, text: str) -> NormalizationResult:
        ...

    def normalize_many(self, texts):
        return [self.normalize(t) for t in texts]

