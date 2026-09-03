"""Dataset-specific adapters into the shared MaintenanceRecord contract."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterator

from avimaint.contracts import MaintenanceRecord


class DatasetAdapter(ABC):
    @abstractmethod
    def iter_records(self, source: Path) -> Iterator[MaintenanceRecord]:
        """Yield validated records without changing source content."""


class AviationAdapter(DatasetAdapter):
    def iter_records(self, source: Path) -> Iterator[MaintenanceRecord]:
        raise NotImplementedError("Port the validated aviation loader here.")


class MaintIEAdapter(DatasetAdapter):
    def iter_records(self, source: Path) -> Iterator[MaintenanceRecord]:
        raise NotImplementedError("Load the official MaintIE release without inventing missing fields.")

