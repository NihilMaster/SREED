from __future__ import annotations

from pathlib import Path
from typing import Protocol

from src.domain.models import DocumentResult


class FileSystemService(Protocol):
    """
    Puerto para operaciones de sistema de archivos usadas por el caso de uso.
    """

    def ensure_layout(self) -> None:
        ...

    def is_supported(self, path: Path) -> bool:
        ...

    def is_ignorable(self, path: Path) -> bool:
        ...

    def wait_until_stable(self, path: Path) -> bool:
        ...

    def move_to_processed(self, path: Path) -> Path:
        ...

    def move_to_failed(self, path: Path) -> Path:
        ...

    def write_result(self, destination_path: Path, result: DocumentResult) -> Path:
        ...