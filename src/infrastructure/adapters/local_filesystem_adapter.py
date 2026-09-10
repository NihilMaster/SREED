from __future__ import annotations

import json
import logging
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from src.domain.models import DocumentResult


class LocalFileSystemAdapter:
    """
    Adaptador de sistema de archivos local.

    Responsabilidades:
    - Crear carpetas de trabajo.
    - Detectar archivos ignorables.
    - Validar extensiones soportadas.
    - Esperar estabilidad de archivos recien copiados.
    - Mover archivos a procesados o fallidos.
    - Escribir JSON lateral de resultado.
    """

    SUPPORTED_SUFFIXES = {
        ".pdf",
        ".png",
        ".jpg",
        ".jpeg",
        ".tif",
        ".tiff",
    }

    IGNORABLE_PREFIXES = (
        ".",
        "~$",
        "tmp_",
        "temp_",
    )

    IGNORABLE_SUFFIXES = {
        ".tmp",
        ".temp",
        ".part",
        ".partial",
        ".crdownload",
        ".download",
        ".lock",
    }

    def __init__(self, settings) -> None:
        self._settings = settings
        self._logger = logging.getLogger(__name__)
        self.ensure_layout()

    def ensure_layout(self) -> None:
        directories = [
            self._settings.root,
            self._settings.input_dir,
            self._settings.processed_dir,
            self._settings.failed_dir,
            self._settings.log_dir,
            self._settings.chroma_dir,
        ]

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

    def is_supported(self, path: Path) -> bool:
        if path.is_dir():
            return False
        return path.suffix.lower() in self.SUPPORTED_SUFFIXES

    def is_ignorable(self, path: Path) -> bool:
        if path.is_dir():
            return True

        name = path.name.lower()

        if name.startswith(self.IGNORABLE_PREFIXES):
            return True

        if path.suffix.lower() in self.IGNORABLE_SUFFIXES:
            return True

        if name in {"desktop.ini", "thumbs.db"}:
            return True

        return False

    def wait_until_stable(
        self,
        path: Path,
        checks: int = 3,
        interval: float = 0.5,
    ) -> bool:
        """
        Espera a que el archivo tenga tamano estable.

        Esto evita procesar archivos que todavia estan siendo copiados
        o descargados por otro proceso.
        """

        if not path.exists():
            return False

        max_attempts = max(checks * 20, 120)
        last_size = -1
        stable_count = 0

        for _ in range(max_attempts):
            try:
                size = path.stat().st_size
            except FileNotFoundError:
                return False
            except OSError as exc:
                self._logger.debug("Error consultando %s: %s", path, exc)
                return False

            if size == last_size and size > 0:
                stable_count += 1
            else:
                stable_count = 0

            if stable_count >= checks:
                return True

            last_size = size
            time.sleep(interval)

        return False

    def move_to_processed(self, path: Path) -> Path:
        return self._move_to(self._settings.processed_dir, path)

    def move_to_failed(self, path: Path) -> Path:
        return self._move_to(self._settings.failed_dir, path)

    def write_result(self, destination_path: Path, result: DocumentResult) -> Path:
        sidecar_path = destination_path.with_suffix(".json")

        payload = result.model_dump(mode="json")

        sidecar_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )

        self._logger.info("Resultado escrito en: %s", sidecar_path)
        return sidecar_path

    def list_files(self, directory: Path, limit: int = 100) -> List[Path]:
        if not directory.exists():
            return []

        entries = []

        try:
            for path in directory.iterdir():
                if not path.is_file():
                    continue

                if path.suffix.lower() == ".json":
                    continue

                try:
                    modified_at = path.stat().st_mtime
                except OSError:
                    continue

                entries.append((modified_at, path))
        except OSError:
            return []

        entries.sort(key=lambda item: item[0], reverse=True)
        return [path for _, path in entries[:limit]]

    def read_sidecar(self, document_path: Path) -> Optional[Dict]:
        sidecar_path = document_path.with_suffix(".json")

        if not sidecar_path.exists():
            return None

        try:
            return json.loads(sidecar_path.read_text(encoding="utf-8"))
        except Exception:
            self._logger.exception("No se pudo leer el JSON lateral: %s", sidecar_path)
            return None

    def _move_to(self, target_directory: Path, path: Path) -> Path:
        if not path.exists():
            raise FileNotFoundError(f"El archivo no existe: {path}")

        target_directory.mkdir(parents=True, exist_ok=True)
        destination = self._build_destination_path(target_directory, path)

        try:
            shutil.move(str(path), str(destination))
        except PermissionError as exc:
            self._logger.error("Permiso denegado moviendo %s hacia %s", path, destination)
            raise
        except OSError as exc:
            self._logger.error("Error de sistema moviendo %s hacia %s", path, destination)
            raise

        return destination

    def _build_destination_path(self, directory: Path, path: Path) -> Path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = f"{timestamp}_{path.name}"
        destination = directory / base_name

        counter = 1

        while destination.exists():
            destination = directory / f"{timestamp}_{path.stem}_{counter}{path.suffix}"
            counter += 1

        return destination