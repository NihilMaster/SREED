from __future__ import annotations

import logging
import queue
import threading
from pathlib import Path
from typing import Dict

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer


class _FolderEventHandler(FileSystemEventHandler):
    def __init__(self, monitor: "WatchdogFolderMonitor") -> None:
        super().__init__()
        self._monitor = monitor

    def on_created(self, event) -> None:
        if event.is_directory:
            return
        self._monitor.schedule(Path(event.src_path))

    def on_modified(self, event) -> None:
        if event.is_directory:
            return
        self._monitor.schedule(Path(event.src_path))

    def on_moved(self, event) -> None:
        if event.is_directory:
            return
        self._monitor.schedule(Path(event.dest_path))


class WatchdogFolderMonitor:
    """
    Monitor de carpeta basado en watchdog.

    Usa:
    - Debounding con timers para evitar multiples eventos seguidos.
    - Cola interna para procesar archivos en un worker thread.
    - Caso de uso de aplicacion para no acoplar watchdog al dominio.
    """

    def __init__(
        self,
        settings,
        use_case,
        file_system,
    ) -> None:
        self._settings = settings
        self._use_case = use_case
        self._file_system = file_system
        self._logger = logging.getLogger(__name__)

        self._queue: queue.Queue = queue.Queue()
        self._timers: Dict[str, threading.Timer] = {}
        self._lock = threading.Lock()
        self._observer = Observer()
        self._worker: threading.Thread | None = None
        self._stopped = threading.Event()
        self._debounce_seconds = 1.0

    def start(self) -> None:
        self._file_system.ensure_layout()

        handler = _FolderEventHandler(self)

        self._observer.schedule(
            handler,
            str(self._settings.input_dir),
            recursive=False,
        )

        self._worker = threading.Thread(
            target=self._worker_loop,
            name="sreed-monitor-worker",
            daemon=True,
        )
        self._worker.start()

        self._observer.start()
        self._scan_existing_files()

        self._logger.info(
            "Monitor watchdog activo en: %s",
            self._settings.input_dir,
        )

    def stop(self) -> None:
        self._stopped.set()

        with self._lock:
            for timer in self._timers.values():
                timer.cancel()
            self._timers.clear()

        self._queue.put(None)

        try:
            self._observer.stop()
        except Exception:
            self._logger.exception("Error deteniendo observer.")

        if self._worker is not None:
            self._worker.join(timeout=5)

        self._logger.info("Monitor watchdog detenido.")

    def schedule(self, path: Path) -> None:
        if self._stopped.is_set():
            return

        if not path.exists():
            return

        if self._file_system.is_ignorable(path):
            return

        key = str(path).lower()

        with self._lock:
            existing = self._timers.get(key)
            if existing is not None:
                existing.cancel()

            timer = threading.Timer(
                self._debounce_seconds,
                self._enqueue,
                args=(path,),
            )
            timer.daemon = True
            self._timers[key] = timer
            timer.start()

    def _enqueue(self, path: Path) -> None:
        key = str(path).lower()

        with self._lock:
            self._timers.pop(key, None)

        if self._stopped.is_set():
            return

        self._logger.debug("Encolando archivo: %s", path)
        self._queue.put(path)

    def _scan_existing_files(self) -> None:
        if not self._settings.input_dir.exists():
            return

        for path in self._settings.input_dir.iterdir():
            if path.is_file():
                self.schedule(path)

    def _worker_loop(self) -> None:
        while not self._stopped.is_set():
            item = self._queue.get()

            if item is None:
                break

            try:
                if item.exists():
                    self._use_case.execute(item)
                else:
                    self._logger.debug("El archivo ya no existe: %s", item)
            except Exception:
                self._logger.exception("Error no controlado en worker de procesamiento.")
            finally:
                self._queue.task_done()