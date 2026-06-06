from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from queue import Queue
from threading import Thread
from typing import Any, Optional


@dataclass
class CheckpointJob:
    snapshot: dict[str, Any]
    path: Path


class CheckpointWriter(ABC):

    def __init__(self) -> None:
        self._queue: Queue[Optional[CheckpointJob]] = Queue()
        self._thread = Thread(target=self._run, daemon=True)
        self._thread.start()

    def enqueue(self, snapshot: dict[str, Any], path: Path) -> None:
        self._queue.put(CheckpointJob(snapshot=snapshot, path=path))

    def wait_for_idle(self) -> None:
        self._queue.join()

    def stop(self) -> None:
        self._queue.put(None)
        self._thread.join()

    @abstractmethod
    def write(self, snapshot: dict[str, Any], path: Path) -> None:
        ...

    def _run(self) -> None:
        while True:
            job = self._queue.get()
            try:
                if job is None:
                    return
                self.write(job.snapshot, job.path)
            finally:
                self._queue.task_done()
