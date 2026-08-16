"""Polling scheduler that materializes triggers and dispatches one root run."""

from __future__ import annotations

import json
import queue
import threading
from typing import Any, Callable

from .models import DurableRunRecord
from .store import AutonomyStore, AutonomyStoreError
from .triggers import probe_public_web_page


class AutonomyScheduler:
    """Owns trigger polling, never Agent/session mutation.

    The scheduler writes durable state and puts ``DURABLE_RUN_DUE`` into the
    root event queue.  The REPL remains the only thread allowed to run Agent.
    """

    def __init__(
        self,
        store: AutonomyStore,
        event_queue: "queue.Queue[tuple[str, object]]",
        *,
        poll_interval: float = 0.5,
        web_probe: Callable[[str], dict[str, Any]] | None = None,
    ) -> None:
        if poll_interval <= 0:
            raise ValueError("poll_interval must be > 0")
        self.store = store
        self.event_queue = event_queue
        self.poll_interval = float(poll_interval)
        self.web_probe = web_probe or probe_public_web_page
        self._wake = threading.Event()
        self._lock = threading.RLock()
        self._inflight: set[str] = set()
        self._thread: threading.Thread | None = None
        self._closed = False

    def start(self, *, active_run_id: str | None = None) -> None:
        with self._lock:
            if self._thread is not None:
                return
            protected: list[str] = []
            if active_run_id:
                try:
                    active = self.store.get_run(active_run_id)
                except AutonomyStoreError:
                    active = None
                if active is not None and active.status == "running":
                    protected.append(active_run_id)
                    self._inflight.add(active_run_id)
            self.store.recover_interrupted(active_run_ids=protected)
            self._thread = threading.Thread(
                target=self._run,
                name="react-autonomy-scheduler",
                daemon=True,
            )
            self._thread.start()

    def close(self, *, timeout: float = 6.0) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            thread = self._thread
        self._wake.set()
        if thread is not None:
            thread.join(timeout=timeout)

    def notify_changed(self) -> None:
        self._wake.set()

    def emit_event(self, name: str, payload: dict[str, Any] | None = None) -> int:
        event_id = self.store.emit_event(name, payload)
        self.notify_changed()
        return event_id

    def finish_run(
        self,
        run_id: str,
        *,
        status: str,
        result: str = "",
        error: str = "",
    ) -> DurableRunRecord:
        record = self.store.finish_run(
            run_id,
            status=status,
            result=result,
            error=error,
        )
        with self._lock:
            self._inflight.discard(run_id)
        self.notify_changed()
        return record

    def runtime_event(self, run_id: str) -> dict[str, Any]:
        run = self.store.get_run(run_id)
        trigger_payload: dict[str, Any] = run.trigger_payload
        encoded_payload = json.dumps(
            trigger_payload, ensure_ascii=False, default=repr
        )
        if len(encoded_payload) > 2_000:
            trigger_payload = {
                "truncated": True,
                "preview": encoded_payload[:2_000],
            }
        return {
            "type": "durable_task_due",
            "task": {
                "id": run.id,
                "kind": "durable",
                "schedule_id": run.automation_id,
                "name": run.automation_name[:200],
                "prompt": run.prompt[:4_000],
                "trigger_type": run.trigger_type,
                "trigger_payload": trigger_payload,
                "attempt": run.attempt,
                "max_retries": run.max_retries,
            },
        }

    def _run(self) -> None:
        while True:
            with self._lock:
                if self._closed:
                    return
                has_inflight = bool(self._inflight)
            try:
                self.store.materialize_due()
                self._poll_web_changes()
                if not has_inflight:
                    run = self.store.claim_next_run()
                    if run is not None:
                        with self._lock:
                            if self._closed:
                                return
                            self._inflight.add(run.id)
                        self.event_queue.put(("DURABLE_RUN_DUE", run.id))
            except Exception as exc:
                # Scheduler failures must be observable but cannot kill the
                # input/Agent loop. The root thread decides how to render them.
                self.event_queue.put((
                    "AUTONOMY_ERROR",
                    f"{type(exc).__name__}: {exc}",
                ))
            self._wake.wait(timeout=self.poll_interval)
            self._wake.clear()

    def _poll_web_changes(self) -> None:
        for automation in self.store.list_due_web_probes():
            try:
                snapshot = self.web_probe(automation.trigger.url)
                self.store.record_web_probe(automation.id, snapshot)
            except Exception as exc:
                self.store.defer_web_probe(
                    automation.id, f"{type(exc).__name__}: {exc}"
                )
                self.event_queue.put((
                    "AUTONOMY_ERROR",
                    f"web trigger {automation.id} probe failed: "
                    f"{type(exc).__name__}: {exc}",
                ))
