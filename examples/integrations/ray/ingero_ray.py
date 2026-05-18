"""Ray helpers that annotate an Ingero eBPF trace per task.

A Ray task runs in a worker process. `annotate_task` (a decorator) and
`task_annotation` (a context manager) write one annotation, scoped to the
worker PID, when the task starts: a `task_id` label plus any Ray-provided
identifiers (job id, the current task's id). The agent (run as `ingero
trace --record --annotate`) joins the label to the eBPF event stream by
process incarnation and time window, so a recorded trace can be sliced
per task with `ingero query` and `ingero explain`.

The protocol and socket work live in `ingero_annotate.py`, which has no
Ray dependency. This file is the thin Ray adapter.

Design:
  - One annotation per task start - the natural per-task boundary. The
    annotation is scoped to os.getpid(), the worker process.
  - Each Ray worker process gets one cached AnnotationWriter, reused
    across every task that runs in that worker. No `ingero annotate`
    subprocess is spawned per task.
  - If the agent is not running with `--annotate`, the writer is inert;
    the helper becomes a no-op and never crashes or slows a task.

Usage - decorator (the primary, recommended path):

    import ray
    from ingero_ray import annotate_task

    @ray.remote
    @annotate_task(task_name="preprocess")
    def preprocess(shard):
        ...

`annotate_task` always runs `_emit` inside the worker that runs the task,
so the annotation is correctly scoped to the worker PID. Prefer it.

Usage - context manager (only inside a running task body):

    @ray.remote
    def train(shard):
        with task_annotation(task_name="train"):
            ...

`task_annotation` MUST be opened inside a running Ray task, never on the
driver. The annotation is scoped to os.getpid(); opening the context on
the driver would mis-scope it to the driver PID and the recorded trace
would not slice correctly. When in doubt, use the `annotate_task`
decorator, which cannot be misused this way.
"""

from __future__ import annotations

import functools
import logging
import os
import threading

from ingero_annotate import (
    KEY_RUN_ID,
    KEY_TASK_ID,
    AnnotationWriter,
)

logger = logging.getLogger("ingero.ray")

# One writer per worker process. Ray runs each task inside a worker; a
# worker serves many tasks over its life, so the socket is opened once
# per worker and reused. The cache is keyed by PID so a forked child
# never inherits a parent's socket fd.
_writer_lock = threading.Lock()
_writer_pid: int | None = None
_writer: AnnotationWriter | None = None
_socket_override: str | None = None


def configure_socket_path(path: str | None) -> None:
    """Override the annotation socket path for the current process.

    Mainly for tests; production callers let the writer resolve the
    agent's canonical socket path. Resets any cached writer.
    """
    global _socket_override, _writer, _writer_pid
    with _writer_lock:
        _socket_override = path
        if _writer is not None:
            _writer.close()
        _writer = None
        _writer_pid = None


def _get_writer() -> AnnotationWriter:
    """Return this worker process's cached AnnotationWriter.

    Re-creates the writer if the PID changed (a fork), so a child never
    shares a socket fd with its parent.
    """
    global _writer, _writer_pid
    with _writer_lock:
        pid = os.getpid()
        if _writer is None or _writer_pid != pid:
            if _writer is not None:
                _writer.close()
            _writer = AnnotationWriter(socket_path=_socket_override)
            _writer_pid = pid
            if _writer.active:
                logger.info(
                    "ingero annotations enabled (socket=%s, pid=%d)",
                    _writer.socket_path, pid,
                )
        return _writer


def _ray_runtime_labels() -> dict:
    """Collect Ray-provided identifiers for the currently running task.

    Returns whatever Ray exposes; an empty dict when Ray is not present
    or not inside a task. Never raises - Ray's runtime-context API is
    optional here.
    """
    labels: dict = {}
    try:  # pragma: no cover - exercised only with Ray installed
        import ray

        ctx = ray.get_runtime_context()
        task_id = ctx.get_task_id()
        if task_id:
            labels[KEY_TASK_ID] = str(task_id)
        job_id = ctx.get_job_id()
        if job_id:
            labels[KEY_RUN_ID] = str(job_id)
    except Exception:
        pass
    return labels


def _emit(task_name: str | None, extra: dict | None) -> None:
    """Write one task-start annotation scoped to this worker PID."""
    labels = _ray_runtime_labels()
    if task_name:
        # task_id is the stable slicing key; prefer the Ray task id when
        # present, otherwise fall back to the caller-supplied name.
        labels.setdefault(KEY_TASK_ID, str(task_name))
        labels["task_name"] = str(task_name)
    if extra:
        labels.update({str(k): str(v) for k, v in extra.items()})
    if not labels:
        # Nothing to scope by - skip rather than send an empty annotation.
        return
    _get_writer().write(labels, pid=os.getpid())


class task_annotation:  # noqa: N801 - context-manager naming, lowercase by convention
    """Context manager that annotates a Ray task on entry.

    MUST be used inside a running Ray task, not on the driver. The
    annotation is scoped to os.getpid(); opening this context on the
    driver mis-scopes it to the driver PID and the trace will not slice
    per task. Prefer the `annotate_task` decorator, which always runs in
    the worker and cannot be misused this way.

    Args:
        task_name: a stable name for the task, used as the `task_id`
            label when Ray does not supply a task id.
        labels: optional extra key/value labels to attach.
    """

    def __init__(self, task_name: str | None = None,
                 labels: dict | None = None):
        self._task_name = task_name
        self._labels = labels

    def __enter__(self) -> "task_annotation":
        _emit(self._task_name, self._labels)
        return self

    def __exit__(self, *exc) -> None:
        return None


def annotate_task(task_name: str | None = None, labels: dict | None = None):
    """Decorator: annotate a function each time it runs as a Ray task.

    Apply it under `@ray.remote` so the annotation is written inside the
    worker process:

        @ray.remote
        @annotate_task(task_name="preprocess")
        def preprocess(shard):
            ...
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            _emit(task_name, labels)
            return func(*args, **kwargs)
        return wrapper
    return decorator
