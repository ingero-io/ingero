"""PyTorch Lightning callback that annotates an Ingero eBPF trace.

`IngeroCallback` injects `step` and `epoch` labels into a live trace as a
training run progresses. The agent (run as `ingero trace --record
--annotate`) joins those labels to the eBPF event stream by process
incarnation and time window, so a recorded trace can later be sliced per
step / per epoch with `ingero query` and `ingero explain`.

The protocol and socket work live in `ingero_annotate.py`, which has no
framework dependency. This file is the thin Lightning adapter: it owns a
single `AnnotationWriter` for the run and emits one annotation per step
and per epoch boundary.

Design:
  - The socket connection is opened once, in `setup`, and reused. No
    `ingero annotate` subprocess is spawned per step.
  - Annotations are scoped to the training process PID (os.getpid()), so
    the agent attributes the eBPF events of this process to the step.
  - If the agent is not running with `--annotate`, the writer is inert;
    the callback becomes a no-op and never slows or crashes the run.

Usage:
    from ingero_lightning import IngeroCallback
    trainer = pl.Trainer(callbacks=[IngeroCallback(run_id="my-run")])
"""

from __future__ import annotations

import logging
import os

from ingero_annotate import (
    KEY_EPOCH,
    KEY_RUN_ID,
    KEY_STEP,
    AnnotationWriter,
)

logger = logging.getLogger("ingero.lightning")

try:  # pragma: no cover - import guard, exercised only with Lightning present
    from pytorch_lightning import Callback as _LightningCallback
except Exception:  # pragma: no cover
    # Lightning is an optional dependency of this example. Falling back to
    # `object` keeps `ingero_annotate` and the callback logic importable
    # (and unit-testable) on a machine without Lightning installed.
    _LightningCallback = object


class IngeroCallback(_LightningCallback):
    """Lightning callback that writes step/epoch annotations to the agent.

    Args:
        run_id: optional identifier emitted on every annotation as the
            `run_id` label, so multiple runs in one trace stay separable.
        socket_path: override the annotation socket path; defaults to the
            agent's canonical location.
        pid: process to scope annotations to; defaults to this process.
    """

    def __init__(self, run_id: str | None = None,
                 socket_path: str | None = None,
                 pid: int | None = None):
        super().__init__()
        self._run_id = run_id
        self._socket_path = socket_path
        self._pid = pid if pid is not None else os.getpid()
        self._writer: AnnotationWriter | None = None

    # -- connection lifecycle ------------------------------------------------

    def setup(self, trainer, pl_module, stage=None):  # noqa: D102
        if self._writer is None:
            self._writer = AnnotationWriter(socket_path=self._socket_path)
            if self._writer.active:
                logger.info(
                    "ingero annotations enabled (socket=%s, pid=%d)",
                    self._writer.socket_path, self._pid,
                )

    def teardown(self, trainer, pl_module, stage=None):  # noqa: D102
        if self._writer is not None:
            self._writer.close()
            self._writer = None

    # -- annotation emission -------------------------------------------------

    def _base_labels(self) -> dict:
        labels: dict = {}
        if self._run_id:
            labels[KEY_RUN_ID] = str(self._run_id)
        return labels

    def _emit(self, labels: dict) -> None:
        if self._writer is None:
            return
        self._writer.write(labels, pid=self._pid)

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):  # noqa: D102
        labels = self._base_labels()
        labels[KEY_STEP] = str(trainer.global_step)
        labels[KEY_EPOCH] = str(trainer.current_epoch)
        self._emit(labels)

    def on_train_epoch_start(self, trainer, pl_module):  # noqa: D102
        labels = self._base_labels()
        labels[KEY_EPOCH] = str(trainer.current_epoch)
        self._emit(labels)

    def on_train_epoch_end(self, trainer, pl_module):  # noqa: D102
        labels = self._base_labels()
        labels[KEY_EPOCH] = str(trainer.current_epoch)
        labels["phase"] = "epoch_end"
        self._emit(labels)
