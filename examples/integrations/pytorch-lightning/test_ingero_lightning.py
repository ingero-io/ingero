"""Tests for the IngeroCallback Lightning adapter.

IngeroCallback falls back to subclassing `object` when pytorch_lightning
is not installed, so the callback's hook methods are driven directly with
a fake trainer here - no PyTorch / Lightning import needed. The hook
signatures match Lightning's Callback API.
"""

from __future__ import annotations

import json
import time

from ingero_lightning import IngeroCallback


class FakeTrainer:
    """Stands in for pl.Trainer: only the fields the callback reads."""

    def __init__(self, global_step=0, current_epoch=0):
        self.global_step = global_step
        self.current_epoch = current_epoch


def _drain(server, count, timeout=2.0):
    deadline = time.time() + timeout
    while time.time() < deadline and len(server.received()) < count:
        time.sleep(0.01)
    return server.received()


def test_callback_emits_step_and_epoch(fake_server):
    cb = IngeroCallback(socket_path=fake_server.path, pid=777)
    cb.setup(trainer=None, pl_module=None)

    trainer = FakeTrainer(global_step=12, current_epoch=2)
    cb.on_train_batch_start(trainer, None, batch=None, batch_idx=0)
    cb.teardown(trainer=None, pl_module=None)

    received = _drain(fake_server, 1)
    assert len(received) == 1
    obj = json.loads(received[0])
    assert obj["pid"] == 777
    assert obj["labels"]["step"] == "12"
    assert obj["labels"]["epoch"] == "2"


def test_callback_emits_epoch_boundaries(fake_server):
    cb = IngeroCallback(socket_path=fake_server.path, pid=5, run_id="r1")
    cb.setup(trainer=None, pl_module=None)

    trainer = FakeTrainer(global_step=0, current_epoch=3)
    cb.on_train_epoch_start(trainer, None)
    cb.on_train_epoch_end(trainer, None)
    cb.teardown(trainer=None, pl_module=None)

    received = _drain(fake_server, 2)
    assert len(received) == 2
    start, end = (json.loads(r) for r in received)
    assert start["labels"] == {"run_id": "r1", "epoch": "3"}
    assert end["labels"] == {"run_id": "r1", "epoch": "3", "phase": "epoch_end"}


def test_callback_run_id_on_every_annotation(fake_server):
    cb = IngeroCallback(socket_path=fake_server.path, pid=1, run_id="job-9")
    cb.setup(trainer=None, pl_module=None)
    cb.on_train_batch_start(FakeTrainer(4, 0), None, None, 0)
    cb.teardown(trainer=None, pl_module=None)

    received = _drain(fake_server, 1)
    assert json.loads(received[0])["labels"]["run_id"] == "job-9"


def test_callback_is_noop_without_agent(tmp_path):
    # Socket absent: setup, hooks, teardown must all run without raising.
    missing = str(tmp_path / "no-agent.sock")
    cb = IngeroCallback(socket_path=missing, pid=1)
    cb.setup(trainer=None, pl_module=None)
    trainer = FakeTrainer(1, 0)
    cb.on_train_epoch_start(trainer, None)
    cb.on_train_batch_start(trainer, None, None, 0)
    cb.on_train_epoch_end(trainer, None)
    cb.teardown(trainer=None, pl_module=None)  # no exception


def test_callback_reuses_single_connection(fake_server):
    cb = IngeroCallback(socket_path=fake_server.path, pid=2)
    cb.setup(trainer=None, pl_module=None)
    for step in range(20):
        cb.on_train_batch_start(FakeTrainer(step, 0), None, None, step)
    cb.teardown(trainer=None, pl_module=None)

    received = _drain(fake_server, 20)
    assert len(received) == 20
    steps = [json.loads(r)["labels"]["step"] for r in received]
    assert steps == [str(i) for i in range(20)]
