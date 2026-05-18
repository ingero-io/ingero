"""Tests for the Ray task-annotation helpers.

`ingero_ray` imports Ray lazily (only inside `_ray_runtime_labels`), so
the decorator and context manager are fully testable without Ray
installed. The fake socket server stands in for a running agent.
"""

from __future__ import annotations

import json
import os
import time

import ingero_ray as ir


def _drain(server, count, timeout=2.0):
    deadline = time.time() + timeout
    while time.time() < deadline and len(server.received()) < count:
        time.sleep(0.01)
    return server.received()


def setup_function(_):
    # Each test starts from a clean writer cache.
    ir.configure_socket_path(None)


def teardown_function(_):
    ir.configure_socket_path(None)


def test_context_manager_emits_task_annotation(fake_server):
    ir.configure_socket_path(fake_server.path)
    with ir.task_annotation(task_name="preprocess"):
        pass

    received = _drain(fake_server, 1)
    assert len(received) == 1
    obj = json.loads(received[0])
    assert obj["pid"] == os.getpid()
    assert obj["labels"]["task_id"] == "preprocess"
    assert obj["labels"]["task_name"] == "preprocess"


def test_context_manager_extra_labels(fake_server):
    ir.configure_socket_path(fake_server.path)
    with ir.task_annotation(task_name="reduce", labels={"shard": "7"}):
        pass

    obj = json.loads(_drain(fake_server, 1)[0])
    assert obj["labels"]["shard"] == "7"
    assert obj["labels"]["task_id"] == "reduce"


def test_decorator_annotates_each_call(fake_server):
    ir.configure_socket_path(fake_server.path)

    @ir.annotate_task(task_name="matmul")
    def work(x):
        return x * 2

    assert work(3) == 6
    assert work(4) == 8

    received = _drain(fake_server, 2)
    assert len(received) == 2
    for line in received:
        obj = json.loads(line)
        assert obj["labels"]["task_id"] == "matmul"
        assert obj["pid"] == os.getpid()


def test_decorator_preserves_function_metadata():
    @ir.annotate_task(task_name="t")
    def documented(x):
        """Original docstring."""
        return x

    assert documented.__name__ == "documented"
    assert documented.__doc__ == "Original docstring."


def test_helper_is_noop_without_agent(tmp_path):
    # Socket absent: the decorator and context manager must still run the
    # wrapped work without raising.
    ir.configure_socket_path(str(tmp_path / "no-agent.sock"))

    @ir.annotate_task(task_name="t")
    def work(x):
        return x + 1

    assert work(10) == 11
    with ir.task_annotation(task_name="t"):
        pass  # no exception


def test_writer_reused_across_tasks(fake_server):
    ir.configure_socket_path(fake_server.path)

    @ir.annotate_task(task_name="loop")
    def work():
        return None

    for _ in range(15):
        work()

    received = _drain(fake_server, 15)
    assert len(received) == 15
    # All 15 annotations carry the same task id - one reused writer.
    assert all(json.loads(r)["labels"]["task_id"] == "loop" for r in received)


def test_emit_skips_when_no_labels(fake_server):
    # No task name and no Ray context: nothing to scope by, so no
    # annotation is sent rather than an empty one (which the agent
    # rejects anyway).
    ir.configure_socket_path(fake_server.path)
    with ir.task_annotation():
        pass
    time.sleep(0.2)
    assert fake_server.received() == []
