"""Tests for the framework-agnostic annotation writer.

These tests import only ingero_annotate (no PyTorch / Lightning), so the
protocol and socket logic is verified independent of the heavy deps.
"""

from __future__ import annotations

import json
import time

import pytest

import ingero_annotate as ia


# -- contract: label-key charset and limits ---------------------------------

def test_valid_label_keys():
    for key in ("step", "epoch", "task_id", "run.id", "a-b", "Z9_"):
        assert ia.is_valid_label_key(key)


def test_invalid_label_keys():
    for key in ("", "has space", "tab\t", "slash/", "x" * 65, "emoji❤"):
        assert not ia.is_valid_label_key(key)


def test_validate_rejects_empty_labels():
    with pytest.raises(ia.AnnotationError):
        ia.validate_labels({})


def test_validate_rejects_too_many_labels():
    labels = {f"k{i}": "v" for i in range(ia.MAX_LABELS_PER_ANNOTATION + 1)}
    with pytest.raises(ia.AnnotationError):
        ia.validate_labels(labels)


def test_validate_accepts_max_labels():
    labels = {f"k{i}": "v" for i in range(ia.MAX_LABELS_PER_ANNOTATION)}
    ia.validate_labels(labels)  # no raise


def test_validate_rejects_overlong_value():
    with pytest.raises(ia.AnnotationError):
        ia.validate_labels({"step": "x" * (ia.MAX_LABEL_VALUE_LEN + 1)})


def test_validate_accepts_max_value():
    ia.validate_labels({"step": "x" * ia.MAX_LABEL_VALUE_LEN})


def test_validate_rejects_control_character_in_value():
    for bad in ("\x00", "\x1b[31m", "line\nbreak", "\x7f"):
        with pytest.raises(ia.AnnotationError):
            ia.validate_labels({"step": bad})


def test_validate_rejects_non_string_key_or_value():
    with pytest.raises(ia.AnnotationError):
        ia.validate_labels({"step": 42})
    with pytest.raises(ia.AnnotationError):
        ia.validate_labels({1: "v"})


# -- contract: NDJSON line encoding -----------------------------------------

def test_encode_annotation_is_one_ndjson_line():
    line = ia.encode_annotation({"step": "7"}, pid=1234, ts_ns=99)
    assert line.endswith(b"\n")
    assert line.count(b"\n") == 1
    obj = json.loads(line)
    assert obj == {"labels": {"step": "7"}, "pid": 1234, "ts": 99}


def test_encode_annotation_omits_optional_fields():
    obj = json.loads(ia.encode_annotation({"step": "1"}))
    assert obj == {"labels": {"step": "1"}}
    assert "pid" not in obj and "ts" not in obj


def test_encode_annotation_rejects_oversized_line(monkeypatch):
    # The per-label caps (32 labels x 256-byte values) keep a valid
    # annotation well under the 16 KiB line cap, so the line-cap branch
    # cannot be reached with otherwise-valid labels. Lower the cap to
    # prove the framing check fires independently of the label checks.
    monkeypatch.setattr(ia, "MAX_LINE_BYTES", 32)
    with pytest.raises(ia.AnnotationError):
        ia.encode_annotation({"step": "x" * 100})


def test_encode_annotation_within_line_cap_at_max_labels():
    # A worst-case valid annotation - max labels, max-length values -
    # still fits inside MAX_LINE_BYTES.
    labels = {f"k{i}": "x" * ia.MAX_LABEL_VALUE_LEN
              for i in range(ia.MAX_LABELS_PER_ANNOTATION)}
    line = ia.encode_annotation(labels)
    assert len(line) <= ia.MAX_LINE_BYTES


def test_protocol_constants_match_contract():
    # Pinned to pkg/contract/annotate.go. A drift here means the agent
    # contract changed and this example must be updated with it.
    assert ia.MAX_LABEL_KEY_LEN == 64
    assert ia.MAX_LABEL_VALUE_LEN == 256
    assert ia.MAX_LABELS_PER_ANNOTATION == 32
    assert ia.MAX_LINE_BYTES == 16 * 1024
    assert ia.ANNOTATION_SOCKET_NAME == "annotate.sock"
    assert ia.ANNOTATION_SOCKET_DIR == "/run/ingero"


# -- writer: connected path over a fake socket ------------------------------

def test_writer_sends_correct_ndjson(fake_server):
    writer = ia.AnnotationWriter(socket_path=fake_server.path)
    assert writer.active
    assert writer.write({"step": "10", "epoch": "0"}, pid=4321)
    writer.close()

    # Give the server thread a moment to drain.
    deadline = time.time() + 2.0
    while time.time() < deadline and not fake_server.received():
        time.sleep(0.01)

    received = fake_server.received()
    assert len(received) == 1
    obj = json.loads(received[0])
    assert obj["labels"] == {"step": "10", "epoch": "0"}
    assert obj["pid"] == 4321


def test_writer_reuses_one_connection(fake_server):
    writer = ia.AnnotationWriter(socket_path=fake_server.path)
    for step in range(5):
        assert writer.write({"step": str(step)}, pid=1)
    writer.close()

    deadline = time.time() + 2.0
    while time.time() < deadline and len(fake_server.received()) < 5:
        time.sleep(0.01)

    received = fake_server.received()
    assert len(received) == 5
    steps = [json.loads(line)["labels"]["step"] for line in received]
    assert steps == ["0", "1", "2", "3", "4"]


def test_writer_raises_on_contract_violation_when_connected(fake_server):
    writer = ia.AnnotationWriter(socket_path=fake_server.path)
    with pytest.raises(ia.AnnotationError):
        writer.write({}, pid=1)
    with pytest.raises(ia.AnnotationError):
        writer.write({"bad key": "v"}, pid=1)
    writer.close()


# -- writer: no-op path when the socket is absent ---------------------------

def test_writer_is_noop_when_socket_missing(tmp_path):
    missing = str(tmp_path / "does-not-exist.sock")
    writer = ia.AnnotationWriter(socket_path=missing)
    assert not writer.active
    # write returns False, does not raise, does not block.
    assert writer.write({"step": "1"}, pid=1) is False
    writer.close()


def test_writer_noop_still_validates_contract(tmp_path):
    # Even on the no-op path a contract violation is a caller bug and is
    # surfaced; only transport failure is swallowed.
    missing = str(tmp_path / "does-not-exist.sock")
    writer = ia.AnnotationWriter(socket_path=missing)
    with pytest.raises(ia.AnnotationError):
        writer.write({}, pid=1)
    writer.close()


def test_writer_degrades_when_server_disappears(fake_server):
    writer = ia.AnnotationWriter(socket_path=fake_server.path)
    assert writer.active
    assert writer.write({"step": "0"}, pid=1)
    fake_server.stop()
    time.sleep(0.1)
    # The next writes may fail on the dead socket; the writer must flip to
    # a no-op and never raise a transport error.
    for _ in range(10):
        result = writer.write({"step": "x"}, pid=1)
        assert result in (True, False)
    writer.close()


def test_writer_context_manager(fake_server):
    with ia.AnnotationWriter(socket_path=fake_server.path) as writer:
        assert writer.write({"step": "1"}, pid=1)
    # Exiting the context closes the socket.
    assert not writer.active


def test_now_ns_is_monotonic_enough():
    a = ia.now_ns()
    b = ia.now_ns()
    assert isinstance(a, int) and b >= a
