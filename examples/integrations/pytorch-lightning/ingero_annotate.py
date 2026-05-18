"""Framework-agnostic writer for the Ingero agent annotation socket.

This module carries no PyTorch / Lightning dependency. It speaks the
agent's NDJSON annotation protocol (agent v0.17.0) and nothing else, so
the protocol and socket behaviour can be unit-tested without importing a
training framework.

Wire protocol (see pkg/contract/annotate.go in the agent repo):

  - The agent, when run as `ingero trace --record --annotate`, binds a
    Unix-domain socket at /run/ingero/annotate.sock (or, when /run is not
    writable, ~/.ingero/annotate/annotate.sock).
  - A writer connects and sends newline-delimited JSON. Each line is one
    annotation object:

        {"labels": {"step": "42"}, "pid": 1234, "ts": 1700000000000000000}

  - `labels` is required and non-empty. `pid` scopes the annotation to a
    process incarnation. `ts` is optional unix nanoseconds; the agent
    stamps receive time when it is absent.

Validation limits below mirror the agent's contract exactly. A line that
violates them is rejected by the agent without dropping the listener;
this writer rejects it locally first so a caller sees a clear error.
"""

from __future__ import annotations

import json
import logging
import os
import socket
import threading
import time

logger = logging.getLogger("ingero.annotate")

# --- Protocol constants, pinned to pkg/contract/annotate.go ----------------

ANNOTATION_PROTOCOL_VERSION = 1
ANNOTATION_SOCKET_NAME = "annotate.sock"
ANNOTATION_SOCKET_DIR = "/run/ingero"

MAX_LABEL_KEY_LEN = 64
MAX_LABEL_VALUE_LEN = 256
MAX_LABELS_PER_ANNOTATION = 32
MAX_LINE_BYTES = 16 * 1024

# Well-known label keys the distribution integrations standardize on.
KEY_STEP = "step"
KEY_EPOCH = "epoch"
KEY_TASK_ID = "task_id"
KEY_PHASE = "phase"
KEY_RUN_ID = "run_id"

# Allowed label-key bytes: ASCII letters, digits, underscore, dot, hyphen.
_KEY_CHARS = frozenset(
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_.-"
)


class AnnotationError(ValueError):
    """Raised when an annotation violates the agent's contract limits."""


def is_valid_label_key(key: str) -> bool:
    """Report whether key satisfies the agent's label-key contract.

    Non-empty, at most MAX_LABEL_KEY_LEN bytes, every byte in the
    [A-Za-z0-9_.-] charset. Mirrors contract.IsValidAnnotationLabelKey.
    """
    if not key or len(key.encode("utf-8")) > MAX_LABEL_KEY_LEN:
        return False
    return all(c in _KEY_CHARS for c in key)


def validate_labels(labels: dict) -> None:
    """Validate a label map against the agent contract.

    Raises AnnotationError on the first violation. Mirrors the checks in
    pkg/annotate/annotation.go Validate().
    """
    if not labels:
        raise AnnotationError("annotation has no labels")
    if len(labels) > MAX_LABELS_PER_ANNOTATION:
        raise AnnotationError(
            f"annotation has {len(labels)} labels, max {MAX_LABELS_PER_ANNOTATION}"
        )
    for key, value in labels.items():
        if not isinstance(key, str) or not isinstance(value, str):
            raise AnnotationError("label keys and values must be strings")
        if not is_valid_label_key(key):
            raise AnnotationError(
                f"invalid label key {key!r} (charset A-Za-z0-9_.-, "
                f"max {MAX_LABEL_KEY_LEN} bytes)"
            )
        vbytes = value.encode("utf-8")
        if len(vbytes) > MAX_LABEL_VALUE_LEN:
            raise AnnotationError(
                f"label {key!r} value is {len(vbytes)} bytes, "
                f"max {MAX_LABEL_VALUE_LEN}"
            )
        for ch in value:
            o = ord(ch)
            if o < 0x20 or o == 0x7F:
                raise AnnotationError(
                    f"label {key!r} value contains a control character "
                    f"(0x{o:02x})"
                )


def encode_annotation(labels: dict, pid: int | None = None,
                       ts_ns: int | None = None) -> bytes:
    """Encode one annotation as a single NDJSON line (terminated by \\n).

    Validates against the contract first, then checks the encoded line
    against MAX_LINE_BYTES. Raises AnnotationError on any violation.
    """
    validate_labels(labels)
    obj: dict = {"labels": labels}
    if pid is not None:
        obj["pid"] = int(pid)
    if ts_ns is not None:
        obj["ts"] = int(ts_ns)
    line = json.dumps(obj, separators=(",", ":"), sort_keys=True).encode("utf-8")
    if len(line) + 1 > MAX_LINE_BYTES:
        raise AnnotationError(
            f"encoded annotation is {len(line) + 1} bytes, max {MAX_LINE_BYTES}"
        )
    return line + b"\n"


def default_socket_path() -> str:
    """Return the annotation socket path the agent binds.

    Prefers the canonical /run/ingero/annotate.sock when a socket is
    actually bound there (the privileged-trace case). Otherwise falls
    back to ~/.ingero/annotate/annotate.sock, mirroring the agent's
    resolveSocketDir / SocketPath logic for the unprivileged-trace case.
    """
    run_path = os.path.join(ANNOTATION_SOCKET_DIR, ANNOTATION_SOCKET_NAME)
    try:
        if os.path.exists(run_path) and _is_socket(run_path):
            return run_path
    except OSError:
        pass
    home = os.path.expanduser("~")
    return os.path.join(home, ".ingero", "annotate", ANNOTATION_SOCKET_NAME)


def _is_socket(path: str) -> bool:
    import stat
    try:
        return stat.S_ISSOCK(os.lstat(path).st_mode)
    except OSError:
        return False


class AnnotationWriter:
    """A reusable, thread-safe connection to the agent annotation socket.

    The writer opens the Unix-domain socket once and reuses it for every
    annotation. It is graceful by design: if the socket does not exist
    (the agent is not running with `--annotate`) or the connection drops,
    the writer degrades to a silent no-op after one log line. It never
    raises into the caller's hot path - a contract violation is the only
    thing surfaced, and only from the explicit `write` call.

    Typical use from a framework hook:

        w = AnnotationWriter()        # connects, or becomes a no-op
        w.write({"step": "10"}, pid=os.getpid())
        ...
        w.close()
    """

    def __init__(self, socket_path: str | None = None,
                 connect_timeout: float = 2.0):
        self._path = socket_path or default_socket_path()
        self._timeout = connect_timeout
        self._lock = threading.Lock()
        self._sock: socket.socket | None = None
        self._active = False
        self._logged_unavailable = False
        self._connect()

    @property
    def active(self) -> bool:
        """True when the socket is connected and annotations will be sent."""
        return self._active

    @property
    def socket_path(self) -> str:
        return self._path

    def _connect(self) -> None:
        try:
            s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            s.settimeout(self._timeout)
            s.connect(self._path)
            s.settimeout(None)
            self._sock = s
            self._active = True
        except OSError as exc:
            self._sock = None
            self._active = False
            if not self._logged_unavailable:
                logger.info(
                    "ingero annotation socket unavailable at %s (%s); "
                    "annotations disabled. Run the agent with "
                    "'trace --record --annotate' to enable.",
                    self._path, exc,
                )
                self._logged_unavailable = True

    def write(self, labels: dict, pid: int | None = None,
              ts_ns: int | None = None) -> bool:
        """Send one annotation. Returns True if it was written to the socket.

        A contract violation raises AnnotationError - that is a caller
        bug, surfaced loudly. A transport failure (socket gone, peer
        closed) is swallowed: the writer flips to no-op and returns False
        so a training loop is never interrupted by agent unavailability.
        """
        line = encode_annotation(labels, pid=pid, ts_ns=ts_ns)
        with self._lock:
            if not self._active or self._sock is None:
                return False
            try:
                self._sock.sendall(line)
                return True
            except OSError as exc:
                logger.info(
                    "ingero annotation socket write failed (%s); "
                    "annotations disabled for the rest of this run.", exc,
                )
                self._close_locked()
                return False

    def _close_locked(self) -> None:
        if self._sock is not None:
            try:
                self._sock.close()
            except OSError:
                pass
        self._sock = None
        self._active = False

    def close(self) -> None:
        """Close the socket. Safe to call more than once."""
        with self._lock:
            self._close_locked()

    def __enter__(self) -> "AnnotationWriter":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def now_ns() -> int:
    """Current time in unix nanoseconds, for an explicit annotation ts."""
    return time.time_ns()
