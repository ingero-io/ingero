"""Shared pytest fixtures: a fake Ingero annotation socket server.

The fake server is a plain Unix-domain stream socket that collects every
NDJSON line a writer sends. It stands in for a running agent, so the
tests exercise the real socket path without `ingero trace`.
"""

from __future__ import annotations

import os
import socket
import threading

import pytest


class FakeAnnotationServer:
    """A minimal Unix-domain socket server collecting NDJSON lines.

    It mimics only what a writer observes: it accepts connections and
    reads newline-delimited bytes. It does not validate (the writer
    validates locally; the real agent validates server-side).
    """

    def __init__(self, path: str):
        self.path = path
        self.lines: list[bytes] = []
        self._sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._sock.bind(path)
        self._sock.listen(8)
        self._stop = False
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()

    def _serve(self) -> None:
        self._sock.settimeout(0.2)
        while not self._stop:
            try:
                conn, _ = self._sock.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            threading.Thread(
                target=self._handle, args=(conn,), daemon=True
            ).start()

    def _handle(self, conn: socket.socket) -> None:
        buf = b""
        with conn:
            conn.settimeout(0.5)
            while not self._stop:
                try:
                    chunk = conn.recv(4096)
                except socket.timeout:
                    continue
                except OSError:
                    break
                if not chunk:
                    break
                buf += chunk
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    with self._lock:
                        self.lines.append(line)

    def received(self) -> list[bytes]:
        with self._lock:
            return list(self.lines)

    def stop(self) -> None:
        self._stop = True
        try:
            self._sock.close()
        except OSError:
            pass
        try:
            os.unlink(self.path)
        except OSError:
            pass


@pytest.fixture
def fake_server(tmp_path):
    """Yield a running FakeAnnotationServer bound under a temp dir."""
    path = str(tmp_path / "annotate.sock")
    server = FakeAnnotationServer(path)
    try:
        yield server
    finally:
        server.stop()
