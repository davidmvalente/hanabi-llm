"""Minimal live viewer for a game feed.

    uv run hanabi-view game.jsonl            # then open http://127.0.0.1:8765

The server tails the JSONL feed (so it follows a game that is still being
played) and streams, over Server-Sent Events, one frame per record:

    {"i": 12, "record": <the feed record>, "state": <the folded state after it>}

The state comes from ``fold.py``, so the page contains no game logic: it only
draws what it is sent. Reconnecting replays from frame 0, so a reload is safe.
Binds to localhost by default; the feed path is fixed at start-up and never
taken from a request.
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Iterator
from functools import partial
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources
from pathlib import Path
from typing import Any

from .fold import FoldState, initial_state, step
from .narrate import segments

POLL_SECONDS = 0.2
KEEPALIVE_SECONDS = 15.0


def frames(path: Path, *, wait_for_file: float = 30.0, follow: bool = True) -> Iterator[dict[str, Any]]:
    """Yield a frame per complete line of the feed, then keep following the file until a result arrives."""
    deadline = time.monotonic() + wait_for_file
    while not path.exists():
        if time.monotonic() > deadline:
            return
        time.sleep(POLL_SECONDS)
    state: FoldState | None = None
    pending, index, last_beat = "", 0, time.monotonic()
    with path.open("r", encoding="utf-8") as fh:
        while True:
            chunk = fh.readline()
            if not chunk:
                if not follow:
                    return
                if time.monotonic() - last_beat > KEEPALIVE_SECONDS:
                    last_beat = time.monotonic()
                    yield {"keepalive": True}
                time.sleep(POLL_SECONDS)
                continue
            pending += chunk
            if not pending.endswith("\n"):
                continue  # a partial line: the writer has not finished it yet
            line, pending = pending.strip(), ""
            if not line:
                continue
            rec = json.loads(line)
            if rec["type"] == "header":
                state = initial_state(rec)
            elif state is None:
                raise ValueError("feed does not start with a header")
            else:
                step(state, rec)
            assert state is not None
            frame: dict[str, Any] = {"i": index, "record": rec, "state": state.to_json()}
            if rec["type"] == "event":
                frame["segments"] = segments(rec)  # the wording is defined in narrate.py, not in the page
            yield frame
            index += 1
            if rec["type"] == "result":
                return


class _Handler(BaseHTTPRequestHandler):
    def __init__(self, feed: Path, *args: Any, **kwargs: Any) -> None:
        self.feed = feed
        super().__init__(*args, **kwargs)

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 - keep the console quiet
        pass

    def do_GET(self) -> None:  # noqa: N802 - http.server API
        route = self.path.split("?", 1)[0]
        if route == "/":
            body = resources.files("hanabi_llm").joinpath("web", "index.html").read_bytes()
            self._send(200, "text/html; charset=utf-8", body)
        elif route == "/stream":
            self._stream()
        else:
            self._send(404, "text/plain", b"not found")

    def _send(self, status: int, content_type: str, body: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _stream(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()
        try:
            for frame in frames(self.feed):
                if frame.get("keepalive"):
                    self.wfile.write(b": keep-alive\n\n")
                else:
                    self.wfile.write(f"data: {json.dumps(frame, ensure_ascii=False)}\n\n".encode())
                self.wfile.flush()
            self.wfile.write(b"event: done\ndata: {}\n\n")  # the game is over: the page stops listening
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass  # the browser went away


def make_server(feed: Path, host: str = "127.0.0.1", port: int = 8765) -> ThreadingHTTPServer:
    server = ThreadingHTTPServer((host, port), partial(_Handler, feed))
    server.daemon_threads = True
    return server


def main() -> None:
    ap = argparse.ArgumentParser(description="Follow a Hanabi game feed in the browser.")
    ap.add_argument("feed", type=Path, help="the JSONL written by `hanabi-llm --log`")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()
    server = make_server(args.feed, args.host, args.port)
    print(f"Following {args.feed}\nOpen http://{args.host}:{server.server_address[1]}  (Ctrl+C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print()
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
