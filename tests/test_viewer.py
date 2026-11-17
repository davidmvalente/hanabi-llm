"""The viewer server: it streams the feed folded into states, and follows a game still being written."""

from __future__ import annotations

import asyncio
import http.client
import json
import threading
from collections.abc import Iterator
from pathlib import Path

import pytest

from hanabi_llm.__main__ import build_parser, run
from hanabi_llm.feed import read_feed
from hanabi_llm.fold import fold
from hanabi_llm.viewer import frames, make_server


def played_feed(tmp_path: Path, seed: int = 4) -> Path:
    path = tmp_path / "g.jsonl"
    asyncio.run(run(build_parser().parse_args(["--models", "random,random,random", "--seed", str(seed), "--log", str(path)])))
    return path


@pytest.fixture
def server(tmp_path: Path) -> Iterator[tuple[http.server.ThreadingHTTPServer, Path]]:  # type: ignore[name-defined]  # noqa: F821
    path = played_feed(tmp_path)
    srv = make_server(path, port=0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield srv, path
    srv.shutdown()
    srv.server_close()


def stream(port: int) -> list[dict]:  # type: ignore[type-arg]
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    conn.request("GET", "/stream")
    resp = conn.getresponse()
    assert resp.status == 200 and resp.getheader("Content-Type") == "text/event-stream"
    out, done = [], False
    for raw in resp.read().decode().split("\n\n"):
        if raw.startswith("data: "):
            out.append(json.loads(raw[6:]))
        elif raw.startswith("event: done"):
            done = True
    assert done, "the stream must end with a done event once the result has been sent"
    return out


def test_the_stream_is_the_feed_with_the_folded_state_after_every_record(server) -> None:  # type: ignore[no-untyped-def]
    srv, path = server
    feed = read_feed(path)
    got = stream(srv.server_address[1])
    assert [f["record"] for f in got] == feed and [f["i"] for f in got] == list(range(len(feed)))
    for f in got[::7] + [got[-1]]:                                   # sampled, and the last
        assert f["state"] == fold(feed[: f["i"] + 1]).to_json()      # the page draws exactly the fold
    assert got[-1]["record"]["type"] == "result"


def test_the_page_is_served_and_unknown_paths_are_refused(server) -> None:  # type: ignore[no-untyped-def]
    srv, _ = server
    conn = http.client.HTTPConnection("127.0.0.1", srv.server_address[1], timeout=5)
    conn.request("GET", "/")
    page = conn.getresponse()
    html = page.read().decode()
    assert page.status == 200 and "EventSource" in html and "<title>Hanabi viewer</title>" in html
    assert "https://" not in html and "http://" not in html            # no external requests
    for bad in ("/../pyproject.toml", "/etc/passwd", "/stream/../x"):
        conn.request("GET", bad)
        resp = conn.getresponse(); resp.read()
        assert resp.status == 404


def test_it_follows_a_feed_that_is_still_being_written(tmp_path: Path) -> None:
    done = played_feed(tmp_path, seed=6)
    lines = done.read_text().splitlines()
    live = tmp_path / "live.jsonl"
    live.write_text(lines[0] + "\n" + lines[1][:20])                    # the header, and half of the next line
    got: list[dict] = []  # type: ignore[type-arg]
    thread = threading.Thread(target=lambda: got.extend(frames(live)), daemon=True)
    thread.start()
    thread.join(0.6)
    assert [f["i"] for f in got] == [0], "a half-written line must wait until it is complete"
    with live.open("a") as fh:
        fh.write(lines[1][20:] + "\n")
        fh.flush()
        thread.join(0.6)
        assert [f["i"] for f in got] == [0, 1]
        fh.write("\n".join(lines[2:]) + "\n")
    thread.join(5)
    assert not thread.is_alive() and got[-1]["record"]["type"] == "result"   # it stops after the result
    assert len(got) == len(lines)


def test_a_missing_feed_ends_quietly(tmp_path: Path) -> None:
    assert list(frames(tmp_path / "nope.jsonl", wait_for_file=0.3)) == []
