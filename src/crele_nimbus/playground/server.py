"""The playground's two servers: the page over HTTP, the sessions over a WebSocket.

Each WebSocket connection gets its own engine and session from the shared model; at most
`max_sessions` at once (one by default: a second page takes the model over and the first
is closed with code 4000). The model's work runs on one thread, so sessions never
interleave on it. No audio is stored by the server.
"""

from __future__ import annotations

import asyncio
import collections
import functools
import json
import logging
import threading
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from crele_nimbus.config import SAMPLE_RATE
from crele_nimbus.playground.engine import Engine, Message, PlaygroundError, _error

STATIC = Path(__file__).resolve().parent / "static"
MAX_MESSAGE = 8 * 1024 * 1024
"""The largest WebSocket message: a 30 s enrollment in base64 fits."""
TAKEN_OVER = 4000
"""The close code a page gets when another page takes the model."""

log = logging.getLogger("crele_nimbus.playground")


class Connection:
    """One page: its messages in arrival order, the engine's work on one thread."""

    def __init__(self, engine: Engine, ws: Any, executor: ThreadPoolExecutor) -> None:
        self.engine = engine
        self.ws = ws
        self.executor = executor
        self.items: collections.deque[tuple[str, Any]] = collections.deque()
        self.arrived = asyncio.Event()
        self.waiting = 0  # samples received and not yet pushed

    async def _call(self, fn: Callable[..., Any], *args: Any, **kw: Any) -> Any:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self.executor, functools.partial(fn, *args, **kw))

    async def _send(self, messages: Sequence[Message]) -> None:
        for m in messages:
            await self.ws.send(json.dumps(m, separators=(",", ":")))

    async def _read(self) -> None:
        try:
            async for msg in self.ws:
                if isinstance(msg, bytes):
                    self.waiting += len(msg) // 4
                    self.items.append(("audio", msg))
                else:
                    self.items.append(("control", msg))
                self.arrived.set()
        finally:
            self.items.append(("closed", None))
            self.arrived.set()

    async def run(self) -> None:
        from websockets.exceptions import ConnectionClosed

        reader = asyncio.create_task(self._read())
        try:
            await self._send([await self._call(self.engine.ready)])
            while True:
                if not self.items:
                    self.arrived.clear()
                    await self.arrived.wait()
                    continue
                kind, payload = self.items.popleft()
                if kind == "closed":
                    break
                if kind == "control":
                    await self._send(await self._call(self.engine.control, payload))
                    continue
                # Audio: everything queued behind it up to the next control message goes
                # in one push, so a slow step catches up.
                chunks = [payload]
                while self.items and self.items[0][0] == "audio":
                    chunks.append(self.items.popleft()[1])
                raw = b"".join(chunks)
                self.waiting -= len(raw) // 4
                lag_s = max(0, self.waiting) / SAMPLE_RATE
                try:
                    out = await self._call(self.engine.audio, raw, lag_s=lag_s)
                except PlaygroundError as exc:
                    out = [_error(str(exc), exc.where)]
                await self._send(out)
        except ConnectionClosed:
            pass
        except Exception:  # noqa: BLE001, one page's failure must not stop the server
            log.exception("the connection failed")
        finally:
            reader.cancel()
            await self._call(self.engine.close)


async def serve_ws(
    make_engine: Callable[[], Engine],
    host: str,
    port: int,
    *,
    origins: Sequence[str | None] | None = None,
    max_sessions: int = 1,
    started: threading.Event | None = None,
    stop: asyncio.Event | None = None,
) -> None:
    """The WebSocket server until `stop` is set (or forever). Each connection gets
    `make_engine()`; past `max_sessions`, the oldest page is closed first."""
    from websockets.asyncio.server import serve

    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="nimbus")
    active: collections.OrderedDict[Any, asyncio.Event] = collections.OrderedDict()

    async def handler(ws: Any) -> None:
        done = asyncio.Event()
        active[ws] = done
        try:
            while len(active) > max_sessions:
                old, old_done = next(iter(active.items()))
                if old is ws:
                    break
                # The old page lets go of its session before this one starts.
                await old.close(TAKEN_OVER, "another page took the model")
                await old_done.wait()
            if ws not in active:
                return
            log.info("a page connected")
            engine = await asyncio.get_running_loop().run_in_executor(executor, make_engine)
            await Connection(engine, ws, executor).run()
            log.info("a page left")
        finally:
            active.pop(ws, None)
            done.set()

    try:
        async with serve(handler, host, port, origins=origins, max_size=MAX_MESSAGE):
            if started is not None:
                started.set()
            await (stop.wait() if stop is not None else asyncio.Future())
    finally:
        executor.shutdown(wait=True)


def make_http_server(
    host: str, port: int, ws_port: int, static: Path = STATIC
) -> ThreadingHTTPServer:
    """The page's files, and `/config.json` with the WebSocket's port."""

    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *a: Any, **kw: Any) -> None:
            super().__init__(*a, directory=str(static), **kw)

        def do_GET(self) -> None:  # noqa: N802, the standard library's name
            if self.path.split("?")[0] == "/config.json":
                body = json.dumps({"ws_port": ws_port}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            super().do_GET()

        def end_headers(self) -> None:
            self.send_header("Cache-Control", "no-store")
            super().end_headers()

        def log_message(self, fmt: str, *a: Any) -> None:
            log.debug("http: " + fmt, *a)

    Handler.extensions_map = {
        **SimpleHTTPRequestHandler.extensions_map,
        ".js": "text/javascript",
        ".css": "text/css",
    }
    return ThreadingHTTPServer((host, port), Handler)


def origins_for(host: str, port: int) -> list[str | None]:
    """The page's own origins (and clients that send none, as a test client): a page on
    any other site cannot reach the model."""
    names = {host, "127.0.0.1", "localhost"}
    return [None, *sorted(f"http://{n}:{port}" for n in names)]
