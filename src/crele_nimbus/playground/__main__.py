"""The playground: the model in a browser page, one command.

    python -m crele_nimbus.playground [--open]

Then open http://127.0.0.1:8780: one microphone button, or play an audio file, and a
board with a row per question. The first run downloads about 1.6 GB (the model and its
text encoder) into the Hugging Face cache.

Exit codes: 0 stopped (ctrl-c); 2 refused (a flag, the weights, a port in use).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import re
import sys
import threading
import webbrowser
from pathlib import Path

from crele_nimbus.playground.engine import MEASURE_QUESTIONS, Engine, measure_step
from crele_nimbus.playground.server import STATIC, make_http_server, origins_for, serve_ws

EXIT_OK, EXIT_REFUSED = 0, 2
HOST, HTTP_PORT, WS_PORT = "127.0.0.1", 8780, 8781
MODEL_NAME = "Crele Nimbus v1"


def read_defaults(index: Path = STATIC / "index.html") -> list[str]:
    """The questions a first visit starts with, from the page's defaults block."""
    html = index.read_text(encoding="utf-8")
    m = re.search(r'<script type="application/json" id="defaults">(.*?)</script>', html, re.S)
    if m is None:
        raise ValueError(f"{index}: no defaults block")
    texts = json.loads(m.group(1))
    if not isinstance(texts, list) or not all(isinstance(t, str) and t.strip() for t in texts):
        raise ValueError(f"{index}: the defaults are a list of question texts")
    return texts


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="python -m crele_nimbus.playground", description=__doc__.splitlines()[0]
    )
    ap.add_argument(
        "--weights", default="crele/nimbus-v1",
        help="a Hugging Face repository id or a local folder (default crele/nimbus-v1)",
    )  # fmt: skip
    ap.add_argument(
        "--text-encoder", default=None,
        help="a local folder with the text encoder's files (default: fetched)",
    )  # fmt: skip
    ap.add_argument("--device", default="cpu", help="cpu or cuda (default cpu)")
    ap.add_argument("--host", default=HOST)
    ap.add_argument("--port", type=int, default=HTTP_PORT, help="the page (default 8780)")
    ap.add_argument("--ws-port", type=int, default=WS_PORT, help="the sessions (default 8781)")
    ap.add_argument(
        "--max-sessions", type=int, default=1,
        help="pages at once; past it the oldest page is closed (default 1)",
    )  # fmt: skip
    ap.add_argument("--open", action="store_true", help="open the page in the browser")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    if args.max_sessions < 1:
        print("error: --max-sessions must be at least 1", file=sys.stderr)
        return EXIT_REFUSED
    if not (STATIC / "index.html").is_file():
        print(f"error: no page at {STATIC / 'index.html'}", file=sys.stderr)
        return EXIT_REFUSED
    from crele_nimbus import Nimbus
    from crele_nimbus.config import ConfigError
    from crele_nimbus.weights import WeightsError

    print(
        "loading the model (the first run downloads about 1.6 GB into the Hugging Face cache)",
        flush=True,
    )
    try:
        model = Nimbus.from_pretrained(
            args.weights, device=args.device, text_encoder=args.text_encoder
        )
        model.encode(list(MEASURE_QUESTIONS) + read_defaults())
        session = model.session()
        for i, q in enumerate(MEASURE_QUESTIONS, 1):
            parts = [p.strip() for p in q.split("||")]
            session.ask(f"m{i}", parts[0], yes_means=parts[1] if len(parts) == 3 else None,
                        no_means=parts[2] if len(parts) == 3 else None)  # fmt: skip
        step_ms = measure_step(session)
    except (WeightsError, ConfigError, ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_REFUSED
    print(
        f"model step: {step_ms['total']:.1f} ms per 80 ms of audio on this machine "
        f"(encoder {step_ms['listener']:.1f}, voice {step_ms['voice']:.1f}, head "
        f"{step_ms['head']:.1f} with {len(MEASURE_QUESTIONS)} questions)"
    )
    if step_ms["total"] > 80:
        print("over real time: the answers will run behind live audio on this machine")

    def make_engine() -> Engine:
        return Engine(model.session(), model_name=MODEL_NAME, step_ms=step_ms)

    try:
        httpd = make_http_server(args.host, args.port, args.ws_port)
    except OSError as exc:
        print(f"error: the page's port {args.port}: {exc}", file=sys.stderr)
        return EXIT_REFUSED
    threading.Thread(target=httpd.serve_forever, name="http", daemon=True).start()
    url = f"http://{args.host}:{args.port}"
    print(f"the playground is at {url} (ctrl-c to stop)", flush=True)
    if args.open:
        webbrowser.open(url)
    code = EXIT_OK
    try:
        asyncio.run(
            serve_ws(
                make_engine,
                args.host,
                args.ws_port,
                origins=origins_for(args.host, args.port),
                max_sessions=args.max_sessions,
            )  # fmt: skip
        )
    except KeyboardInterrupt:
        pass
    except OSError as exc:
        print(f"error: the sessions' port {args.ws_port}: {exc}", file=sys.stderr)
        code = EXIT_REFUSED
    finally:
        httpd.shutdown()
        httpd.server_close()
    print("stopped")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
