"""The playground's engine: one session behind the page's protocol, without the network.

Control messages and audio come in, JSON messages go out; `server.py` moves them over a
WebSocket. The protocol is `README.md` in this folder. The engine is not thread-safe:
one caller at a time, as the server guarantees through its executor.
"""

from __future__ import annotations

import base64
import binascii
import json
import math
import re
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np

from crele_nimbus.config import SAMPLE_RATE, Settings
from crele_nimbus.session import SIGNAL_IDS, Event, Session, Update
from crele_nimbus.speaker import MIN_SAMPLES

MAX_TYPED = 6
"""Typed questions at once on the page: its lanes keep one colour each, eight in all."""
MAX_TEXT = 200
MAX_ENROLL_S = 30.0
CRITERIA_SEP = "||"
ID_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9_-]{0,31}$")
SETTINGS: dict[str, tuple[float, float]] = {
    "threshold": (0.1, 0.95),
    "start_secs": (0.04, 3.0),
    "stop_threshold": (0.05, 0.9),
    "stop_secs": (0.04, 10.0),
}
"""The four event settings a page may set per question, with the range each accepts."""
STEP_BUDGET_MS = 80.0
MEASURE_QUESTIONS = (
    "Is music playing?",
    "Is someone whistling?",
    "Is someone laughing?",
    "Is a dog barking?",
    "Is it noisy? || loud background noise, traffic or machines || silence",
    "Is someone typing on a keyboard?",
)
"""The step is timed with `MAX_TYPED` questions, the most the page asks at once."""

Message = dict[str, Any]


class PlaygroundError(ValueError):
    """A request the playground refuses; `where` names the part of the protocol."""

    def __init__(self, message: str, where: str = "protocol") -> None:
        super().__init__(message)
        self.where = where


def split_criteria(text: str) -> tuple[str, str | None, str | None]:
    """`text || yes means || no means` into its parts; plain text has no criteria."""
    if CRITERIA_SEP not in text:
        return text.strip(), None, None
    parts = [p.strip() for p in text.split(CRITERIA_SEP)]
    if len(parts) != 3 or not all(parts):
        raise PlaygroundError(
            f"criteria are written as `question {CRITERIA_SEP} what yes means "
            f"{CRITERIA_SEP} what no means`",
            "questions",
        )
    return parts[0], parts[1], parts[2]


def measure_step(session: Session, *, warmup: int = 70, steps: int = 25) -> dict[str, float]:
    """Mean milliseconds per 80 ms step on silence, by part and in total; the session is
    reset after. The default warm-up runs past 5.04 s, where the encoder's second stream
    starts and the step reaches its steady cost."""
    zeros = np.zeros(session.step_samples, dtype=np.float32)
    for _ in range(warmup):
        session.push(zeros)
    before, n0 = dict(session.seconds), session.steps
    for _ in range(steps):
        session.push(zeros)
    n = session.steps - n0
    out = {p: (session.seconds[p] - before[p]) * 1000 / n for p in before}
    out["total"] = sum(out.values())
    session.reset()
    return out


def _round(x: float, n: int = 4) -> float | None:
    return round(float(x), n) if math.isfinite(x) else None


def _pcm(raw: bytes | bytearray, where: str) -> np.ndarray:
    if len(raw) % 4:
        raise PlaygroundError(f"{where}: float32 PCM, got {len(raw)} bytes", where)
    wave = np.frombuffer(bytes(raw), dtype="<f4").astype(np.float32)
    if not np.all(np.isfinite(wave)):
        raise PlaygroundError(f"{where}: the samples must be finite", where)
    return wave


def _settings_of(d: Any, where: str) -> dict[str, float]:
    """A question's settings from a message, checked against the ranges."""
    if d is None:
        return {}
    if not isinstance(d, Mapping):
        raise PlaygroundError(f"{where}: the settings are an object", "questions")
    out: dict[str, float] = {}
    for name, (lo, hi) in SETTINGS.items():
        v = d.get(name)
        if v is None:
            continue
        if isinstance(v, bool) or not isinstance(v, int | float) or not math.isfinite(v):
            raise PlaygroundError(f"{where}.{name}: not a number: {v!r}", "questions")
        if not lo <= v <= hi:
            raise PlaygroundError(f"{where}.{name}: {v} is outside {lo} to {hi}", "questions")
        out[name] = float(v)
    return out


def _error(message: str, where: str = "protocol") -> Message:
    return {"type": "error", "message": message, "where": where}


def _event_message(e: Event, text: str) -> Message:
    msg: Message = {
        "type": "event", "event": e.kind, "question": e.question, "text": text,
        "start_s": round(e.start_s, 3), "confirmed_s": round(e.confirmed_s, 3),
    }  # fmt: skip
    if e.kind == "stopped":
        msg["end_s"] = round(float(e.end_s), 3)
        msg["duration_s"] = round(float(e.duration_s), 3)
    return msg


class Engine:
    """One session behind the protocol (module docstring)."""

    def __init__(
        self,
        session: Session,
        *,
        model_name: str,
        step_ms: Mapping[str, float],
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self.session = session
        self.model_name = model_name
        self.step_ms = dict(step_ms)
        self.clock = clock
        self.typed: list[dict[str, Any]] = []
        self._texts: dict[str, str] = {}
        self.custom: dict[str, dict[str, float]] = {}
        self.state = "idle"
        self.source: str | None = None
        self.name: str | None = None
        self._idle_warned = False
        self._reset_stats()

    # ------------------------------------------------------------------ reading

    def ready(self) -> Message:
        defaults = self.session.model.config.defaults
        questions = []
        by_id = {q["id"]: q for q in self.typed}
        for qid, text in self.session.questions:
            s = self.session.settings_of(qid)
            entry = by_id.get(qid, {})
            questions.append(
                {
                    "id": qid,
                    "kind": "signal" if qid in SIGNAL_IDS else "typed",
                    "text": entry.get("text", text),
                    "yes_means": entry.get("yes_means"),
                    "no_means": entry.get("no_means"),
                    **{k: _round(v) for k, v in s.as_dict().items()},
                    "cant_tell_below": self.session.model.config.cant_tell_below,
                    "custom": dict(self.custom.get(qid, {})),
                }
            )
        return {
            "type": "ready",
            "state": self.state,
            "source": self.source,
            "name": self.name,
            "model": {"name": self.model_name, "device": str(self.session.device)},
            "step_ms": {k: _round(v, 2) for k, v in self.step_ms.items()},
            "budget_ms": STEP_BUDGET_MS,
            "defaults": defaults.as_dict(),
            "questions": questions,
            "max_typed": MAX_TYPED,
            "enrolled": self.session.enrolled,
            "enroll_s": self.session.enrollment_s,
            "sample_rate": SAMPLE_RATE,
            "chunk_samples": self.session.step_samples,
        }

    # ------------------------------------------------------------------ control

    def control(self, raw: str | Mapping[str, Any]) -> list[Message]:
        """One control message; the replies in order. A refused request gives one
        `error` and changes nothing."""
        try:
            msg = json.loads(raw) if isinstance(raw, str) else dict(raw)
        except json.JSONDecodeError as exc:
            return [_error(f"not JSON: {exc}")]
        if not isinstance(msg, dict):
            return [_error("a control message is a JSON object")]
        kind = msg.get("type")
        handlers = {
            "questions": self._questions,
            "enroll": self._enroll,
            "start": self._start,
            "wav": self._start,
            "stop": self._stop,
        }
        if kind not in handlers:
            return [_error(f"unknown message type {kind!r}; known: {sorted(handlers)}")]
        try:
            return handlers[kind](msg)
        except PlaygroundError as exc:
            return [_error(str(exc), exc.where)]

    def set_typed(
        self,
        typed: Sequence[Mapping[str, Any]],
        signals: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> list[Message]:
        """The typed questions and every setting. A question that stays keeps its events;
        one that goes has its open event stopped, sent with the next answers."""
        if not isinstance(typed, list | tuple):
            raise PlaygroundError("questions: expected a list", "questions")
        if len(typed) > MAX_TYPED:
            raise PlaygroundError(
                f"at most {MAX_TYPED} questions at once, got {len(typed)}", "questions"
            )
        entries: list[dict[str, Any]] = []
        custom: dict[str, dict[str, float]] = {}
        for i, q in enumerate(typed):
            if not isinstance(q, Mapping):
                raise PlaygroundError(f"questions[{i}]: expected an object", "questions")
            qid = q.get("id")
            if not isinstance(qid, str) or not ID_PATTERN.match(qid) or qid in SIGNAL_IDS:
                raise PlaygroundError(
                    f"questions[{i}].id: {qid!r} is not an id (a letter, then letters, digits, "
                    f"_ or -; not {SIGNAL_IDS[0]} or {SIGNAL_IDS[1]})",
                    "questions",
                )
            if any(e["id"] == qid for e in entries):
                raise PlaygroundError(f"questions: the id {qid!r} is used twice", "questions")
            text = str(q.get("text", "")).strip()
            if len(text) > MAX_TEXT:
                raise PlaygroundError(f"questions[{i}]: at most {MAX_TEXT} characters", "questions")
            shown, yes_means, no_means = split_criteria(text)
            if q.get("yes_means") or q.get("no_means"):
                if yes_means is not None:
                    raise PlaygroundError(
                        f"{CRITERIA_SEP!r} in the text and yes_means beside it: give one",
                        "questions",
                    )
                yes_means = str(q.get("yes_means") or "").strip() or None
                no_means = str(q.get("no_means") or "").strip() or None
            if not shown:
                raise PlaygroundError(f"questions[{i}]: the question is empty", "questions")
            entries.append({"id": qid, "text": shown, "yes_means": yes_means, "no_means": no_means})
            own = _settings_of(q, f"questions[{i}]")
            if own:
                custom[qid] = own
        for sid, d in (signals or {}).items():
            if sid not in SIGNAL_IDS:
                raise PlaygroundError(f"signals: no signal {sid!r}; they are user and other")
            own = _settings_of(d, f"signals.{sid}")
            if own:
                custom[sid] = own
        # Checked before anything changes: every setting resolves over the defaults.
        defaults = self.session.model.config.defaults
        full: dict[str, Settings] = {}
        for e in entries:
            full[e["id"]] = _resolve(defaults, custom.get(e["id"], {}), e["id"])
        for sid in SIGNAL_IDS:
            full[sid] = _resolve(defaults, custom.get(sid, {}), sid)
        wanted = {e["id"] for e in entries}
        for qid, _ in self.session.questions:
            if qid not in SIGNAL_IDS and qid not in wanted:
                self.session.remove(qid)
        current = {qid: text for qid, text in self.session.questions}
        for e in entries:
            s = full[e["id"]]
            if e["id"] in current:
                self.session.tune(e["id"], **s.as_dict())
            try:
                self.session.ask(
                    e["id"], e["text"], yes_means=e["yes_means"], no_means=e["no_means"],
                    **s.as_dict(),
                )  # fmt: skip
            except ValueError as exc:
                raise PlaygroundError(f"{e['text']}: {exc}", "questions") from exc
        for sid in SIGNAL_IDS:
            self.session.tune(sid, **full[sid].as_dict())
        self.typed = entries
        self.custom = custom
        self._texts.update({e["id"]: e["text"] for e in entries})
        current_ids = {qid for qid, _ in self.session.questions}
        self.stats = {k: v for k, v in self.stats.items() if k in current_ids}
        for qid in current_ids:
            self._stats_for(qid)
        return []

    def _questions(self, msg: Mapping[str, Any]) -> list[Message]:
        out = self.set_typed(msg.get("questions", []), msg.get("signals"))
        return [*out, self.ready()]

    def _enroll(self, msg: Mapping[str, Any]) -> list[Message]:
        pcm = msg.get("pcm")
        if pcm is None:
            self.session.enroll(None)
            return [self.ready()]
        if not isinstance(pcm, str):
            raise PlaygroundError("enroll.pcm: base64 float32 PCM, or null", "enroll")
        try:
            raw = base64.b64decode(pcm, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise PlaygroundError(f"enroll.pcm: not base64 ({exc})", "enroll") from exc
        wave = _pcm(raw, "enroll")
        if wave.shape[0] < MIN_SAMPLES:
            raise PlaygroundError(
                f"the recording is {wave.shape[0] / SAMPLE_RATE:.2f} s; it needs at least "
                f"{MIN_SAMPLES / SAMPLE_RATE:.2f} s of speech",
                "enroll",
            )
        if wave.shape[0] > MAX_ENROLL_S * SAMPLE_RATE:
            raise PlaygroundError(
                f"the recording is {wave.shape[0] / SAMPLE_RATE:.1f} s; "
                f"at most {MAX_ENROLL_S:.0f} s",
                "enroll",
            )
        try:
            self.session.enroll(wave)
        except ValueError as exc:
            raise PlaygroundError(str(exc), "enroll") from exc
        return [self.ready()]

    def _start(self, msg: Mapping[str, Any]) -> list[Message]:
        out: list[Message] = []
        if self.state == "listening":
            out += self._finish(stopped_by="a new stream")
        if msg.get("type") == "wav":
            name = msg.get("name")
            self.source, self.name = "wav", str(name)[:200] if name else "a file"
        else:
            source = msg.get("source", "mic")
            if source != "mic":
                raise PlaygroundError(f"start.source: mic, got {source!r}", "start")
            self.source, self.name = "mic", None
        self.session.reset()
        self._reset_stats()
        self.state = "listening"
        self._idle_warned = False
        return [*out, self.ready()]

    def _stop(self, msg: Mapping[str, Any]) -> list[Message]:
        if self.state != "listening":
            raise PlaygroundError("stop: not listening", "stop")
        return [*self._finish(stopped_by=str(msg.get("reason") or "stop")), self.ready()]

    def close(self) -> None:
        """The page went away: a stream still running ends, nothing is sent."""
        if self.state == "listening":
            self._finish(stopped_by="the page closed")

    # ------------------------------------------------------------------ audio

    def audio(self, data: bytes | np.ndarray, *, lag_s: float = 0.0) -> list[Message]:
        """PCM of any length (float32 LE bytes or an array): the answers and events of
        every whole step it completes."""
        if self.state != "listening":
            if self._idle_warned:
                return []
            self._idle_warned = True
            raise PlaygroundError("audio before start: send start or wav first", "audio")
        wave = _pcm(data, "audio") if isinstance(data, bytes | bytearray) else data
        if wave.size == 0:
            return []
        t0 = self.clock()
        updates = self.session.push(np.asarray(wave, dtype=np.float32))
        self.model_s += self.clock() - t0
        self.audio_in += int(wave.shape[0])
        return self._messages(updates, lag_s=lag_s)

    # ------------------------------------------------------------------ internals

    def _reset_stats(self) -> None:
        self.stats: dict[str, dict[str, Any]] = {}
        self.model_s = 0.0
        self.audio_in = 0
        for qid, _ in self.session.questions:
            self._stats_for(qid)

    def _stats_for(self, qid: str) -> None:
        rec = self.stats.setdefault(qid, {"events": 0, "on_s": 0.0})
        rec["text"] = self._text(qid)
        rec["kind"] = "signal" if qid in SIGNAL_IDS else "typed"

    def _text(self, qid: str) -> str:
        if qid in self._texts:
            return self._texts[qid]
        return dict(self.session.questions).get(qid, qid)

    def _event(self, e: Event) -> Message:
        stats = self.stats.get(e.question)  # None for a question removed meanwhile
        if stats is not None:
            if e.kind == "stopped":
                stats["on_s"] += float(e.duration_s)
            else:
                stats["events"] += 1
        return _event_message(e, self._text(e.question))

    def _messages(self, updates: Sequence[Update], *, lag_s: float) -> list[Message]:
        if not updates:
            return []
        events: list[Message] = []
        steps = []
        for u in updates:
            steps.append(
                {
                    "t": u.frame,
                    "t_s": round(u.time_s, 3),
                    "answers": [
                        {"id": a.question, "p": _round(a.probability),
                         "p_can_tell": _round(a.can_tell), "state": a.answer}
                        for a in u.answers.values()
                    ],
                }
            )  # fmt: skip
            events += [self._event(e) for e in u.events]
        msg = {
            "type": "decisions",
            "steps": steps,
            "step_ms": _round(self.session.timing["total"], 2),
            "lag_s": round(float(lag_s), 3),
        }
        # The answers first, so a lane has its trace when the event arrives.
        return [msg, *events]

    def _finish(self, *, stopped_by: str) -> list[Message]:
        t0 = self.clock()
        updates = self.session.finish()
        self.model_s += self.clock() - t0
        out = self._messages(updates, lag_s=0.0)
        open_events = []
        for e in self.session.events:
            if e.kind == "started":
                self.stats[e.question]["on_s"] += self.session.time_s - e.start_s
                open_events.append(_event_message(e, self._text(e.question)))
        audio_s = self.session.time_s
        timing = self.session.timing
        summary = {
            "type": "summary",
            "source": self.source,
            "name": self.name,
            "stopped_by": stopped_by,
            "audio_s": round(audio_s, 3),
            "steps": self.session.steps,
            "step_ms": {k: _round(v, 2) for k, v in timing.items()},
            "realtime_factor": _round(audio_s / self.model_s, 2) if self.model_s > 0 else None,
            "questions": [
                {"id": qid, "text": s["text"], "kind": s["kind"], "events": s["events"],
                 "on_s": round(s["on_s"], 3)}
                for qid, s in self.stats.items()
            ],
            "open": open_events,
        }  # fmt: skip
        self.state = "idle"
        return [*out, summary]


def _resolve(defaults: Settings, custom: Mapping[str, float], qid: str) -> Settings:
    try:
        return defaults.override(f"settings of {qid!r}", **custom)
    except ValueError as exc:
        raise PlaygroundError(str(exc), "questions") from exc
