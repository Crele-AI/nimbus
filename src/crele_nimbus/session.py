"""A session: audio in, an answer per question every 40 ms and events out.

```python
session = model.session()
session.ask("music", "Is music playing?")
for update in session.push(pcm):          # float32 mono 16 kHz, any length
    for event in update.events:
        print(event.question, event.kind, event.start_s)
```

**Audio in.** `push` takes 16 kHz mono audio of any length (float32 in [-1, 1], int16,
or raw little-endian 16-bit PCM bytes) and consumes it 80 ms at a time; the remainder
waits for the next push. Each 80 ms step gives two 40 ms frames, each handed over as its
own `Update` with its time, so an answer arrives 80 to 120 ms of audio after the moment it
describes, plus compute.
`finish` consumes what is left at the end of a stream; `reset` starts a new stream with
the same questions and enrollment.

**Questions.** `ask(id, text)` adds a question or replaces one with the same id. Two
voice signals are always there, first: `user` (the enrolled voice, which answers
`cant_tell` until `enroll` has heard it) and `other` (any other voice). Each question
has the four event settings; `tune` changes them while the stream runs without
disturbing its events. Adding or removing a question leaves the other questions' events
running; a removed or replaced question's open event is stopped, and the `stopped` event
arrives with the next update.

**What comes out.** Per frame an `Update` with `answers` by id (`Answer`: `yes`, `no` or
`cant_tell`, the model's `probability` of yes, `can_tell`) and the `events` that started
or stopped at that frame (`Event`). `session.answers` holds the newest answers and
`session.events` every event of the stream so far.

The session is a pure function of the audio, the questions and the enrollment: the same
audio pushed in any chunking gives the same answers. It is not thread-safe; call it from
one thread, as the playground does through an executor.
"""

from __future__ import annotations

import re
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from torch import Tensor

from crele_nimbus.config import FRAME_HZ, FRAME_SAMPLES, SAMPLE_RATE, Settings
from crele_nimbus.events import EventLayer, FrameEvent, Params
from crele_nimbus.head import audio_side
from crele_nimbus.listener import StaggeredStream
from crele_nimbus.speaker import MIN_SAMPLES
from crele_nimbus.text import question_text

if TYPE_CHECKING:
    from crele_nimbus.model import Nimbus

SIGNAL_IDS = ("user", "other")
MAX_QUESTIONS = 32
"""Typed questions per session, beyond the two voice signals."""
ID_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9_-]{0,63}$")
PARTS = ("listener", "voice", "head")
FRAME_READY_SAMPLES = FRAME_SAMPLES - 160
"""Samples frame 0 needs: 480; frame t needs `t * 640 + 480`."""


@dataclass(frozen=True)
class Answer:
    """One question's answer at one frame."""

    question: str
    answer: str
    """`yes`, `no` or `cant_tell`."""
    probability: float
    """The model's probability of yes."""
    can_tell: float
    """The probability that the model can tell at all; under 0.5 the answer is held."""
    state_since_s: float | None
    """When the current yes or no began, in seconds of the stream."""


@dataclass(frozen=True)
class Event:
    """A question turning yes (`started`) or back to no (`stopped`)."""

    question: str
    kind: str
    """`started` or `stopped`."""
    start_s: float
    """The estimated start of the sound, in seconds of the stream."""
    confirmed_s: float
    """When the start was confirmed and reported."""
    end_s: float | None = None
    """`stopped` only: the end, in seconds of the stream."""
    duration_s: float | None = None


@dataclass(frozen=True)
class Update:
    """The answers of one 40 ms frame."""

    time_s: float
    """The end of the frame, in seconds of the stream."""
    frame: int
    answers: dict[str, Answer]
    events: list[Event]


@dataclass
class _Question:
    id: str
    text: str
    """The text as asked."""
    embedded: str
    """What the model reads: the text, with the criteria when given."""
    yes_means: str | None
    no_means: str | None
    settings: Settings
    is_signal: bool


def _as_audio(x: Any, what: str) -> np.ndarray:
    """Mono 16 kHz audio as float32: float arrays as they are, int16 arrays and raw
    16-bit PCM bytes scaled by 1/32768."""
    if isinstance(x, bytes | bytearray | memoryview):
        raw = bytes(x)
        if len(raw) % 2:
            raise ValueError(f"{what}: raw PCM must be 16-bit samples, got {len(raw)} bytes")
        x = np.frombuffer(raw, dtype="<i2")
    if isinstance(x, Tensor):
        x = x.detach().cpu().numpy()
    arr = np.asarray(x)
    if arr.dtype == np.int16:
        arr = arr.astype(np.float32) / 32768.0
    elif arr.dtype.kind not in "f":
        raise ValueError(
            f"{what}: expected float32 samples in [-1, 1], int16 samples or 16-bit PCM bytes, "
            f"got {arr.dtype}"
        )
    if arr.ndim == 2 and 1 in arr.shape:
        arr = arr.reshape(-1)
    if arr.ndim != 1:
        raise ValueError(f"{what}: expected mono audio as a 1-D array, got shape {arr.shape}")
    arr = arr.astype(np.float32, copy=False)
    if arr.size and not np.all(np.isfinite(arr)):
        raise ValueError(f"{what}: the samples must be finite")
    return arr


class Session:
    """One stream of audio through the model (module docstring). Made by
    `Nimbus.session()`."""

    def __init__(self, model: Nimbus) -> None:
        self.model = model
        cfg = model.config
        self.device = model.device
        self.chunk_frames = cfg.chunk_frames
        self.step_samples = cfg.chunk_samples
        self.n_frames = int(cfg.head["context_frames"])
        self.n_spk = int(cfg.head["voice_slots"])
        self.spk_dim = int(cfg.head["spk_dim"])
        self.spk_window = int(round(cfg.voice["window_s"] * SAMPLE_RATE))
        self.spk_hop = int(round(cfg.voice["hop_s"] * SAMPLE_RATE))
        if self.spk_window < MIN_SAMPLES:
            raise ValueError("the voice window is shorter than the voice model's shortest clip")
        if self.spk_hop <= 0 or self.spk_hop % FRAME_SAMPLES:
            raise ValueError("the voice hop must be a whole number of frames")
        self._questions: dict[str, _Question] = {}
        for qid in SIGNAL_IDS:
            self._questions[qid] = _Question(
                qid, cfg.voice_signals[qid], cfg.voice_signals[qid], None, None, cfg.defaults, True
            )
        self._vectors: Tensor | None = None
        self._ref: Tensor | None = None
        self._enroll_s: float | None = None
        self._events = EventLayer([], {}, cant_tell_below=cfg.cant_tell_below)
        self._answers: dict[str, Answer] = {}
        self._carry: list[Event] = []
        self.seconds: dict[str, float] = {}
        self.steps = 0
        self._sync_questions()
        self.reset()

    # ------------------------------------------------------------------ questions

    @property
    def questions(self) -> list[tuple[str, str]]:
        """`(id, text)` of every question, the two voice signals first."""
        return [(q.id, q.text) for q in self._questions.values()]

    def settings_of(self, question: str) -> Settings:
        """The event settings the question runs with."""
        return self._question(question).settings

    def ask(
        self,
        id: str,  # noqa: A002
        text: str,
        *,
        yes_means: str | None = None,
        no_means: str | None = None,
        threshold: float | None = None,
        start_secs: float | None = None,
        stop_threshold: float | None = None,
        stop_secs: float | None = None,
    ) -> None:
        """Add a question, or replace the one with this id. `yes_means` and `no_means`
        together spell out the criteria. Settings not given take the defaults. Replacing
        a question with new text stops its open event; the `stopped` event arrives with
        the next update."""
        if not isinstance(id, str) or not ID_PATTERN.match(id) or id in SIGNAL_IDS:
            raise ValueError(
                f"{id!r} is not a question id: a letter, then letters, digits, _ or -, and "
                f"not {SIGNAL_IDS[0]!r} or {SIGNAL_IDS[1]!r}, which are built in"
            )
        if id not in self._questions and len(self._questions) - len(SIGNAL_IDS) >= MAX_QUESTIONS:
            raise ValueError(f"at most {MAX_QUESTIONS} questions per session")
        embedded = question_text(text, yes_means, no_means)
        settings = self.model.config.defaults.override(
            f"settings of {id!r}", threshold=threshold, start_secs=start_secs,
            stop_threshold=stop_threshold, stop_secs=stop_secs,
        )  # fmt: skip
        self.model.encode([embedded])  # raises on a text the encoder refuses
        old = self._questions.get(id)
        if old is not None and old.embedded != embedded:
            self.remove(id)
        self._questions[id] = _Question(
            id, str(text).strip(), embedded, yes_means or None, no_means or None, settings, False
        )
        self._sync_questions()

    def remove(self, id: str) -> None:  # noqa: A002
        """Drop a question. Its open event stops; the `stopped` event arrives with the
        next update."""
        q = self._question(id)
        if q.is_signal:
            raise ValueError(f"{id!r} is built in and cannot be removed")
        del self._questions[id]
        self._sync_questions()

    def tune(self, id: str, **settings: float | None) -> Settings:  # noqa: A002
        """New event settings for a question or a voice signal, from the next frame on;
        its events keep running. Returns the settings it now runs with."""
        q = self._question(id)
        q.settings = q.settings.override(f"settings of {id!r}", **settings)
        self._events.retune({id: Params.from_settings(q.settings)})
        return q.settings

    def _question(self, qid: str) -> _Question:
        q = self._questions.get(qid)
        if q is None:
            raise KeyError(f"no question {qid!r}; the questions are {list(self._questions)}")
        return q

    def _sync_questions(self) -> None:
        """Encode the question list and align the event layer to it; events closed by
        a removal wait for the next update."""
        ids = list(self._questions)
        texts = [q.embedded for q in self._questions.values()]
        vectors = self.model.encode(texts)
        self._vectors = torch.from_numpy(np.ascontiguousarray(vectors)).to(self.device)[None]
        params = {q.id: Params.from_settings(q.settings) for q in self._questions.values()}
        closed = self._events.set_questions(ids, params)
        self._carry += [self._event(e) for e in closed]
        for qid in list(self._answers):
            if qid not in self._questions:
                del self._answers[qid]

    # ------------------------------------------------------------------ the voice

    def enroll(self, voice: Any | None) -> None:
        """The user's voice: a clip of their speech (16 kHz mono, a few seconds; at least
        25 ms), or None to forget it. The `user` signal answers from the next frame on."""
        if voice is None:
            self._ref, self._enroll_s = None, None
            return
        x = _as_audio(voice, "enroll")
        if x.shape[0] < MIN_SAMPLES:
            raise ValueError(
                f"the enrollment needs at least {MIN_SAMPLES} samples, got {x.shape[0]}"
            )
        if float(np.max(np.abs(x))) < 1e-4:
            raise ValueError("the enrollment is silent")
        wave = torch.from_numpy(x).to(self.model.voice.device)[None]
        self._ref = self.model.voice.embed(wave)[0].float().to(self.device)
        self._enroll_s = x.shape[0] / SAMPLE_RATE

    @property
    def enrolled(self) -> bool:
        return self._ref is not None

    @property
    def enrollment_s(self) -> float | None:
        """The enrollment clip's length in seconds, or None."""
        return self._enroll_s

    # ------------------------------------------------------------------ the stream

    def reset(self) -> None:
        """A new stream with the same questions and enrollment."""
        model = self.model
        self.seconds = dict.fromkeys(PARTS, 0.0)
        self.steps = 0
        self._stream = StaggeredStream(
            model.listener, chunk_frames=self.chunk_frames, layer=model.read_layer
        )
        self._pending = np.zeros(0, dtype=np.float32)
        self._n = 0
        self._audio = np.zeros(0, dtype=np.float32)
        self._audio_base = 0
        self._frames = torch.zeros(0, model.listener.embed_dim, device=self.device)
        self._frames_base = 0
        self._t = 0
        self._spk = torch.zeros(0, self.spk_dim, device=self.device)
        self._spk_base = 0
        self._windows = 0
        self._finished = False
        self._answers = {}
        self._carry = []
        ids = list(self._questions)
        params = {q.id: Params.from_settings(q.settings) for q in self._questions.values()}
        self._events = EventLayer(ids, params, cant_tell_below=model.config.cant_tell_below)

    @property
    def samples_consumed(self) -> int:
        """Samples turned into frames so far (the buffered remainder not counted)."""
        return self._n

    @property
    def frames_emitted(self) -> int:
        return self._t

    @property
    def time_s(self) -> float:
        """The stream time of the newest answer."""
        return self._t / FRAME_HZ

    @property
    def finished(self) -> bool:
        return self._finished

    def push(self, audio: Any) -> list[Update]:
        """Add audio of any length; run every whole 80 ms step it completes."""
        if self._finished:
            raise RuntimeError("push after finish: call reset() first")
        x = _as_audio(audio, "push")
        self._pending = np.concatenate([self._pending, x]) if self._pending.size else x
        out: list[Update] = []
        while self._pending.shape[0] >= self.step_samples:
            chunk, self._pending = (
                self._pending[: self.step_samples],
                self._pending[self.step_samples :],
            )
            out += self._step(chunk, flush=False)
        return out

    def finish(self) -> list[Update]:
        """Run the buffered remainder and the stream's last frames."""
        if self._finished:
            return []
        chunk, self._pending = self._pending, np.zeros(0, dtype=np.float32)
        out = self._step(chunk, flush=True)
        self._finished = True
        return out

    @torch.no_grad()
    def _step(self, chunk: np.ndarray, *, flush: bool) -> list[Update]:
        assert self._vectors is not None
        self.steps += 1
        self._n += int(chunk.shape[0])
        self._audio = np.concatenate([self._audio, chunk]) if self._audio.size else chunk
        t0 = time.perf_counter()
        wave = torch.from_numpy(np.ascontiguousarray(chunk)).to(self.device)
        new = self._stream.push(wave) if chunk.shape[0] else None
        if flush:
            tail = self._stream.flush()
            new = tail if new is None else torch.cat([new, tail])
        count = 0 if new is None else int(new.shape[0])
        if count:
            self._frames = torch.cat([self._frames, new.float().to(self.device)])
        t1 = time.perf_counter()
        self.seconds["listener"] += t1 - t0
        self._embed_windows()
        t2 = time.perf_counter()
        self.seconds["voice"] += t2 - t1
        if not count:
            return []
        frames = list(range(self._t, self._t + count))
        p, pc = self._forward(frames)
        self.seconds["head"] += time.perf_counter() - t2
        out: list[Update] = []
        ids = list(self._questions)
        for row, t in enumerate(frames):
            frame_events = self._events.update(t, p[row], pc[row])
            events = [self._event(e) for e in frame_events]
            state = self._events.state()
            answers = {}
            for q, (qid, (answer, since)) in enumerate(zip(ids, state, strict=True)):
                answers[qid] = Answer(
                    qid, answer, float(p[row, q]), float(pc[row, q]),
                    None if since is None else since / FRAME_HZ,
                )  # fmt: skip
            self._answers = answers
            if self._carry:
                events = [*self._carry, *events]
                self._carry = []
            out.append(Update((t + 1) / FRAME_HZ, t, answers, events))
        self._t += count
        keep = self.n_frames - 1
        if self._frames.shape[0] > keep:
            drop = self._frames.shape[0] - keep
            self._frames = self._frames[drop:]
            self._frames_base += drop
        self._trim_voices()
        return out

    def _window_ends(self, first: int, count: int) -> Tensor:
        """The frame from which each voice window `first .. first + count - 1` is
        visible: the first frame whose input holds the window's last sample."""
        i = torch.arange(first, first + count, device=self.device)
        return -(-(self.spk_window + i * self.spk_hop - FRAME_READY_SAMPLES) // FRAME_SAMPLES)

    def _trim_voices(self) -> None:
        """Drop the voice vectors no future frame can read: every frame from now on sees
        at least the windows already visible, and only the newest `n_spk` of those."""
        m = self._spk.shape[0]
        if m <= self.n_spk:
            return
        visible = int((self._window_ends(self._spk_base, m) <= self._t).sum())
        drop = visible - self.n_spk
        if drop > 0:
            self._spk = self._spk[drop:]
            self._spk_base += drop

    def _embed_windows(self) -> None:
        ready: list[np.ndarray] = []
        while self._windows * self.spk_hop + self.spk_window <= self._n:
            start = self._windows * self.spk_hop - self._audio_base
            ready.append(self._audio[start : start + self.spk_window])
            self._windows += 1
        if ready:
            voice = self.model.voice
            batch = torch.from_numpy(np.stack(ready)).to(voice.device)
            self._spk = torch.cat([self._spk, voice.embed(batch).float().to(self.device)])
        next_start = self._windows * self.spk_hop
        drop = next_start - self._audio_base
        if drop > 0:
            self._audio = self._audio[drop:]
            self._audio_base = next_start

    def _forward(self, frames: Sequence[int]) -> tuple[np.ndarray, np.ndarray]:
        """`(p [B, Q], p_can_tell [B, Q])` for the given frames, in one head forward."""
        assert self._vectors is not None
        t = torch.tensor(frames, dtype=torch.int64, device=self.device)
        B = t.shape[0]
        local = t - self._frames_base
        offsets = torch.arange(self.n_frames, device=self.device) - (self.n_frames - 1)
        index = local[:, None] + offsets[None, :]
        frame_mask = (t[:, None] + offsets[None, :]) >= 0
        frames_in = self._frames[index.clamp(min=0)] * frame_mask[:, :, None].to(self._frames.dtype)
        spk, spk_age, spk_mask = self._voice_slots(t)
        if self._ref is None:
            ref, ref_mask = None, None
        else:
            ref = self._ref[None].expand(B, -1)
            ref_mask = torch.ones(B, dtype=torch.bool, device=self.device)
        side = audio_side(
            frames_in, n_frames=self.n_frames, n_spk=self.n_spk, spk_dim=self.spk_dim,
            frame_mask=frame_mask, spk=spk, spk_age=spk_age, spk_mask=spk_mask,
            ref=ref, ref_mask=ref_mask,
        )  # fmt: skip
        memory, mask = self.model.head.encode_audio(side)
        p, pc = self.model.head.answer(memory, mask, self._vectors.expand(B, -1, -1))
        p = np.clip(p.float().cpu().numpy().astype(np.float64), 0.0, 1.0)
        pc = np.clip(pc.float().cpu().numpy().astype(np.float64), 0.0, 1.0)
        return p, pc

    def _voice_slots(self, t: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """The up to `n_spk` newest voice windows that have ended by each frame, newest
        last, left-padded and masked out; a window of samples `[a, b)` reaches the head
        from the first frame whose input holds `b`."""
        n, m = t.shape[0], self._spk.shape[0]
        spk = torch.zeros(n, self.n_spk, self.spk_dim, device=self.device)
        age = torch.zeros(n, self.n_spk, dtype=torch.int64, device=self.device)
        mask = torch.zeros(n, self.n_spk, dtype=torch.bool, device=self.device)
        if m == 0:
            return spk, age, mask
        ends = self._window_ends(self._spk_base, m)
        for k in range(n):
            ready = int((ends <= t[k]).sum())
            if ready == 0:
                continue
            take = min(ready, self.n_spk)
            lo = ready - take
            spk[k, self.n_spk - take :] = self._spk[lo:ready]
            age[k, self.n_spk - take :] = t[k] - ends[lo:ready]
            mask[k, self.n_spk - take :] = True
        return spk, age, mask

    # ------------------------------------------------------------------ reading

    @property
    def answers(self) -> dict[str, Answer]:
        """The newest answer per question id (empty before the first frame)."""
        return dict(self._answers)

    @property
    def events(self) -> list[Event]:
        """Every event of the stream so far, open ones as `started`."""
        return [self._event(e) for e in self._events.events_so_far()]

    def frame_events(self) -> list[FrameEvent]:
        """The events in frame indices, for tests and tools."""
        return self._events.events_so_far()

    @property
    def timing(self) -> dict[str, float]:
        """Mean milliseconds of compute per 80 ms step, by part and in total (nan before
        the first step)."""
        n = self.steps
        out = {p: (self.seconds[p] * 1000 / n if n else float("nan")) for p in PARTS}
        out["total"] = sum(self.seconds.values()) * 1000 / n if n else float("nan")
        return out

    @staticmethod
    def _event(e: FrameEvent) -> Event:
        if e.end is None:
            return Event(e.question, "started", e.start / FRAME_HZ, e.confirm / FRAME_HZ)
        return Event(
            e.question, "stopped", e.start / FRAME_HZ, e.confirm / FRAME_HZ,
            end_s=e.end / FRAME_HZ, duration_s=(e.end - e.start) / FRAME_HZ,
        )  # fmt: skip
