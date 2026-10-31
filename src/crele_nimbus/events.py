"""The event layer: a probability per frame in, start and stop events out.

Every 40 ms the model gives each question a yes probability `p` and a can-tell
probability `p_can_tell`. This module turns those into events with hysteresis and
minimum durations, one state machine per question, fed one frame at a time.

**The rules, per question.** A frame is confident when `p_can_tell >= cant_tell_below`;
otherwise it is held: the state stays as it was and any pending change is cancelled, so
no change ever rests on frames the model said it could not judge.

- Off. A confident frame with `p >= on` starts a start-candidate; it lasts while frames
  stay confident with `p > off` and is confirmed after `n_on` frames: the state turns on
  and a `started` event is reported. A held frame or a confident `p <= off` cancels it.
- On. A confident frame with `p <= off` starts a stop-candidate; it lasts while frames
  stay confident with `p < on` and is confirmed after `n_off` frames: the state turns off
  and the event is reported again, `stopped`, with its end. A held frame or a confident
  `p >= on` cancels it.

`on` is the question's `threshold`, `off` its `stop_threshold`, `n_on` and `n_off` the
frames of `start_secs` and `stop_secs` (`config.min_frames`).

**What an event carries**, in frame indices. `detect` is the frame the start-candidate
began; `confirm` the frame it was confirmed, which is when the event is reported; `start`
the estimated onset, `detect` moved back over the confident frames with `p > off` just
before it (never before the first frame fed, a held frame, a frame with `p <= off`, or the
previous event's end); `end` the frame the stop-candidate began (exclusive), None while
on; `end_confirm` the frame the end was confirmed and reported.

**While a stream runs**, `retune` changes a question's settings from the next frame on
without touching its state, and `set_questions` adds and removes questions: a question
that stays keeps its state and its pending candidates, a removed question's open event is
closed at the current frame, a new one starts off. Frames must be fed in order, every
frame, so the durations count.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace

import numpy as np

from crele_nimbus.config import FRAME_HZ, Settings, min_frames


@dataclass(frozen=True)
class FrameEvent:
    """One run of a question's state, in frame indices (module docstring)."""

    question: str
    start: int
    end: int | None
    detect: int
    confirm: int
    end_confirm: int | None = None


@dataclass(frozen=True)
class Params:
    """A question's settings in the layer's own units."""

    on: float
    off: float
    n_on: int
    n_off: int

    @classmethod
    def from_settings(cls, s: Settings, frame_hz: int = FRAME_HZ) -> Params:
        return cls(
            on=float(s.threshold),
            off=float(s.stop_threshold),
            n_on=min_frames(s.start_secs, frame_hz),
            n_off=min_frames(s.stop_secs, frame_hz),
        )


@dataclass
class _Candidate:
    first: int
    count: int = 1


class _Track:
    """One question's state machine."""

    def __init__(self, qid: str) -> None:
        self.qid = qid
        self.first_t: int | None = None
        self.run: FrameEvent | None = None  # the open event while on
        self.last_end: int | None = None
        self.last_stop = -1  # the newest frame that was held or had p <= off
        self.held = False
        self.cand_on: _Candidate | None = None
        self.cand_off: _Candidate | None = None

    def cancel(self) -> None:
        self.cand_on = self.cand_off = None

    def step(self, t: int, x: float, confident: bool, p: Params) -> list[FrameEvent]:
        if self.first_t is None:
            self.first_t = t
            self.last_stop = t - 1
        self.held = not confident
        if not confident:
            self.cancel()
            self.last_stop = t
            return []
        if x <= p.off:
            self.last_stop = t
        return self._step_off(t, x, p) if self.run is None else self._step_on(t, x, p)

    def _step_off(self, t: int, x: float, p: Params) -> list[FrameEvent]:
        c = self.cand_on
        if c is not None:
            if x > p.off:
                c.count += 1
            else:
                self.cand_on = None
        if self.cand_on is None and x >= p.on:
            self.cand_on = _Candidate(t)
        c = self.cand_on
        if c is None or c.count < p.n_on:
            return []
        start = self.last_stop + 1
        if self.last_end is not None:
            start = max(start, self.last_end)
        event = FrameEvent(self.qid, start, None, c.first, t)
        self.run = event
        self.cancel()
        return [event]

    def _step_on(self, t: int, x: float, p: Params) -> list[FrameEvent]:
        assert self.run is not None
        c = self.cand_off
        if c is not None:
            if x < p.on:
                c.count += 1
            else:
                self.cand_off = None
        if self.cand_off is None and x <= p.off:
            self.cand_off = _Candidate(t)
        c = self.cand_off
        if c is None or c.count < p.n_off:
            return []
        ended = replace(self.run, end=c.first, end_confirm=t)
        self.run = None
        self.last_end = c.first
        self.cancel()
        return [ended]

    def close(self, t: int) -> FrameEvent | None:
        """End an open run at frame `t` (exclusive), for a question being removed."""
        if self.run is None:
            return None
        ended = replace(self.run, end=max(t, self.run.start + 1), end_confirm=t)
        self.run = None
        self.last_end = ended.end
        self.cancel()
        return ended


class EventLayer:
    """The event machines of a list of questions (module docstring).

    `params` maps each question id to its `Params`; `cant_tell_below` is the can-tell
    level under which a frame is held. Questions are answered in the order of `ids`.
    """

    def __init__(
        self, ids: Sequence[str], params: Mapping[str, Params], *, cant_tell_below: float
    ) -> None:
        self.cant_tell_below = float(cant_tell_below)
        self._tracks: list[_Track] = []
        self._params: list[Params] = []
        self._events: list[FrameEvent] = []
        self._t: int | None = None
        self.set_questions(ids, params)

    # Configuration.

    @property
    def ids(self) -> list[str]:
        return [tr.qid for tr in self._tracks]

    @property
    def last_frame(self) -> int | None:
        """The last frame fed, or None before the first."""
        return self._t

    def params_of(self, qid: str) -> Params:
        return self._params[self.ids.index(qid)]

    def retune(self, params: Mapping[str, Params]) -> None:
        """New settings for the ids named, from the next frame on; the state, the pending
        candidates and the events so far are kept."""
        ids = self.ids
        for qid in params:
            if qid not in ids:
                raise KeyError(qid)
        self._params = [
            params.get(tr.qid, p) for tr, p in zip(self._tracks, self._params, strict=True)
        ]

    def set_questions(self, ids: Sequence[str], params: Mapping[str, Params]) -> list[FrameEvent]:
        """The new list of questions. A question that stays keeps its state; a removed
        one has its open event closed at the current frame and returned; a new one starts
        off. Ids must be unique and each must have its `Params`."""
        ids = list(ids)
        if len(set(ids)) != len(ids):
            raise ValueError("question ids must be unique")
        missing = [q for q in ids if q not in params]
        if missing:
            raise ValueError(f"no settings for {missing}")
        old = {tr.qid: tr for tr in self._tracks}
        closed: list[FrameEvent] = []
        if self._t is not None:
            for qid, tr in old.items():
                if qid not in ids:
                    e = tr.close(self._t + 1)
                    if e is not None:
                        self._record(e)
                        closed.append(e)
        self._tracks = [old.get(q) or _Track(q) for q in ids]
        self._params = [params[q] for q in ids]
        return closed

    # Streaming.

    def update(self, t: int, p: np.ndarray, p_can_tell: np.ndarray) -> list[FrameEvent]:
        """Feed frame `t`: `p` and `p_can_tell` are `[Q]` in question order. Returns the
        events started or stopped at this frame, in question order."""
        t = int(t)
        if self._t is not None and t != self._t + 1:
            raise ValueError(f"expected frame {self._t + 1}, got {t}: feed every frame in order")
        q = len(self._tracks)
        x = np.asarray(p, dtype=np.float64).reshape(-1)
        c = np.asarray(p_can_tell, dtype=np.float64).reshape(-1)
        if x.shape != (q,) or c.shape != (q,):
            raise ValueError(f"p and p_can_tell must each hold {q} values")
        if not (np.all(np.isfinite(x)) and np.all(np.isfinite(c))):
            raise ValueError("probabilities must be finite")
        self._t = t
        out: list[FrameEvent] = []
        for track, xi, ci, par in zip(self._tracks, x, c, self._params, strict=True):
            for e in track.step(t, float(xi), bool(ci >= self.cant_tell_below), par):
                self._record(e)
                out.append(e)
        return out

    def _record(self, e: FrameEvent) -> None:
        if e.end is None:
            self._events.append(e)
            return
        for i in range(len(self._events) - 1, -1, -1):
            o = self._events[i]
            if o.question == e.question and o.end is None:
                self._events[i] = e
                return
        raise AssertionError(f"an end of {e.question!r} with no open event")

    # Reading.

    def state(self) -> list[tuple[str, int | None]]:
        """Per question in order: the answer, `yes`, `no` or `cant_tell`, and the frame
        the current state began (the open event's start when on, the last end or the
        first frame when off; None before any frame)."""
        out = []
        for tr in self._tracks:
            if tr.held:
                answer = "cant_tell"
            elif tr.run is not None:
                answer = "yes"
            else:
                answer = "no"
            if tr.run is not None:
                since = tr.run.start
            elif tr.last_end is not None:
                since = tr.last_end
            else:
                since = tr.first_t
            out.append((answer, since))
        return out

    def events_so_far(self) -> list[FrameEvent]:
        """Every event of the stream in the order started; open ones have `end=None`."""
        return list(self._events)
