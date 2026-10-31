"""The event layer against the reference's cases, and the settings' rules."""

from __future__ import annotations

import numpy as np
import pytest

from crele_nimbus.config import FRAME_HZ, Settings, min_frames
from crele_nimbus.events import EventLayer, FrameEvent, Params

DEFAULTS = Settings(threshold=0.7, start_secs=0.12, stop_threshold=0.35, stop_secs=0.3)


def layer_for(settings: Settings, ids=("q",)) -> EventLayer:
    p = Params.from_settings(settings)
    return EventLayer(ids, dict.fromkeys(ids, p), cant_tell_below=0.5)


def rows(events: list[FrameEvent], seq: int) -> list[tuple[int, ...]]:
    return [
        (seq, e.start, -1 if e.end is None else e.end, e.detect, e.confirm,
         -1 if e.end_confirm is None else e.end_confirm)
        for e in events
    ]  # fmt: skip


@pytest.mark.parametrize("case", [0, 1, 2])
def test_reference_cases_exactly(reference, case):
    """24 seeded probability sequences of 400 frames, three settings: the events must equal
    the stored ones row for row."""
    p, pc = reference.data["case_p"], reference.data["case_p_can_tell"]
    t, s, st, ss = reference.data[f"case{case}_settings"].tolist()
    settings = Settings(threshold=t, start_secs=s, stop_threshold=st, stop_secs=ss)
    got = []
    for j in range(p.shape[0]):
        layer = layer_for(settings)
        for k in range(p.shape[1]):
            layer.update(k, np.array([p[j, k]]), np.array([pc[j, k]]))
        got += rows(layer.events_so_far(), j)
    expected = reference.data[f"case{case}_events"]
    assert np.array_equal(np.array(got, dtype=np.int64).reshape(-1, 6), expected)
    assert len(expected) > 0


def test_min_frames():
    assert min_frames(0.3, FRAME_HZ) == 8
    assert min_frames(0.12, FRAME_HZ) == 3
    assert min_frames(0.0, FRAME_HZ) == 1
    assert min_frames(0.04, FRAME_HZ) == 1
    with pytest.raises(ValueError):
        min_frames(-0.1, FRAME_HZ)


def test_settings_rules():
    s = DEFAULTS.override(threshold=0.5)
    assert s.stop_threshold == 0.35  # threshold never moves stop_threshold
    with pytest.raises(ValueError, match="stop_threshold 0.7 must be under threshold 0.7"):
        DEFAULTS.override(stop_threshold=0.7)
    with pytest.raises(ValueError, match="unknown settings"):
        DEFAULTS.override(min_off_s=1.0)
    with pytest.raises(ValueError, match="not a finite number"):
        DEFAULTS.override(threshold=float("nan"))
    with pytest.raises(ValueError, match="not a number"):
        DEFAULTS.override(threshold=True)
    assert DEFAULTS.override(start_secs=None) == DEFAULTS
    assert Settings.from_mapping(DEFAULTS.as_dict()) == DEFAULTS


def test_held_frames_cancel_and_hold_state():
    layer = layer_for(DEFAULTS.override(start_secs=0.08, stop_secs=0.08))  # 2 frames each
    out = []
    for t, (p, c) in enumerate([(0.9, 0.9), (0.9, 0.2), (0.9, 0.9), (0.9, 0.9)]):
        out += layer.update(t, np.array([p]), np.array([c]))
    # The held frame 1 cancelled the candidate of frame 0; the next one started at 2.
    assert [(e.detect, e.confirm, e.start) for e in out] == [(2, 3, 2)]
    assert layer.state()[0][0] == "yes"
    layer.update(4, np.array([0.9]), np.array([0.1]))
    assert layer.state()[0][0] == "cant_tell"
    layer.update(5, np.array([0.9]), np.array([0.9]))
    assert layer.state()[0][0] == "yes"


def test_retune_keeps_state_and_pending_counts():
    layer = layer_for(DEFAULTS.override(start_secs=0.4))  # 10 frames to start
    for t in range(5):
        assert layer.update(t, np.array([0.9]), np.array([0.9])) == []
    layer.retune({"q": Params.from_settings(DEFAULTS.override(start_secs=0.2))})  # 5 frames
    out = layer.update(5, np.array([0.9]), np.array([0.9]))
    assert len(out) == 1 and out[0].detect == 0 and out[0].confirm == 5
    with pytest.raises(KeyError):
        layer.retune({"zzz": Params.from_settings(DEFAULTS)})


def test_set_questions_keeps_staying_questions_and_closes_removed():
    p = Params.from_settings(DEFAULTS.override(start_secs=0.04))  # one frame to start
    layer = EventLayer(["a", "b"], {"a": p, "b": p}, cant_tell_below=0.5)
    started = []
    for t in range(3):
        started += layer.update(t, np.array([0.9, 0.9]), np.array([0.9, 0.9]))
    assert sorted(e.question for e in started) == ["a", "b"]
    closed = layer.set_questions(["b", "c"], {"b": p, "c": p})
    assert [(e.question, e.end, e.end_confirm) for e in closed] == [("a", 3, 3)]
    assert layer.ids == ["b", "c"]
    assert layer.state()[0][0] == "yes"  # b kept its run
    assert layer.state()[1] == ("no", None)  # c is new
    out = layer.update(3, np.array([0.9, 0.9]), np.array([0.9, 0.9]))
    assert [e.question for e in out] == ["c"]
    events = layer.events_so_far()
    assert [(e.question, e.end) for e in events] == [("a", 3), ("b", None), ("c", None)]
    with pytest.raises(ValueError, match="unique"):
        layer.set_questions(["b", "b"], {"b": p})
    with pytest.raises(ValueError, match="no settings"):
        layer.set_questions(["b", "d"], {"b": p})


def test_frames_must_be_in_order():
    layer = layer_for(DEFAULTS)
    layer.update(0, np.array([0.1]), np.array([0.9]))
    with pytest.raises(ValueError, match="expected frame 1"):
        layer.update(2, np.array([0.1]), np.array([0.9]))
    with pytest.raises(ValueError, match="finite"):
        layer.update(1, np.array([float("nan")]), np.array([0.9]))
