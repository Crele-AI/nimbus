"""The playground's engine on tiny models: the protocol, without the network."""

from __future__ import annotations

import base64

import numpy as np
import pytest

from conftest import synth_audio, tiny_model
from crele_nimbus.playground.engine import MAX_TYPED, Engine, PlaygroundError, measure_step


def engine() -> Engine:
    model = tiny_model(0)
    return Engine(model.session(), model_name="tiny", step_ms={"total": 1.0})


def kinds(msgs):
    return [m["type"] for m in msgs]


def test_ready_questions_and_settings():
    e = engine()
    r = e.ready()
    assert r["type"] == "ready" and r["state"] == "idle" and r["max_typed"] == MAX_TYPED
    assert [q["id"] for q in r["questions"]] == ["user", "other"]
    out = e.control(
        {
            "type": "questions",
            "questions": [
                {"id": "a", "text": "Is music playing?", "threshold": 0.5},
                {"id": "b", "text": "Is it loud? || louder than speech || quiet"},
            ],
            "signals": {"user": {"stop_secs": 1.0}},
        }
    )
    assert kinds(out) == ["ready"]
    qs = {q["id"]: q for q in out[0]["questions"]}
    assert list(qs) == ["user", "other", "a", "b"]
    assert qs["a"]["threshold"] == 0.5 and qs["a"]["stop_threshold"] == 0.35
    assert qs["a"]["custom"] == {"threshold": 0.5}
    assert qs["b"]["yes_means"] == "louder than speech" and qs["b"]["text"] == "Is it loud?"
    assert qs["user"]["stop_secs"] == 1.0 and qs["other"]["stop_secs"] == 0.3
    # Dropping a setting goes back to the default.
    out = e.control({"type": "questions", "questions": [{"id": "a", "text": "Is music playing?"}]})
    qs = {q["id"]: q for q in out[0]["questions"]}
    assert list(qs) == ["user", "other", "a"] and qs["a"]["threshold"] == 0.7
    assert qs["user"]["stop_secs"] == 0.3


def test_refusals_change_nothing():
    e = engine()
    e.control({"type": "questions", "questions": [{"id": "a", "text": "Is music playing?"}]})
    bad = [
        {"questions": [{"id": "user", "text": "x"}]},
        {"questions": [{"id": "a", "text": "x"}, {"id": "a", "text": "y"}]},
        {"questions": [{"id": "a", "text": "x || only one"}]},
        {"questions": [{"id": "a", "text": "x", "threshold": 2.0}]},
        {"questions": [{"id": "a", "text": "x", "threshold": 0.3, "stop_threshold": 0.3}]},
        {"questions": [{"id": f"q{i}", "text": "x"} for i in range(MAX_TYPED + 1)]},
        {"questions": [{"id": "a", "text": ""}]},
        {"signals": {"nobody": {"threshold": 0.5}}},
    ]
    for msg in bad:
        out = e.control({"type": "questions", **msg})
        assert kinds(out) == ["error"], msg
    assert [q["id"] for q in e.ready()["questions"]] == ["user", "other", "a"]
    assert kinds(e.control("not json")) == ["error"]
    assert kinds(e.control({"type": "dance"})) == ["error"]
    assert kinds(e.control({"type": "stop"})) == ["error"]
    with pytest.raises(PlaygroundError, match="before start"):
        e.audio(np.zeros(1280, dtype=np.float32))
    assert e.audio(np.zeros(1280, dtype=np.float32)) == []  # warned once


def test_a_stream_with_events_and_a_summary():
    e = engine()
    e.control(
        {
            "type": "questions",
            "questions": [
                {"id": "a", "text": "Is music playing?", "threshold": 0.15,
                 "stop_threshold": 0.1, "start_secs": 0.04, "stop_secs": 0.04}
            ],
        }
    )  # fmt: skip
    out = e.control({"type": "start", "source": "mic"})
    assert kinds(out) == ["ready"] and out[0]["state"] == "listening"
    wave = synth_audio(1, 3.0)
    msgs = []
    for s in range(0, wave.shape[0], 1280):
        msgs += e.audio(wave[s : s + 1280].tobytes(), lag_s=0.0)
    decisions = [m for m in msgs if m["type"] == "decisions"]
    assert decisions and all(len(d["steps"]) == 2 for d in decisions)
    assert set(decisions[0]["steps"][0]["answers"][0]) == {"id", "p", "p_can_tell", "state"}
    assert decisions[0]["steps"][0]["answers"][0]["state"] in ("yes", "no", "cant_tell")
    events = [m for m in msgs if m["type"] == "event"]
    assert events and {m["event"] for m in events} <= {"started", "stopped"}
    assert all("start_s" in m and "confirmed_s" in m for m in events)
    # Replacing the question while listening: its open event stops with the next answers.
    out = e.control({"type": "questions", "questions": [{"id": "b", "text": "Is a dog barking?"}]})
    assert kinds(out) == ["ready"]
    more = e.audio(wave[:1280].tobytes())
    stopped = [m for m in more if m["type"] == "event" and m["question"] == "a"]
    assert all(m["event"] == "stopped" and m["text"] == "Is music playing?" for m in stopped)
    out = e.control({"type": "stop", "reason": "done"})
    assert kinds(out)[-2:] == ["summary", "ready"]
    summary = out[-2]
    assert summary["stopped_by"] == "done" and summary["audio_s"] > 2.9
    assert {q["id"] for q in summary["questions"]} == {"user", "other", "b"}
    assert out[-1]["state"] == "idle"


def test_enroll_and_file_streams():
    e = engine()
    voice = synth_audio(2, 2.0)
    out = e.control({"type": "enroll", "pcm": base64.b64encode(voice.tobytes()).decode()})
    assert kinds(out) == ["ready"] and out[0]["enrolled"] and out[0]["enroll_s"] == 2.0
    assert kinds(e.control({"type": "enroll", "pcm": "###"})) == ["error"]
    short = base64.b64encode(np.zeros(100, dtype=np.float32).tobytes()).decode()
    assert kinds(e.control({"type": "enroll", "pcm": short})) == ["error"]
    out = e.control({"type": "enroll", "pcm": None})
    assert not out[0]["enrolled"]
    out = e.control({"type": "wav", "name": "clip.wav", "duration_s": 1.0})
    assert out[0]["source"] == "wav" and out[0]["name"] == "clip.wav"
    e.audio(synth_audio(3, 1.0).tobytes())
    out = e.control({"type": "start", "source": "mic"})  # a new stream ends the old one
    assert kinds(out) == ["summary", "ready"] or kinds(out)[-2:] == ["summary", "ready"]
    e.close()
    assert e.state == "idle"


def test_measure_step():
    model = tiny_model(0)
    s = model.session()
    s.ask("q", "Is music playing?")
    ms = measure_step(s, warmup=1, steps=3)
    assert set(ms) == {"listener", "voice", "head", "total"} and ms["total"] > 0
    assert s.steps == 0 and not s.finished
