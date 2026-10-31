"""The session on tiny random models: any chunking gives the same answers; the question
list and the enrollment may change while the stream runs."""

from __future__ import annotations

import numpy as np
import pytest

from conftest import synth_audio, tiny_model
from crele_nimbus import Settings, Update
from crele_nimbus.listener import n_frames


def probabilities(updates: list[Update]) -> dict[str, np.ndarray]:
    ids = list(updates[0].answers)
    return {q: np.array([u.answers[q].probability for u in updates]) for q in ids}


def run(session, wave: np.ndarray, chunk: int) -> list[Update]:
    out = []
    for s in range(0, wave.shape[0], chunk):
        out += session.push(wave[s : s + chunk])
    return out + session.finish()


@pytest.mark.parametrize("chunk", [1280, 4096, 999, 16000 * 6])
def test_any_chunking_equals_one_push(model, chunk):
    wave = synth_audio(1, 5.0)
    a = model.session()
    a.ask("q1", "Is music playing?")
    a.ask("q2", "Is it loud?", yes_means="very loud", no_means="quiet", threshold=0.6)
    whole = a.push(wave) + a.finish()
    b = model.session()
    b.ask("q1", "Is music playing?")
    b.ask("q2", "Is it loud?", yes_means="very loud", no_means="quiet", threshold=0.6)
    parts = run(b, wave, chunk)
    assert len(whole) == len(parts) == n_frames(wave.shape[0])
    assert [u.frame for u in whole] == list(range(len(whole)))
    assert whole[0].time_s == 0.04 and whole[-1].time_s == len(whole) / 25
    assert list(whole[0].answers) == ["user", "other", "q1", "q2"]
    pa, pb = probabilities(whole), probabilities(parts)
    for q in pa:
        assert np.array_equal(pa[q], pb[q]), q
    assert a.frame_events() == b.frame_events()
    assert a.answers == b.answers and a.events == b.events


def test_int16_input_and_input_checks(model):
    wave = synth_audio(2, 1.0)
    s = model.session()
    s.ask("q", "Is someone whistling?")
    a = s.push(wave) + s.finish()
    s.reset()
    i16 = np.round(wave * 32768).clip(-32768, 32767).astype(np.int16)
    b = s.push(i16) + s.finish()
    assert np.allclose(probabilities(a)["q"], probabilities(b)["q"], atol=1e-3)
    with pytest.raises(RuntimeError, match="finish"):
        s.push(wave)
    s.reset()
    with pytest.raises(ValueError, match="mono"):
        s.push(np.zeros((100, 2), dtype=np.float32))
    with pytest.raises(ValueError, match="finite"):
        s.push(np.array([np.nan], dtype=np.float32))
    assert s.push(np.zeros(0, dtype=np.float32)) == []


def test_enrollment_changes_the_user_answers_only_after_it_is_set(model):
    wave = synth_audio(3, 3.0)
    voice = synth_audio(4, 2.0)
    a = model.session()
    a.ask("q", "Is music playing?")
    without = a.push(wave) + a.finish()
    assert not a.enrolled
    b = model.session()
    b.ask("q", "Is music playing?")
    b.enroll(voice)
    assert b.enrolled and b.enrollment_s == 2.0
    with_voice = b.push(wave) + b.finish()
    pw, pv = probabilities(without), probabilities(with_voice)
    assert not np.allclose(pw["user"], pv["user"])
    b.enroll(None)
    assert not b.enrolled
    with pytest.raises(ValueError, match="at least"):
        b.enroll(np.zeros(100, dtype=np.float32))
    with pytest.raises(ValueError, match="silent"):
        b.enroll(np.zeros(16000, dtype=np.float32))


def test_asking_mid_stream_keeps_the_other_answers_and_events(model):
    wave = synth_audio(5, 4.0)
    a = model.session()
    a.ask("q1", "Is a dog barking?")
    a.ask("q2", "Is music playing?")
    both = a.push(wave) + a.finish()
    b = model.session()
    b.ask("q1", "Is a dog barking?")
    first = b.push(wave[: 16000 * 2])
    assert "q2" not in first[-1].answers
    b.ask("q2", "Is music playing?")
    rest = b.push(wave[16000 * 2 :]) + b.finish()
    pa, p1 = probabilities(both), probabilities(first)
    # Another question in the batch changes nothing but float32 reduction order.
    assert np.allclose(pa["q1"][: len(first)], p1["q1"], atol=1e-6)
    assert np.allclose(pa["q1"][len(first) :], probabilities(rest)["q1"], atol=1e-6)
    assert np.allclose(pa["q2"][len(first) :], probabilities(rest)["q2"], atol=1e-6)
    assert "q2" in rest[0].answers
    assert [e for e in a.frame_events() if e.question == "q1"] == [
        e for e in b.frame_events() if e.question == "q1"
    ]


def test_remove_tune_and_replace(model):
    wave = synth_audio(6, 3.0)
    s = model.session()
    s.ask("q1", "Is music playing?", threshold=0.2, stop_threshold=0.1, start_secs=0.04)
    s.push(wave)
    assert s.settings_of("q1") == Settings(0.2, 0.04, 0.1, 0.3)
    assert s.tune("q1", stop_secs=1.0).stop_secs == 1.0
    with pytest.raises(ValueError, match="stop_threshold 0.5 must be under threshold 0.2"):
        s.tune("q1", stop_threshold=0.5)
    s.remove("q1")
    assert s.questions == [("user", "Is the user speaking?"),
                           ("other", "Is someone other than the user speaking?")]  # fmt: skip
    with pytest.raises(KeyError):
        s.tune("q1", threshold=0.5)
    with pytest.raises(ValueError, match="built in"):
        s.remove("user")
    s.ask("q1", "Is music playing?", threshold=0.2, stop_threshold=0.1, start_secs=0.04)
    s.push(wave)
    s.ask("q1", "Is a siren sounding?")  # a new text under the same id starts it over
    assert s.settings_of("q1") == model.config.defaults
    s.tune("other", threshold=0.9)
    assert s.settings_of("other").threshold == 0.9
    s.finish()


def test_ids_and_limits(model):
    s = model.session()
    for bad in ("user", "other", "1abc", "", "a b", None):
        with pytest.raises(ValueError):
            s.ask(bad, "Is music playing?")
    with pytest.raises(ValueError, match="both sides"):
        s.ask("q", "Is it loud?", yes_means="loud")
    with pytest.raises(ValueError, match="text"):
        s.ask("q", "   ")
    for i in range(32):
        s.ask(f"q{i}", f"Question {i}?")
    with pytest.raises(ValueError, match="at most 32"):
        s.ask("one_more", "Another?")
    s.ask("q3", "Question 3?")  # replacing is fine at the limit
    assert len(s.questions) == 34


def test_reset_and_timing(model):
    wave = synth_audio(7, 2.0)
    s = model.session()
    s.ask("q", "Is music playing?")
    a = s.push(wave) + s.finish()
    assert (
        s.finished
        and s.timing["total"] > 0
        and set(s.timing) == {"listener", "voice", "head", "total"}
    )
    s.reset()
    assert not s.finished and s.answers == {} and s.events == [] and s.time_s == 0.0
    b = s.push(wave) + s.finish()
    assert np.array_equal(probabilities(a)["q"], probabilities(b)["q"])


def test_the_text_encoder_is_asked_once_per_text():
    model = tiny_model(0)
    fake = model._encoder
    s = model.session()
    s.ask("q1", "Is music playing?")
    s.ask("q2", "Is music playing?")
    t = model.session()
    t.ask("q1", "Is music playing?")
    assert fake.calls == 2  # the two signals at the first session, then the one new text
    model.release_text_encoder()
    assert model._encoder is None


def test_a_removed_questions_open_event_arrives_in_the_next_update():
    model = tiny_model(0, threshold=0.05, stop_threshold=0.01, start_secs=0.04)
    s = model.session()
    s.ask("q", "Is music playing?")
    s.push(synth_audio(8, 2.0))
    assert any(e.question == "q" and e.kind == "started" for e in s.events)
    now = s.time_s
    s.remove("q")
    nxt = s.push(synth_audio(9, 0.08))
    stopped = [e for u in nxt for e in u.events if e.question == "q"]
    assert len(stopped) == 1 and stopped[0].kind == "stopped" and stopped[0].end_s == now
    assert all("q" not in u.answers for u in nxt)
    assert [e for u in s.push(synth_audio(10, 0.08)) for e in u.events if e.question == "q"] == []


def test_bytes_and_dtypes():
    model = tiny_model(0)
    wave = synth_audio(11, 1.0)
    i16 = np.round(wave * 32768).clip(-32768, 32767).astype(np.int16)
    a = model.session()
    a.ask("q", "Is music playing?")
    from_array = a.push(i16) + a.finish()
    b = model.session()
    b.ask("q", "Is music playing?")
    from_bytes = b.push(i16.tobytes()[:1000]) + b.push(i16.tobytes()[1000:]) + b.finish()
    assert np.array_equal(probabilities(from_array)["q"], probabilities(from_bytes)["q"])
    b.reset()
    with pytest.raises(ValueError, match="int32"):
        b.push(np.zeros(1280, dtype=np.int32))
    with pytest.raises(ValueError, match="16-bit"):
        b.push(b"\x00\x01\x02")


def test_voice_memory_stays_bounded_and_does_not_change_answers():
    model = tiny_model(0)
    wave = synth_audio(12, 20.0)
    kept = model.session()
    kept.ask("q", "Is music playing?")
    kept._trim_voices = lambda: None  # the untrimmed reference
    trimmed = model.session()
    trimmed.ask("q", "Is music playing?")
    a = run(kept, wave, 1280)
    b = run(trimmed, wave, 1280)
    for q in probabilities(a):
        assert np.array_equal(probabilities(a)[q], probabilities(b)[q]), q
    assert kept._spk.shape[0] > 40
    assert trimmed._spk.shape[0] <= trimmed.n_spk + 2
