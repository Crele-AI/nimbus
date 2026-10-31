"""Parity with the reference: the released weights on the reference audio must give the
reference answers to within 5e-5 and its states and events exactly. The bar allows for
float32 differences between PyTorch versions; it is far below any setting a user would
choose.

Needs the weights: set `NIMBUS_WEIGHTS` to a folder holding `config.json` and the three
weight files, and optionally `NIMBUS_TEXT_ENCODER` to a local folder with the text
encoder's files (else it is fetched from Hugging Face at the pinned revision).
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from crele_nimbus import Nimbus

WEIGHTS = os.environ.get("NIMBUS_WEIGHTS")
TEXT = os.environ.get("NIMBUS_TEXT_ENCODER")
pytestmark = pytest.mark.skipif(not WEIGHTS, reason="NIMBUS_WEIGHTS is not set")
STATE = {"no": 0, "yes": 1, "cant_tell": 2}
TOLERANCE = 5e-5


@pytest.fixture(scope="module")
def model():
    return Nimbus.from_pretrained(WEIGHTS, text_encoder=TEXT)


def ask_reference_questions(session, reference):
    for i, line in enumerate(reference.questions, start=1):
        parts = [p.strip() for p in line.split("||")]
        kw = dict(reference.question_settings.get(f"q{i}", {}))
        if len(parts) == 3:
            session.ask(f"q{i}", parts[0], yes_means=parts[1], no_means=parts[2], **kw)
        else:
            session.ask(f"q{i}", parts[0], **kw)


def run(session, reference, chunk: int) -> dict[str, np.ndarray]:
    updates = []
    x = reference.stream
    for s in range(0, x.shape[0], chunk):
        updates += session.push(x[s : s + chunk])
    updates += session.finish()
    ids = [q for q, _ in session.questions]
    return {
        "ids": np.array(ids),
        "t": np.array([u.frame for u in updates]),
        "p": np.array([[u.answers[q].probability for q in ids] for u in updates]),
        "p_can_tell": np.array([[u.answers[q].can_tell for q in ids] for u in updates]),
        "state": np.array(
            [[STATE[u.answers[q].answer] for q in ids] for u in updates], dtype=np.int8
        ),
        "events": np.array(
            [
                (ids.index(e.question), e.start, -1 if e.end is None else e.end, e.detect,
                 e.confirm, -1 if e.end_confirm is None else e.end_confirm)
                for e in session.frame_events()
            ],
            dtype=np.int64,
        ).reshape(-1, 6),
    }  # fmt: skip


@pytest.mark.parametrize("name", ["enrolled", "no_enrollment"])
def test_answers_states_and_events_match_the_reference(model, reference, name):
    session = model.session()
    if name == "enrolled":
        session.enroll(reference.enroll)
    ask_reference_questions(session, reference)
    got = run(session, reference, reference.chunk_samples)
    expected = reference.run(name)
    assert np.array_equal(got["ids"], expected["ids"])
    assert np.array_equal(got["t"], expected["t"])
    dp = float(np.abs(got["p"] - expected["p"]).max())
    dc = float(np.abs(got["p_can_tell"] - expected["p_can_tell"]).max())
    assert dp <= TOLERANCE and dc <= TOLERANCE, (dp, dc)
    assert np.array_equal(got["state"], expected["state"])
    assert np.array_equal(got["events"], expected["events"])
    print(
        f"\n{name}: max |dp| {dp:.2e}, max |dcan_tell| {dc:.2e}, {len(expected['events'])} events"
    )


def test_other_chunkings_give_the_same_frames(model, reference):
    a = model.session()
    ask_reference_questions(a, reference)
    whole = run(a, reference, reference.stream.shape[0])
    b = model.session()
    ask_reference_questions(b, reference)
    pieces = run(b, reference, 4096)
    assert np.array_equal(whole["p"], pieces["p"])
    assert np.array_equal(whole["events"], pieces["events"])
