"""The decision head: shapes, masks, the independence of questions, the calibration."""

from __future__ import annotations

import pytest
import torch

from crele_nimbus.head import DecisionHead, audio_side


def tiny_head(calibration: float = 1.0) -> DecisionHead:
    torch.manual_seed(0)
    return DecisionHead(
        d_model=32, n_heads=2, audio_layers=1, question_layers=1, frame_dim=16, spk_dim=8,
        text_dim=12, max_age=80, calibration=calibration,
    ).eval()  # fmt: skip


def side(B: int = 2, T: int = 5, s: int = 3, ref: bool = True):
    torch.manual_seed(1)
    frames = torch.randn(B, T, 16)
    spk = torch.randn(B, s, 8)
    r = torch.randn(B, 8) if ref else None
    return audio_side(frames, n_frames=75, n_spk=8, spk_dim=8, spk=spk, ref=r)


def test_audio_side_padding_and_ages():
    a = side()
    assert a.frames.shape == (2, 75, 16) and a.spk.shape == (2, 8, 8)
    assert a.frame_mask.sum().item() == 2 * 5 and a.spk_mask.sum().item() == 2 * 3
    assert a.frame_age[0, -1].item() == 0 and a.frame_age[0, 0].item() == 74
    assert a.spk_age[0, -3:].tolist() == [20, 10, 0]
    b = side(ref=False)
    assert not b.ref_mask.any() and not b.ref.any()
    with pytest.raises(ValueError):
        audio_side(torch.randn(1, 76, 16), n_frames=75, n_spk=8, spk_dim=8)


def test_shapes_and_questions_are_independent():
    head = tiny_head()
    a = side()
    v = torch.randn(2, 4, 12)
    with torch.no_grad():
        p, c = head(a, v)
        p2, c2 = head(a, v[:, [2, 0]])
        p3, _ = head(a, v[:, :1])
    assert p.shape == c.shape == (2, 4)
    close = lambda a, b: torch.testing.assert_close(a, b, atol=1e-6, rtol=0)  # noqa: E731
    close(p2[:, 0], p[:, 2])
    close(p2[:, 1], p[:, 0])
    close(p3[:, 0], p[:, 0])
    assert ((p > 0) & (p < 1)).all() and ((c > 0) & (c < 1)).all()


def test_masked_slots_do_not_matter():
    head = tiny_head()
    a = side(T=3, s=1)
    v = torch.randn(2, 2, 12)
    with torch.no_grad():
        p, c = head(a, v)
        a.frames[:, :10] = 7.0  # padded frames
        a.spk[:, :5] = -3.0  # empty voice slots
        q, d = head(a, v)
    torch.testing.assert_close(p, q, atol=1e-6, rtol=0)
    torch.testing.assert_close(c, d, atol=1e-6, rtol=0)


def test_calibration_divides_the_yes_logit_only():
    v = torch.randn(1, 3, 12)
    a = side(B=1)
    with torch.no_grad():
        p1, c1 = tiny_head(1.0)(a, v)
        p2, c2 = tiny_head(2.0)(a, v)
    logit = torch.logit(p1)
    torch.testing.assert_close(torch.sigmoid(logit / 2.0), p2, atol=1e-5, rtol=0)
    torch.testing.assert_close(c1, c2, atol=0, rtol=0)
    with pytest.raises(ValueError):
        tiny_head(0.0)


def test_load_weights_is_strict():
    head = tiny_head()
    state = dict(head.state_dict())
    head.load_weights(state)
    with pytest.raises(RuntimeError):
        head.load_weights({**state, "extra.weight": torch.zeros(1)})
    del state["yes_head.bias"]
    with pytest.raises(RuntimeError):
        head.load_weights(state)
    bad = side()
    bad.frame_age[0, 0] = 80
    with pytest.raises(ValueError, match="max_age"):
        head.encode_audio(bad)
