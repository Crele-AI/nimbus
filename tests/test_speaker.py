"""The voice model's front end against Kaldi's, and its shapes."""

from __future__ import annotations

import pytest
import torch

from conftest import synth_audio
from crele_nimbus.speaker import MIN_SAMPLES, VoiceModel, kaldi_mel_banks


def test_mel_banks_and_fbank_match_torchaudio_kaldi():
    pytest.importorskip("torchaudio")
    from torchaudio.compliance import kaldi

    theirs, _ = kaldi.get_mel_banks(80, 512, 16000.0, 20.0, 0.0, 100.0, -500.0, 1.0)
    assert torch.equal(kaldi_mel_banks(), theirs)
    model = VoiceModel()
    wave = torch.from_numpy(synth_audio(1, 1.7))[None]
    wave = torch.cat([wave, 0.5 * wave.flip(1)])
    ours = model.fbank(wave)
    for b in range(wave.shape[0]):
        ref = kaldi.fbank(wave[b : b + 1], num_mel_bins=80, sample_frequency=16000, dither=0.0)
        ref = ref - ref.mean(dim=0, keepdim=True)
        torch.testing.assert_close(ours[b], ref, atol=1e-4, rtol=1e-5)


def test_embed_shapes_and_independence():
    model = VoiceModel()
    a = torch.from_numpy(synth_audio(2, 2.0))
    b = torch.from_numpy(synth_audio(3, 2.0))
    both = model.embed(torch.stack([a, b]))
    assert both.shape == (2, 512)
    torch.testing.assert_close(model.embed(a[None])[0], both[0], atol=1e-5, rtol=0)
    with pytest.raises(ValueError):
        model.embed(torch.zeros(1, MIN_SAMPLES - 1))
    assert not any(p.requires_grad for p in model.parameters())
    model.train()
    assert not model.training
