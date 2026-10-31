"""CAM++ as one module on waveforms: 16 kHz audio in, a 512-dim voice vector out.

`VoiceModel.embed(wave [B, S])` embeds each row of a batch of equal-length clips. The
features are Kaldi's 80-bin log mel filterbank as 3D-Speaker computes them for this
checkpoint, reproduced here in plain torch (`fbank`), batched over rows:

| Step | Value |
|---|---|
| frames | 400 samples (25 ms) every 160 (10 ms); only frames wholly inside the signal |
| DC offset | each frame's mean subtracted |
| pre-emphasis | `x[j] -= 0.97 x[j-1]`, the first sample against itself |
| window | a symmetric Hann window to the power 0.85 (Kaldi's "povey") |
| FFT | frames zero-padded to 512; the power spectrum |
| mel | 80 triangular bins from 20 Hz to the Nyquist frequency (`kaldi_mel_banks`) |
| log | the natural log, floored at float32 epsilon |
| mean | the mean over frames of each bin subtracted (3D-Speaker's `mean_nor`) |

The model is frozen: always in eval mode, no gradients.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from crele_nimbus.config import SAMPLE_RATE

from .DTDNN import CAMPPlus

EMBEDDING_SIZE = 512
N_MELS = 80
FRAME_LENGTH = 400
FRAME_SHIFT = 160
N_FFT = 512
PREEMPHASIS = 0.97
LOW_FREQ_HZ = 20.0
HIGH_FREQ_HZ = 0.0
"""Kaldi's convention: 0 or less is an offset from the Nyquist frequency, so 0 is Nyquist."""
POVEY_POWER = 0.85
MIN_SAMPLES = FRAME_LENGTH
"""The shortest clip `fbank` accepts: one 25 ms frame."""
CAMPPLUS_ARGS: dict[str, Any] = {"feat_dim": N_MELS, "embedding_size": EMBEDDING_SIZE}
_EPS = float(torch.finfo(torch.float32).eps)


def kaldi_mel_banks(
    num_bins: int = N_MELS,
    n_fft: int = N_FFT,
    sample_rate: float = float(SAMPLE_RATE),
    low_freq: float = LOW_FREQ_HZ,
    high_freq: float = HIGH_FREQ_HZ,
) -> Tensor:
    """Kaldi's mel filterbank `[num_bins, n_fft // 2]` (no warping), as torchaudio's
    `compliance.kaldi.get_mel_banks` computes it, step for step in float32."""
    nyquist = 0.5 * sample_rate
    if high_freq <= 0.0:
        high_freq += nyquist
    if not (0.0 <= low_freq < nyquist and 0.0 < high_freq <= nyquist and low_freq < high_freq):
        raise ValueError(f"bad mel bank limits: {low_freq} to {high_freq} Hz at {sample_rate} Hz")
    fft_bin_width = sample_rate / n_fft
    mel_low = 1127.0 * math.log(1.0 + low_freq / 700.0)
    mel_high = 1127.0 * math.log(1.0 + high_freq / 700.0)
    mel_delta = (mel_high - mel_low) / (num_bins + 1)
    bin = torch.arange(num_bins).unsqueeze(1)
    left = mel_low + bin * mel_delta
    center = mel_low + (bin + 1.0) * mel_delta
    right = mel_low + (bin + 2.0) * mel_delta
    freqs = fft_bin_width * torch.arange(n_fft / 2)
    mel = (1127.0 * (1.0 + freqs / 700.0).log()).unsqueeze(0)
    up = (mel - left) / (center - left)
    down = (right - mel) / (right - center)
    return torch.max(torch.zeros(1), torch.min(up, down))


class VoiceModel(nn.Module):
    """CAM++ with its filterbank front end, frozen (module docstring). Built with random
    weights; load `speaker.safetensors` into `.model` with `load_state_dict`."""

    def __init__(self, *, embedding_size: int = EMBEDDING_SIZE) -> None:
        super().__init__()
        self.model = CAMPPlus(**{**CAMPPLUS_ARGS, "embedding_size": embedding_size})
        self.dim = int(embedding_size)
        mel = kaldi_mel_banks()
        # Kaldi pads the banks with a zero column for the Nyquist bin; so does this.
        mel_fb = F.pad(mel.float(), (0, 1)).T.contiguous()  # [N_FFT // 2 + 1, N_MELS]
        window = torch.hann_window(FRAME_LENGTH, periodic=False).pow(POVEY_POWER)
        self.register_buffer("mel_fb", mel_fb, persistent=False)
        self.register_buffer("window", window, persistent=False)
        self.requires_grad_(False)
        self.train(False)

    @property
    def device(self) -> torch.device:
        return self.mel_fb.device

    def train(self, mode: bool = True) -> VoiceModel:
        return super().train(False)

    def fbank(self, wave: Tensor) -> Tensor:
        """`[B, S]` float waveforms in [-1, 1] -> `[B, T, 80]` mean-normalised log
        filterbanks, `T = 1 + (S - 400) // 160`."""
        if wave.dim() != 2:
            raise ValueError(f"wave must be [B, S], got shape {tuple(wave.shape)}")
        if wave.shape[1] < MIN_SAMPLES:
            raise ValueError(f"wave must have at least {MIN_SAMPLES} samples, got {wave.shape[1]}")
        x = wave.to(device=self.device, dtype=torch.float32)
        frames = x.unfold(-1, FRAME_LENGTH, FRAME_SHIFT)  # [B, T, 400]
        frames = frames - frames.mean(dim=-1, keepdim=True)
        previous = torch.cat([frames[..., :1], frames[..., :-1]], dim=-1)
        frames = (frames - PREEMPHASIS * previous) * self.window
        power = torch.fft.rfft(frames, n=N_FFT).abs().pow(2.0)  # [B, T, 257]
        logmel = torch.clamp_min(power @ self.mel_fb, _EPS).log()  # [B, T, 80]
        return logmel - logmel.mean(dim=1, keepdim=True)

    @torch.no_grad()
    def embed(self, wave: Tensor) -> Tensor:
        """`[B, S]` 16 kHz waveforms -> `[B, 512]` voice vectors; each row depends only
        on its own clip."""
        return self.model(self.fbank(wave))

    def forward(self, wave: Tensor) -> Tensor:
        return self.embed(wave)
