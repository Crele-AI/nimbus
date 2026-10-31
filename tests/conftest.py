"""Shared fixtures: the parity reference and the tiny random models.

Every test here runs on the CPU without a download. The reference (`tests/data/
reference.npz`) holds synthetic audio and the answers the released weights give on it;
only `test_parity.py` needs the weights themselves, and it is skipped unless
`NIMBUS_WEIGHTS` names a folder holding them.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

DATA = Path(__file__).resolve().parent / "data"
REFERENCE = DATA / "reference.npz"


class Reference:
    """The reference file, with the convenient views the tests use."""

    def __init__(self, path: Path = REFERENCE) -> None:
        self.data = np.load(path)
        self.stream = (self.data["stream_int16"].astype(np.float32) / 32768.0).astype(np.float32)
        self.enroll = (self.data["enroll_int16"].astype(np.float32) / 32768.0).astype(np.float32)
        self.questions = [str(q) for q in self.data["questions"]]
        self.question_settings = json.loads(str(self.data["question_settings"]))
        self.chunk_samples = int(self.data["chunk_samples"])
        self.states = [str(s) for s in self.data["states"]]

    def run(self, name: str) -> dict[str, np.ndarray]:
        return {
            k: self.data[f"{name}_{k}"] for k in ("ids", "t", "p", "p_can_tell", "state", "events")
        }


@pytest.fixture(scope="session")
def reference() -> Reference:
    return Reference()


# ---------------------------------------------------------------------------- tiny models


def synth_audio(seed: int, seconds: float) -> np.ndarray:
    """Speech-like synthetic audio: a buzzing tone with a slow envelope under noise,
    float32 in [-1, 1]. A fact about the code paths, not about the model."""
    rng = np.random.default_rng(seed)
    n = int(round(seconds * 16000))
    t = np.arange(n) / 16000.0
    f0 = 110 + 40 * np.sin(2 * np.pi * 0.7 * t)
    phase = np.cumsum(2 * np.pi * f0 / 16000.0)
    voice = sum(np.sin(k * phase) / k for k in range(1, 8))
    envelope = 0.5 + 0.5 * np.sin(2 * np.pi * 2.3 * t) ** 2
    x = 0.3 * voice * envelope + 0.02 * rng.standard_normal(n)
    return np.clip(x, -1, 1).astype(np.float32)


def tiny_listener(seed: int = 0):
    """A narrow random encoder (width 32, 4 blocks, 4 heads) with random but plausible
    front-end statistics, in eval mode."""
    import torch

    from crele_nimbus.listener import Listener

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = Listener(embed_dim=32, depth=4, num_heads=4, mlp_ratio=4.0, qkv_bias=True)
        bn = model.init_bn.bn
        with torch.no_grad():
            bn.running_mean.copy_(torch.randn(bn.num_features) * 10.0 - 30.0)
            bn.running_var.copy_(torch.rand(bn.num_features) * 300.0 + 100.0)
            bn.weight.copy_(torch.rand(bn.num_features) + 0.5)
            bn.bias.copy_(torch.randn(bn.num_features) * 0.1)
            fb = torch.rand(257, 64) * (torch.rand(257, 64) < 0.05)
            model.front_end.mel.mel_scale.fb.copy_(fb)
    return model.eval()


class FakeTextEncoder:
    """Deterministic unit vectors from a hash of the text: no meaning, no download."""

    def __init__(self, dim: int = 24) -> None:
        self.dim = dim
        self.calls = 0

    def embed(self, texts):
        import hashlib

        self.calls += 1
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            seed = int.from_bytes(hashlib.sha256(t.encode()).digest()[:8], "little")
            v = np.random.default_rng(seed).standard_normal(self.dim)
            out[i] = v / np.linalg.norm(v)
        return out


def tiny_config_mapping() -> dict:
    return {
        "model": "tiny",
        "sample_rate": 16000,
        "frame_ms": 40,
        "chunk_frames": 2,
        "listener": {"embed_dim": 32, "depth": 4, "num_heads": 4, "mlp_ratio": 4.0,
                     "qkv_bias": True, "read_layer": 4},
        "head": {"d_model": 32, "n_heads": 2, "audio_layers": 1, "question_layers": 1,
                 "frame_dim": 32, "spk_dim": 64, "text_dim": 24, "max_age": 80,
                 "context_frames": 75, "voice_slots": 8},
        "voice": {"window_s": 2.0, "hop_s": 0.4},
        "text_encoder": {"repo": "none", "revision": "none", "instruction": "test",
                         "pooling": "last_token", "normalize": True, "dim": 24, "max_length": 256},
        "calibration": 1.3,
        "defaults": {"threshold": 0.7, "stop_threshold": 0.35, "start_secs": 0.12,
                     "stop_secs": 0.3, "cant_tell_below": 0.5},
        "voice_signals": {"user": "Is the user speaking?",
                          "other": "Is someone other than the user speaking?"},
    }  # fmt: skip


def tiny_model(seed: int = 0, **overrides):
    """A `Nimbus` from tiny random parts; `overrides` patch the config's defaults block."""
    import torch

    from crele_nimbus.config import NimbusConfig
    from crele_nimbus.head import DecisionHead
    from crele_nimbus.model import Nimbus
    from crele_nimbus.speaker import VoiceModel

    data = tiny_config_mapping()
    data["defaults"].update(overrides)
    config = NimbusConfig.from_mapping(data)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed + 1)
        head = DecisionHead.from_config(config.head, config.calibration)
        voice = VoiceModel(embedding_size=64)
    return Nimbus(config, tiny_listener(seed), head, voice, text_encoder=FakeTextEncoder(24))


@pytest.fixture
def model():
    return tiny_model(0)
