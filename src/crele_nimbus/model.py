"""The loaded model: the three networks, the text encoder and the config, shared by any
number of sessions.

```python
model = Nimbus.from_pretrained("crele/nimbus-v1")       # or a local folder
session = model.session()
```

`from_pretrained` reads `config.json` and the three weight files from a local folder or
from a Hugging Face repository (fetched into the Hugging Face cache, about 0.4 GB). The
text encoder (Qwen3-Embedding-0.6B, about 1.2 GB, at the config's pinned revision) is
loaded at the first `session()`, which encodes the two voice signals' questions, and
kept; `release_text_encoder` frees it until the next new question.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from typing import Any

import numpy as np
import torch

from crele_nimbus.config import NimbusConfig
from crele_nimbus.head import DecisionHead
from crele_nimbus.listener import Listener
from crele_nimbus.session import Session
from crele_nimbus.speaker import VoiceModel
from crele_nimbus.text import TextEncoder, VectorCache


class Nimbus:
    """The model (module docstring). Build it with `from_pretrained`, or directly from
    its parts for tests."""

    def __init__(
        self,
        config: NimbusConfig,
        listener: Listener,
        head: DecisionHead,
        voice: VoiceModel,
        *,
        device: str | torch.device = "cpu",
        text_encoder: Any | str | os.PathLike[str] | None = None,
    ) -> None:
        self.config = config
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError(
                "device cuda was asked for, but this PyTorch build sees no CUDA device"
            )
        self.listener = listener.to(self.device).eval().requires_grad_(False)
        self.head = head.to(self.device).eval().requires_grad_(False)
        self.voice = voice.to(self.device).eval()
        self.read_layer = self.listener.check_layer(int(config.listener["read_layer"]))
        if self.listener.embed_dim != int(config.head["frame_dim"]):
            raise ValueError("the encoder's width differs from the head's frame_dim")
        if self.voice.dim != int(config.head["spk_dim"]):
            raise ValueError("the voice model's width differs from the head's spk_dim")
        self._text_source: str | os.PathLike[str] | None = None
        self._encoder: Any | None = None
        if text_encoder is None or isinstance(text_encoder, str | os.PathLike):
            self._text_source = text_encoder
        else:
            self._encoder = text_encoder
        self._cache: VectorCache | None = None

    @classmethod
    def from_pretrained(
        cls,
        path_or_id: str | os.PathLike[str] = "crele/nimbus-v1",
        *,
        device: str | torch.device = "cpu",
        text_encoder: str | os.PathLike[str] | None = None,
        revision: str | None = None,
        token: str | bool | None = None,
    ) -> Nimbus:
        """Load the model from a local folder holding `config.json` and the weights, or
        from a Hugging Face repository id. `text_encoder` names a local folder with the
        text encoder's files; without it the pinned revision is fetched on the first
        question. `revision` and `token` go to the Hugging Face download."""
        from crele_nimbus.weights import load_folder, resolve

        folder = resolve(path_or_id, revision=revision, token=token)
        config, listener, head, voice = load_folder(folder)
        return cls(config, listener, head, voice, device=device, text_encoder=text_encoder)

    def session(self) -> Session:
        """A new stream through the model."""
        return Session(self)

    # ------------------------------------------------------------------ the text encoder

    @property
    def text_encoder(self) -> Any:
        """The text encoder, loaded on first use."""
        if self._encoder is None:
            self._encoder = TextEncoder(
                self.config.text_encoder, source=self._text_source, device=str(self.device)
            )
        return self._encoder

    def release_text_encoder(self) -> None:
        """Free the text encoder; the next question loads it again. The vectors of the
        questions already asked are kept."""
        self._encoder = None

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """`[len(texts), text_dim]` question vectors, each text encoded once per model."""
        if self._cache is None:
            self._cache = VectorCache(_LazyEncoder(self))
        return self._cache.get(list(texts))


class _LazyEncoder:
    """The model's text encoder behind the cache, loaded only when a text is missing."""

    def __init__(self, model: Nimbus) -> None:
        self.model = model

    @property
    def dim(self) -> int:
        return int(self.model.config.text_encoder["dim"])

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        return self.model.text_encoder.embed(texts)
