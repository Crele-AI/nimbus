"""The text encoder: a question in plain English to one unit vector, once.

Questions are read by Qwen3-Embedding-0.6B, unchanged, at a pinned revision. Each text
is embedded in the model's query form, `Instruct: {instruction}\\nQuery:{text}`, with
one fixed instruction (from the model's config); the hidden state of the last token is
pooled and L2-normalised. The vectors are what the decision head answers against, so the
template, the pooling and the precision must stay exactly as the config says.

A question with criteria is embedded as three lines: the question, `True: {yes_means}`
and `False: {no_means}` (`question_text`).

Precision: float32 on the CPU, bfloat16 on CUDA (the published weights are bfloat16; the
pooled vector is cast to float32 before the norm). The two differ by bfloat16 noise, so
answers on CUDA differ slightly from the CPU's.

`TextEncoder` loads `transformers` and the weights when built; `VectorCache` keeps every
vector by text so a question is encoded once per process.
"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np


def question_text(text: str, yes_means: str | None = None, no_means: str | None = None) -> str:
    """The text the model reads: the question alone, or with its criteria. Both criteria
    or neither; empty strings count as missing."""
    text = str(text).strip()
    if not text:
        raise ValueError("a question needs some text")
    yes_means = (yes_means or "").strip()
    no_means = (no_means or "").strip()
    if bool(yes_means) != bool(no_means):
        raise ValueError("criteria need both sides: yes_means and no_means")
    if not yes_means:
        return text
    return f"{text}\nTrue: {yes_means}\nFalse: {no_means}"


def template(instruction: str, text: str) -> str:
    return f"Instruct: {instruction}\nQuery:{text}"


def last_token_pool(hidden, attention_mask):
    """The hidden state of the last real token of each row, under left or right padding."""
    import torch

    left_padding = bool((attention_mask[:, -1].sum() == attention_mask.shape[0]).item())
    if left_padding:
        return hidden[:, -1]
    lengths = attention_mask.sum(dim=1) - 1
    rows = torch.arange(hidden.shape[0], device=hidden.device)
    return hidden[rows, lengths]


class TextEncoder:
    """Qwen3-Embedding-0.6B as the config names it (module docstring).

    `source` is a local folder with the model's files, or None for the config's
    repository at its pinned revision, fetched into the Hugging Face cache on first use
    (about 1.2 GB).
    """

    def __init__(
        self,
        cfg: Mapping[str, Any],
        *,
        source: str | os.PathLike[str] | None = None,
        device: str = "cpu",
        batch_size: int = 32,
    ) -> None:
        import torch
        import transformers
        from transformers import AutoModel, AutoTokenizer

        self.instruction = str(cfg["instruction"])
        self.dim = int(cfg["dim"])
        self.max_length = int(cfg["max_length"])
        self.repo = str(cfg["repo"])
        self.revision = str(cfg["revision"])
        self.device = str(device)
        self.batch_size = int(batch_size)
        self.dtype = torch.bfloat16 if self.device.startswith("cuda") else torch.float32
        if source is not None:
            where, kw = str(Path(source)), {"local_files_only": True}
            if not (Path(where) / "config.json").is_file():
                raise FileNotFoundError(f"{where} is not a model folder (no config.json)")
        else:
            where, kw = self.repo, {"revision": self.revision}
        dtype_key = "dtype" if int(transformers.__version__.split(".")[0]) >= 5 else "torch_dtype"
        self.tokenizer = AutoTokenizer.from_pretrained(where, padding_side="left", **kw)
        self.tokenizer.padding_side = "left"
        self.model = AutoModel.from_pretrained(where, **{dtype_key: self.dtype}, **kw)
        self.model = self.model.to(self.device).eval()
        hidden = getattr(self.model.config, "hidden_size", None)
        if hidden is not None and hidden < self.dim:
            raise ValueError(f"text_encoder.dim {self.dim} is over the model's width {hidden}")

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        """`[len(texts), dim]` float32 unit vectors, row i for `texts[i]`. A text longer
        than `max_length` tokens in its templated form is refused, never truncated."""
        import torch
        import torch.nn.functional as F

        if isinstance(texts, str):
            raise TypeError("embed takes a list of texts")
        texts = list(texts)
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        with torch.no_grad():
            for start in range(0, len(texts), self.batch_size):
                chunk = texts[start : start + self.batch_size]
                batch = self.tokenizer(
                    [template(self.instruction, t) for t in chunk],
                    padding=True, truncation=False, return_tensors="pt",
                )  # fmt: skip
                lengths = batch["attention_mask"].sum(dim=1).tolist()
                for t, n in zip(chunk, lengths, strict=True):
                    if n > self.max_length:
                        raise ValueError(
                            f"the question is {n} tokens, over {self.max_length}: {t!r}"
                        )
                batch = {k: v.to(self.device) for k, v in batch.items()}
                hidden = self.model(**batch).last_hidden_state
                pooled = last_token_pool(hidden, batch["attention_mask"])[:, : self.dim].float()
                out[start : start + len(chunk)] = F.normalize(pooled, p=2, dim=1).cpu().numpy()
        return out


class VectorCache:
    """Vectors by text, computed once through an encoder."""

    def __init__(self, encoder: Any) -> None:
        self.encoder = encoder
        self._rows: dict[str, np.ndarray] = {}

    @property
    def dim(self) -> int:
        return int(self.encoder.dim)

    def get(self, texts: Sequence[str]) -> np.ndarray:
        """`[len(texts), dim]`, encoding the texts not yet held, in one batch."""
        missing = list(dict.fromkeys(t for t in texts if t not in self._rows))
        if missing:
            vectors = np.asarray(self.encoder.embed(missing), dtype=np.float32)
            for t, v in zip(missing, vectors, strict=True):
                self._rows[t] = v
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.stack([self._rows[t] for t in texts]).astype(np.float32)
