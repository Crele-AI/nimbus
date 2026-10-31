"""The decision head: read the audio once, answer every question against it.

A small transformer reads the audio side of one moment (the last 3 s of encoder frames
with their ages, the recent voice vectors with theirs, and the reference voice or a
learned no-reference token) into a memory, then answers every question by cross-attending
to that memory. Listening is shared: `encode_audio` runs once per moment, `answer` reads
the memory for any number of questions. Questions never see each other: a question's
token attends to the memory and to itself only, so adding, removing or reordering
questions leaves every other answer unchanged.

Audio side (`encode_audio`): frames, voice vectors and the reference each go through
their own LayerNorm and linear projection to `d_model`, plus a learned kind embedding
and, for frames and voice vectors, a learned age embedding (age in frames, 0 = now). A
missing reference is the learned `no_ref` vector. The three are one sequence of
`F + S + 1` positions through `audio_layers` pre-norm self-attention blocks with the
padded positions masked out of the keys, then a LayerNorm.

Question side (`answer`): each question's text vector goes through a LayerNorm and a
two-layer projection plus a type embedding, then `question_layers` blocks of
self-attention (within the question alone), cross-attention to the memory and an MLP,
all pre-norm, a final LayerNorm, and two linear readouts: the yes logit, divided by the
calibration before its sigmoid, and the can-tell logit.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, fields
from typing import Any

import torch
from torch import Tensor, nn

LAYER_NORM_EPS = 1e-6
KIND_FRAME, KIND_VOICE, KIND_REF = 0, 1, 2
"""Audio input kinds, the rows of the kind embedding."""
QUESTION_TYPE = 0
"""The type embedding row of a yes/no question."""


@dataclass
class AudioSide:
    """The audio side of a batch of moments. Masks are bool, True = valid; ages are in
    frames, 0 = now; the newest frame and voice vector are last."""

    frames: Tensor
    """`[B, F, frame_dim]` encoder frames, newest last."""
    frame_age: Tensor
    """`[B, F]` int."""
    frame_mask: Tensor
    """`[B, F]` bool, False before the stream's start."""
    spk: Tensor
    """`[B, S, spk_dim]` voice vectors, newest last."""
    spk_age: Tensor
    """`[B, S]` int."""
    spk_mask: Tensor
    """`[B, S]` bool, False on an empty slot."""
    ref: Tensor
    """`[B, spk_dim]` the reference voice, zeros where there is none."""
    ref_mask: Tensor
    """`[B]` bool, False = no reference, the learned token instead."""

    def to(self, *args, **kwargs) -> AudioSide:
        out = {}
        for f in fields(self):
            t = getattr(self, f.name)
            out[f.name] = (
                t.to(*args, **kwargs)
                if t.is_floating_point()
                else t.to(t.to(*args, **kwargs).device)
            )
        return AudioSide(**out)

    def validate(self) -> AudioSide:
        B, F, _ = _check_3d(self.frames, "frames")
        S = _check_3d(self.spk, "spk")[1]
        if self.spk.shape[0] != B or self.ref.shape != (B, self.spk.shape[2]):
            raise ValueError("frames, spk and ref must share the batch size and spk its width")
        _check_int(self.frame_age, (B, F), "frame_age")
        _check_int(self.spk_age, (B, S), "spk_age")
        _check_bool(self.frame_mask, (B, F), "frame_mask")
        _check_bool(self.spk_mask, (B, S), "spk_mask")
        _check_bool(self.ref_mask, (B,), "ref_mask")
        for age, name in ((self.frame_age, "frame_age"), (self.spk_age, "spk_age")):
            if age.numel() and int(age.min()) < 0:
                raise ValueError(f"{name} must be non-negative")
        return self


def default_ages(n: int, stride: int, device=None) -> Tensor:
    """`[n]` int64: `(n - 1 - i) * stride`, so the last slot is age 0."""
    return (n - 1 - torch.arange(n, device=device)) * stride


def audio_side(
    frames: Tensor,
    *,
    n_frames: int,
    n_spk: int,
    spk_dim: int,
    frame_mask: Tensor | None = None,
    spk: Tensor | None = None,
    spk_age: Tensor | None = None,
    spk_mask: Tensor | None = None,
    ref: Tensor | None = None,
    ref_mask: Tensor | None = None,
    spk_stride: int = 10,
) -> AudioSide:
    """An `AudioSide` from what the caller has, the rest filled: fewer frames or voice
    vectors than the slots are left-padded with zeros and masked out, so the newest stays
    last; frame ages are `n_frames - 1 - i`; voice ages default to `spk_stride` apart;
    no `spk` means every slot empty; no `ref` means no reference."""
    B, T, _ = _check_3d(frames, "frames")
    if T > n_frames:
        raise ValueError(f"frames has {T} frames, more than the {n_frames} slots")
    device = frames.device
    if frame_mask is None:
        frame_mask = torch.ones(B, T, dtype=torch.bool, device=device)
    _check_bool(frame_mask, (B, T), "frame_mask")
    frames = _left_pad(frames, n_frames)
    frame_mask = _left_pad(frame_mask, n_frames)
    frame_age = default_ages(n_frames, 1, device).expand(B, n_frames).clone()
    if spk is None:
        if spk_age is not None or spk_mask is not None:
            raise ValueError("spk_age and spk_mask need spk")
        spk = frames.new_zeros(B, n_spk, spk_dim)
        spk_mask = torch.zeros(B, n_spk, dtype=torch.bool, device=device)
        spk_age = torch.zeros(B, n_spk, dtype=torch.int64, device=device)
    else:
        Bs, s, _ = _check_3d(spk, "spk")
        if Bs != B or s > n_spk:
            raise ValueError(f"spk must be [{B}, <= {n_spk}, {spk_dim}], got {tuple(spk.shape)}")
        if spk_mask is None:
            spk_mask = torch.ones(B, s, dtype=torch.bool, device=device)
        _check_bool(spk_mask, (B, s), "spk_mask")
        if spk_age is None:
            spk_age = default_ages(n_spk, spk_stride, device)[n_spk - s :].expand(B, s).clone()
        _check_int(spk_age, (B, s), "spk_age")
        spk, spk_mask, spk_age = (
            _left_pad(spk, n_spk),
            _left_pad(spk_mask, n_spk),
            _left_pad(spk_age, n_spk),
        )
    if ref is None:
        if ref_mask is not None:
            raise ValueError("ref_mask needs ref")
        ref = spk.new_zeros(B, spk_dim)
        ref_mask = torch.zeros(B, dtype=torch.bool, device=device)
    else:
        if ref.shape != (B, spk_dim):
            raise ValueError(f"ref must be [{B}, {spk_dim}], got {tuple(ref.shape)}")
        if ref_mask is None:
            ref_mask = torch.ones(B, dtype=torch.bool, device=device)
        _check_bool(ref_mask, (B,), "ref_mask")
        ref = ref * ref_mask[:, None].to(ref.dtype)
    return AudioSide(
        frames, frame_age, frame_mask, spk, spk_age, spk_mask, ref, ref_mask
    ).validate()


def _left_pad(x: Tensor, n: int) -> Tensor:
    short = n - x.shape[1]
    if short == 0:
        return x
    pad = x.new_zeros((x.shape[0], short) + tuple(x.shape[2:]))
    return torch.cat([pad, x], dim=1)


def _check_3d(x: Tensor, name: str) -> tuple[int, int, int]:
    if not isinstance(x, Tensor) or x.dim() != 3 or not x.is_floating_point():
        raise ValueError(f"{name} must be a floating [B, T, D] tensor")
    return x.shape[0], x.shape[1], x.shape[2]


def _check_bool(x: Tensor, shape: tuple[int, ...], name: str) -> None:
    if not isinstance(x, Tensor) or x.dtype != torch.bool or tuple(x.shape) != shape:
        raise ValueError(f"{name} must be a bool tensor of shape {shape}")


def _check_int(x: Tensor, shape: tuple[int, ...], name: str) -> None:
    if (
        not isinstance(x, Tensor)
        or x.dtype not in (torch.int32, torch.int64)
        or tuple(x.shape) != shape
    ):
        raise ValueError(f"{name} must be an int tensor of shape {shape}")


# ---------------------------------------------------------------------------- the network


class Attention(nn.Module):
    """Multi-head attention of `x` over `context` with `allowed [B, Nq, Nk]` (True = may
    attend). Masked scores take the dtype's minimum before the softmax, so they weigh
    zero; a fully masked row averages the values instead of giving NaN."""

    def __init__(self, dim: int, n_heads: int) -> None:
        super().__init__()
        if dim % n_heads != 0:
            raise ValueError(f"d_model {dim} is not divisible by n_heads {n_heads}")
        self.n_heads = n_heads
        self.scale = (dim // n_heads) ** -0.5
        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, dim * 2)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: Tensor, context: Tensor, allowed: Tensor | None) -> Tensor:
        B, Nq, C = x.shape
        Nk = context.shape[1]
        H = self.n_heads
        q = self.q(x).reshape(B, Nq, H, C // H).transpose(1, 2)
        k, v = self.kv(context).reshape(B, Nk, 2, H, C // H).permute(2, 0, 3, 1, 4).unbind(0)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        if allowed is not None:
            attn = attn.masked_fill(~allowed[:, None, :, :], torch.finfo(attn.dtype).min)
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, Nq, C)
        return self.proj(out)


class Mlp(nn.Module):
    def __init__(self, dim: int, hidden: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x: Tensor) -> Tensor:
        return self.fc2(self.act(self.fc1(x)))


class AudioBlock(nn.Module):
    def __init__(self, dim: int, n_heads: int, mlp_ratio: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=LAYER_NORM_EPS)
        self.attn = Attention(dim, n_heads)
        self.norm2 = nn.LayerNorm(dim, eps=LAYER_NORM_EPS)
        self.mlp = Mlp(dim, int(dim * mlp_ratio))

    def forward(self, x: Tensor, allowed: Tensor) -> Tensor:
        h = self.norm1(x)
        x = x + self.attn(h, h, allowed)
        return x + self.mlp(self.norm2(x))


class QuestionBlock(nn.Module):
    def __init__(self, dim: int, n_heads: int, mlp_ratio: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=LAYER_NORM_EPS)
        self.self_attn = Attention(dim, n_heads)
        self.norm2 = nn.LayerNorm(dim, eps=LAYER_NORM_EPS)
        self.cross_attn = Attention(dim, n_heads)
        self.norm3 = nn.LayerNorm(dim, eps=LAYER_NORM_EPS)
        self.mlp = Mlp(dim, int(dim * mlp_ratio))

    def forward(
        self, x: Tensor, memory: Tensor, allowed_self: Tensor, allowed_cross: Tensor
    ) -> Tensor:
        h = self.norm1(x)
        x = x + self.self_attn(h, h, allowed_self)
        x = x + self.cross_attn(self.norm2(x), memory, allowed_cross)
        return x + self.mlp(self.norm3(x))


class DecisionHead(nn.Module):
    """See the module docstring. Ages must be below `max_age`."""

    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 4,
        audio_layers: int = 2,
        question_layers: int = 2,
        frame_dim: int = 768,
        spk_dim: int = 512,
        text_dim: int = 1024,
        max_age: int = 80,
        mlp_ratio: float = 4.0,
        calibration: float = 1.0,
    ) -> None:
        super().__init__()
        if audio_layers < 1 or question_layers < 1:
            raise ValueError("audio_layers and question_layers must be at least 1")
        if not math.isfinite(calibration) or calibration <= 0:
            raise ValueError(f"calibration must be a positive number, got {calibration}")
        self.d_model = int(d_model)
        self.max_age = int(max_age)
        self.frame_dim, self.spk_dim, self.text_dim = int(frame_dim), int(spk_dim), int(text_dim)
        self.calibration = float(calibration)
        self.frame_norm = nn.LayerNorm(frame_dim, eps=LAYER_NORM_EPS)
        self.frame_proj = nn.Linear(frame_dim, d_model)
        self.spk_norm = nn.LayerNorm(spk_dim, eps=LAYER_NORM_EPS)
        self.spk_proj = nn.Linear(spk_dim, d_model)
        self.ref_norm = nn.LayerNorm(spk_dim, eps=LAYER_NORM_EPS)
        self.ref_proj = nn.Linear(spk_dim, d_model)
        self.no_ref = nn.Parameter(torch.randn(d_model) * 0.02)
        self.age_embed = nn.Embedding(max_age, d_model)
        self.kind_embed = nn.Embedding(3, d_model)
        self.audio_blocks = nn.ModuleList(
            [AudioBlock(d_model, n_heads, mlp_ratio) for _ in range(audio_layers)]
        )
        self.memory_norm = nn.LayerNorm(d_model, eps=LAYER_NORM_EPS)
        self.text_norm = nn.LayerNorm(text_dim, eps=LAYER_NORM_EPS)
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, d_model), nn.GELU(), nn.Linear(d_model, d_model)
        )
        self.type_embed = nn.Embedding(3, d_model)
        self.question_blocks = nn.ModuleList(
            [QuestionBlock(d_model, n_heads, mlp_ratio) for _ in range(question_layers)]
        )
        self.question_norm = nn.LayerNorm(d_model, eps=LAYER_NORM_EPS)
        self.yes_head = nn.Linear(d_model, 1)
        self.can_tell_head = nn.Linear(d_model, 1)
        self.apply(self._init_weights)

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any], calibration: float) -> DecisionHead:
        return cls(
            d_model=int(cfg["d_model"]), n_heads=int(cfg["n_heads"]),
            audio_layers=int(cfg["audio_layers"]), question_layers=int(cfg["question_layers"]),
            frame_dim=int(cfg["frame_dim"]), spk_dim=int(cfg["spk_dim"]),
            text_dim=int(cfg["text_dim"]), max_age=int(cfg["max_age"]),
            calibration=calibration,
        )  # fmt: skip

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)

    def load_weights(self, state: Mapping[str, Tensor]) -> None:
        """Load a state dict strictly: every key must match."""
        self.load_state_dict(dict(state), strict=True)

    def _ages(self, age: Tensor, name: str) -> Tensor:
        if age.numel() and int(age.max()) >= self.max_age:
            raise ValueError(f"{name} must be below max_age {self.max_age}, got {int(age.max())}")
        return self.age_embed(age)

    def encode_audio(self, side: AudioSide) -> tuple[Tensor, Tensor]:
        """`(memory [B, M, d_model], memory_mask [B, M])`, `M = F + S + 1`."""
        side.validate()
        kind = self.kind_embed.weight
        f = self.frame_proj(self.frame_norm(side.frames)) + self._ages(side.frame_age, "frame_age")
        f = f + kind[KIND_FRAME]
        s = self.spk_proj(self.spk_norm(side.spk)) + self._ages(side.spk_age, "spk_age")
        s = s + kind[KIND_VOICE]
        r = self.ref_proj(self.ref_norm(side.ref))
        r = torch.where(side.ref_mask[:, None], r, self.no_ref[None, :].to(r.dtype))
        r = (r + kind[KIND_REF])[:, None, :]
        x = torch.cat([f, s, r], dim=1)
        mask = torch.cat(
            [side.frame_mask, side.spk_mask, torch.ones_like(side.ref_mask)[:, None]], dim=1
        )
        allowed = mask[:, None, :].expand(-1, x.shape[1], -1)
        for block in self.audio_blocks:
            x = block(x, allowed)
        return self.memory_norm(x), mask

    def answer(self, memory: Tensor, memory_mask: Tensor, vectors: Tensor) -> tuple[Tensor, Tensor]:
        """`(p_yes [B, N], p_can_tell [B, N])` for `vectors [B, N, text_dim]`, one text
        vector per question; the yes logit is divided by the calibration."""
        if vectors.dim() != 3 or vectors.shape[0] != memory.shape[0]:
            raise ValueError(
                f"vectors must be [B={memory.shape[0]}, N, {self.text_dim}], "
                f"got {tuple(vectors.shape)}"
            )
        B, N, _ = vectors.shape
        x = self.text_proj(self.text_norm(vectors)) + self.type_embed.weight[QUESTION_TYPE]
        allowed_self = torch.eye(N, dtype=torch.bool, device=x.device)[None].expand(B, -1, -1)
        allowed_cross = memory_mask[:, None, :].expand(-1, N, -1)
        for block in self.question_blocks:
            x = block(x, memory, allowed_self, allowed_cross)
        hidden = self.question_norm(x)
        yes_logit = self.yes_head(hidden).squeeze(-1) / self.calibration
        can_tell_logit = self.can_tell_head(hidden).squeeze(-1)
        return torch.sigmoid(yes_logit), torch.sigmoid(can_tell_logit)

    def forward(self, side: AudioSide, vectors: Tensor) -> tuple[Tensor, Tensor]:
        memory, mask = self.encode_audio(side)
        return self.answer(memory, mask, vectors)
