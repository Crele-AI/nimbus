"""The streaming audio encoder: 16 kHz audio in, one 768-wide frame per 40 ms out.

The encoder has the structure of Dasheng-Base (a mel front end, a patch embedding, 12
pre-norm transformer blocks, a final LayerNorm) and was trained to hear only the past:
every block attends within aligned chunks of `chunk_frames` frames (two, 80 ms) and to all
earlier chunks, so the frame at index t depends on the audio up to the end of its chunk
and on nothing later. Module names follow the weight file's keys.

**The time grid.** A mel frame every 160 samples (10 ms), four mel frames per encoder
frame (640 samples, 40 ms). The mel front end is centred: mel frame k reads samples
[160k - 256, 160k + 256), with the samples before 0 mirrored. Positions are learned over a
window of `BLOCK_FRAMES` (252) encoder frames, about 10 s; longer audio is encoded in
windows of that size, each restarting the positions and the chunks.

**Streaming** (`BlockStream`). One window, fed sample chunk by sample chunk, reproduces the
batch forward of the whole window exactly. It keeps the window's samples, and per block
the queries, keys and values projected so far, and emits a whole chunk of frames as soon
as the chunk's last sample plus the front end's reach (96 samples past the chunk) has
arrived. `flush()` ends the window: it mirrors the right edge as the batch front end does,
embeds the last frame and finalises the last, possibly short, chunk.

`StaggeredStream` runs two block streams, the second starting 126 frames (5.04 s) later,
and reads each frame from the one that is further into its window, so after the first
window every frame has at least 5 s of audio behind it. This is what the session uses.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor, nn

from crele_nimbus.config import FRAME_SAMPLES

MEL_HOP = 160
MEL_FRAMES_PER_FRAME = 4
N_FFT = 512
WIN_LENGTH = 512
N_MELS = 64
CENTER_PAD = N_FFT // 2
"""Samples mirrored on each side by the centred front end: 256."""
BLOCK_FRAMES = 252
"""Encoder frames in one attention window (1008 mel frames): 10.08 s."""
BLOCK_SAMPLES = BLOCK_FRAMES * FRAME_SAMPLES
"""Samples in one window: 161,280."""
DEFAULT_OFFSET_FRAMES = BLOCK_FRAMES // 2
FRAME_REACH = 3 * MEL_HOP + CENTER_PAD - FRAME_SAMPLES
"""Samples a frame reads past its own 640: 96."""
PATCH = (N_MELS, MEL_FRAMES_PER_FRAME)
LAYER_NORM_EPS = 1e-6
DB_AMIN = 1e-10


def n_frames(n_samples: int) -> int:
    """Encoder frames the front end yields for `n_samples`: `(n // 160 + 1) // 4`."""
    if n_samples < 0:
        raise ValueError(f"n_samples must be non-negative, got {n_samples}")
    return (n_samples // MEL_HOP + 1) // MEL_FRAMES_PER_FRAME


def chunk_mask(t: int, chunk_frames: int, device: torch.device | str | None = None) -> Tensor:
    """`[t, t]` bool, True where frame i may attend to frame j: the same or an earlier
    aligned chunk of `chunk_frames`."""
    chunk = torch.arange(t, device=device) // chunk_frames
    return chunk[None, :] <= chunk[:, None]


# ---------------------------------------------------------------------------- the front end


class _Spectrogram(nn.Module):
    """Power spectrogram with a periodic Hann window of 512 samples (`window` is in the
    weight file, as the front end's key `front_end.0.spectrogram.window`)."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("window", torch.hann_window(WIN_LENGTH, periodic=True))

    def forward(self, wave: Tensor, *, center: bool) -> Tensor:
        spec = torch.stft(
            wave, n_fft=N_FFT, hop_length=MEL_HOP, win_length=WIN_LENGTH, window=self.window,
            center=center, pad_mode="reflect", normalized=False, onesided=True,
            return_complex=True,
        )  # fmt: skip
        return spec.abs().pow(2.0)


class _MelScale(nn.Module):
    """The 257-by-64 mel filterbank (`fb`, in the weight file)."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("fb", torch.zeros(N_FFT // 2 + 1, N_MELS))

    def forward(self, power: Tensor) -> Tensor:
        return torch.matmul(power.transpose(-1, -2), self.fb).transpose(-1, -2)


class _MelSpectrogram(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.spectrogram = _Spectrogram()
        self.mel_scale = _MelScale()


class LogMel(nn.Sequential):
    """The mel front end: `forward(wave [B, S]) -> [B, 1, 64, F]` log-power mel frames,
    centred, and `frames(segment)` for the streaming path. Index 0 holds the buffers; the
    decibel step at index 1 has none."""

    def __init__(self) -> None:
        super().__init__(_MelSpectrogram(), nn.Identity())

    @property
    def mel(self) -> _MelSpectrogram:
        return self[0]

    @staticmethod
    def to_db(power: Tensor) -> Tensor:
        """`10 log10(max(x, 1e-10))`, the decibel step with no global clamp."""
        return 10.0 * torch.log10(torch.clamp(power, min=DB_AMIN))

    def forward(self, wave: Tensor) -> Tensor:
        if wave.dim() != 2:
            raise ValueError(f"wave must be [B, S], got shape {tuple(wave.shape)}")
        power = self.mel.mel_scale(self.mel.spectrogram(wave, center=True))
        return self.to_db(power).unsqueeze(1)

    def frames(self, segment: Tensor) -> Tensor:
        """Log-mel frames of an already padded 1-D `segment`, uncentred: `[64, k]` with
        frame i reading `segment[160 i : 160 i + 512]`, so `k = (len - 512) // 160 + 1`."""
        if segment.dim() != 1:
            raise ValueError(f"segment must be 1-D, got shape {tuple(segment.shape)}")
        if segment.shape[0] < N_FFT:
            raise ValueError(f"segment holds {segment.shape[0]} samples, fewer than one window")
        power = self.mel.spectrogram(segment, center=False)
        return self.to_db(self.mel.mel_scale(power))


class FrozenBatchNorm2d(nn.BatchNorm2d):
    """A BatchNorm2d that stays in eval mode: a fixed per-bin affine from its running
    statistics."""

    def train(self, mode: bool = True) -> FrozenBatchNorm2d:
        return super().train(False)


class _ToBins(nn.Module):
    def forward(self, x: Tensor) -> Tensor:  # [B, 1, 64, F] -> [B, 64, 1, F]
        return x.transpose(1, 2)


class InitBN(nn.Sequential):
    """A BatchNorm over the 64 mel bins of a `[B, 1, 64, F]` input; its keys are
    `init_bn.1.*`."""

    def __init__(self) -> None:
        super().__init__(_ToBins(), FrozenBatchNorm2d(N_MELS, momentum=0.01), _ToBins())

    @property
    def bn(self) -> FrozenBatchNorm2d:
        return self[1]

    def affine(self) -> tuple[Tensor, Tensor]:
        """`(scale, shift)`, each `[64]`: what the norm applies in eval mode."""
        bn = self.bn
        with torch.no_grad():
            scale = bn.weight / torch.sqrt(bn.running_var + bn.eps)
            shift = bn.bias - bn.running_mean * scale
        return scale.detach(), shift.detach()


# ---------------------------------------------------------------------------- the transformer


class AudioPatchEmbed(nn.Module):
    """One frame per 4 mel frames over all 64 bins: `[B, 1, 64, F] -> [B, D, 1, F // 4]`."""

    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.proj = nn.Conv2d(1, embed_dim, kernel_size=PATCH, stride=PATCH)

    def forward(self, x: Tensor) -> Tensor:
        return self.proj(x)


class Attention(nn.Module):
    """Multi-head self-attention with one `qkv` linear. `split_qkv` and `attend` are the
    two halves the stream calls apart to cache keys and values."""

    def __init__(self, dim: int, num_heads: int, qkv_bias: bool) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim {dim} is not divisible by num_heads {num_heads}")
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def split_qkv(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        return q, k, v

    def attend(self, q: Tensor, k: Tensor, v: Tensor, allowed: Tensor | None) -> Tensor:
        """`allowed` broadcasts to `[B, H, Nq, Nk]`, True where the query may attend.
        Masked scores take the dtype's minimum before the softmax, so they weigh zero."""
        attn = (q @ k.transpose(-2, -1)) * self.scale
        if allowed is not None:
            attn = attn.masked_fill(~allowed, torch.finfo(attn.dtype).min)
        attn = torch.nan_to_num(attn.softmax(dim=-1))
        B, H, Nq, hd = q.shape
        x = (attn @ v).transpose(1, 2).reshape(B, Nq, H * hd)
        return self.proj(x)

    def forward(self, x: Tensor, allowed: Tensor | None = None) -> Tensor:
        return self.attend(*self.split_qkv(x), allowed)


class Mlp(nn.Module):
    def __init__(self, dim: int, hidden: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x: Tensor) -> Tensor:
        return self.fc2(self.act(self.fc1(x)))


class Block(nn.Module):
    """Pre-norm: x + attn(norm1(x)), then x + mlp(norm2(x))."""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float, qkv_bias: bool) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=LAYER_NORM_EPS)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias)
        self.norm2 = nn.LayerNorm(dim, eps=LAYER_NORM_EPS)
        self.mlp = Mlp(dim, int(dim * mlp_ratio))

    def forward(self, x: Tensor, allowed: Tensor | None = None) -> Tensor:
        x = x + self.attn(self.norm1(x), allowed)
        return x + self.mlp(self.norm2(x))


class Listener(nn.Module):
    """The encoder (module docstring). `forward` is the batch path over whole audio,
    used by the tests; the session streams through `StaggeredStream`."""

    def __init__(
        self,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
    ) -> None:
        super().__init__()
        if depth < 1:
            raise ValueError(f"depth must be at least 1, got {depth}")
        self.embed_dim = int(embed_dim)
        self.depth = int(depth)
        self.num_heads = int(num_heads)
        self.front_end = LogMel()
        self.init_bn = InitBN()
        self.patch_embed = AudioPatchEmbed(self.embed_dim)
        self.time_pos_embed = nn.Parameter(torch.randn(1, self.embed_dim, 1, BLOCK_FRAMES) * 0.02)
        self.freq_pos_embed = nn.Parameter(torch.randn(1, self.embed_dim, 1, 1) * 0.02)
        self.blocks = nn.ModuleList(
            [Block(self.embed_dim, self.num_heads, mlp_ratio, qkv_bias) for _ in range(self.depth)]
        )
        self.norm = nn.LayerNorm(self.embed_dim, eps=LAYER_NORM_EPS)
        self.apply(self._init_weights)

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any]) -> Listener:
        return cls(
            embed_dim=int(cfg["embed_dim"]), depth=int(cfg["depth"]),
            num_heads=int(cfg["num_heads"]), mlp_ratio=float(cfg["mlp_ratio"]),
            qkv_bias=bool(cfg["qkv_bias"]),
        )  # fmt: skip

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)

    def check_layer(self, layer: int) -> int:
        layer = int(layer)
        if not 1 <= layer <= self.depth:
            raise ValueError(f"the read layer must be in [1, {self.depth}], got {layer}")
        return layer

    def embed(self, spec: Tensor) -> Tensor:
        """Patch frames without positions: `[B, 1, 64, F] -> [B, D, 1, T]`."""
        return self.patch_embed(spec)

    def add_positions(self, tokens: Tensor, start: int = 0) -> Tensor:
        """`[B, D, 1, t] -> [B, t, D]` with the positions `start .. start + t` of a window."""
        t = tokens.shape[-1]
        if start + t > BLOCK_FRAMES:
            raise ValueError(f"positions {start}..{start + t} exceed the window of {BLOCK_FRAMES}")
        x = tokens + self.time_pos_embed[:, :, :, start : start + t] + self.freq_pos_embed
        return x.flatten(2).transpose(1, 2)

    def forward(self, wave: Tensor, *, chunk_frames: int = 2, layer: int | None = None) -> Tensor:
        """`[B, S] -> [B, T, D]`: the output of `layer` (block k's output for k < depth,
        the final LayerNorm's for k == depth) at every frame, with the chunk mask, windows
        of `BLOCK_FRAMES` frames restarting the positions."""
        if wave.dim() != 2:
            raise ValueError(f"wave must be [B, S], got shape {tuple(wave.shape)}")
        layer = self.check_layer(self.depth if layer is None else layer)
        x = self.embed(self.init_bn(self.front_end(wave)))
        T = x.shape[-1]
        if T < 1:
            raise ValueError(f"{wave.shape[1]} samples is too short for one frame")
        parts = []
        for start in range(0, T, BLOCK_FRAMES):
            window = x[..., start : start + BLOCK_FRAMES]
            t = window.shape[-1]
            h = self.add_positions(window)
            allowed = chunk_mask(t, chunk_frames, h.device)[None, None]
            for b in range(layer):
                h = self.blocks[b](h, allowed)
            parts.append(self.norm(h) if layer == self.depth else h)
        return torch.cat(parts, dim=1)


# ---------------------------------------------------------------------------- streaming


class BlockStream:
    """One window of the encoder, fed in chunks (module docstring)."""

    def __init__(self, encoder: Listener, *, chunk_frames: int, layer: int) -> None:
        self.encoder = encoder
        self.layer = encoder.check_layer(layer)
        self.chunk_frames = int(chunk_frames)
        if self.chunk_frames < 1:
            raise ValueError(f"chunk_frames must be at least 1, got {chunk_frames}")
        param = encoder.time_pos_embed
        self.device, self.dtype = param.device, param.dtype
        self._scale, self._shift = encoder.init_bn.affine()
        self.reset()

    def reset(self) -> None:
        D, H = self.encoder.embed_dim, self.encoder.num_heads
        kw = {"device": self.device, "dtype": self.dtype}
        self._wave = torch.zeros(0, **kw)
        self._h = [torch.zeros(1, 0, D, **kw) for _ in range(self.layer + 1)]
        self._q = [torch.zeros(1, H, 0, D // H, **kw) for _ in range(self.layer)]
        self._k = [t.clone() for t in self._q]
        self._v = [t.clone() for t in self._q]
        self._emitted = 0
        self._finished = False

    @property
    def n_samples(self) -> int:
        return int(self._wave.shape[0])

    @property
    def frames_embedded(self) -> int:
        return int(self._h[0].shape[1])

    @property
    def frames_emitted(self) -> int:
        return self._emitted

    @property
    def finished(self) -> bool:
        return self._finished

    def push(self, chunk: Tensor) -> Tensor:
        """Add samples; return the newly final frames `[t, D]`, t possibly 0."""
        if self._finished:
            raise RuntimeError("push after flush: call reset() first")
        if chunk.dim() != 1:
            raise ValueError(f"chunk must be 1-D, got shape {tuple(chunk.shape)}")
        if self.n_samples + chunk.shape[0] > BLOCK_SAMPLES:
            raise ValueError(
                f"{self.n_samples + chunk.shape[0]} samples exceed the window of "
                f"{BLOCK_SAMPLES}; flush() and reset() first"
            )
        with torch.no_grad():
            self._wave = torch.cat([self._wave, chunk.to(self.device, self.dtype)])
            n = self.n_samples
            # Mel frame k is computable without mirroring once 160 k + 256 <= n; the left
            # mirror of frame 0 needs samples 1..256, so n > 256 for any frame.
            mel = (n - CENTER_PAD) // MEL_HOP + 1 if n > CENTER_PAD else 0
            self._embed_up_to(mel // MEL_FRAMES_PER_FRAME)
            self._advance(flush=False)
            return self._emit()

    def flush(self) -> Tensor:
        """End the window: embed the last frame with the right-edge mirror, finalise the
        last (possibly short) chunk, return it. Idempotent until `reset()`."""
        if self._finished:
            return self._emit()
        with torch.no_grad():
            n = self.n_samples
            self._embed_up_to(n_frames(n) if n > CENTER_PAD else 0)
            self._advance(flush=True)
            self._finished = True
            return self._emit()

    def _padded_slice(self, a: int, b: int) -> Tensor:
        """Samples [a, b) of the window, mirrored past either end as the centred front
        end does."""
        x, n = self._wave, self.n_samples
        if a < -CENTER_PAD or b > n + CENTER_PAD or n <= CENTER_PAD:
            raise ValueError(f"slice [{a}, {b}) is not available from {n} samples")
        parts = []
        if a < 0:
            parts.append(x[1 : 1 - a].flip(0))
        parts.append(x[max(a, 0) : min(b, n)])
        if b > n:
            parts.append(x[2 * n - 1 - b : n - 1].flip(0))
        return torch.cat(parts)

    def _embed_up_to(self, t1: int) -> None:
        t0 = self.frames_embedded
        if t1 <= t0:
            return
        if t1 > BLOCK_FRAMES:
            raise ValueError(f"{t1} frames exceed the window of {BLOCK_FRAMES}")
        f0, f1 = t0 * MEL_FRAMES_PER_FRAME, t1 * MEL_FRAMES_PER_FRAME
        segment = self._padded_slice(
            MEL_HOP * f0 - CENTER_PAD, MEL_HOP * (f1 - 1) + N_FFT - CENTER_PAD
        )
        frames = self.encoder.front_end.frames(segment)
        frames = frames * self._scale[:, None] + self._shift[:, None]
        tokens = self.encoder.embed(frames[None, None])
        self._h[0] = torch.cat([self._h[0], self.encoder.add_positions(tokens, start=t0)], dim=1)

    def _advance(self, *, flush: bool) -> None:
        C = self.chunk_frames
        for d in range(self.layer):
            block = self.encoder.blocks[d]
            src = self._h[d]
            n_src = src.shape[1]
            cached = self._k[d].shape[2]
            if n_src > cached:
                q, k, v = block.attn.split_qkv(block.norm1(src[:, cached:n_src]))
                self._q[d] = torch.cat([self._q[d], q], dim=2)
                self._k[d] = torch.cat([self._k[d], k], dim=2)
                self._v[d] = torch.cat([self._v[d], v], dim=2)
            done = self._h[d + 1].shape[1]
            target = n_src if flush else (n_src // C) * C
            if target <= done:
                continue
            i = torch.arange(done, target, device=self.device)[:, None] // C
            j = torch.arange(n_src, device=self.device)[None, :] // C
            allowed = (j <= i)[None, None]
            attended = block.attn.attend(
                self._q[d][:, :, done:target], self._k[d], self._v[d], allowed
            )
            h = src[:, done:target] + attended
            h = h + block.mlp(block.norm2(h))
            self._h[d + 1] = torch.cat([self._h[d + 1], h], dim=1)

    def _emit(self) -> Tensor:
        ready = self._h[self.layer].shape[1]
        rows = self._h[self.layer][0, self._emitted : ready]
        self._emitted = ready
        return self.encoder.norm(rows) if self.layer == self.encoder.depth else rows


class StaggeredStream:
    """Two `BlockStream`s offset by `offset_frames`, each frame read from the one further
    into its window (module docstring). `push` and `flush` return `[t, D]`."""

    def __init__(
        self,
        encoder: Listener,
        *,
        chunk_frames: int,
        layer: int,
        offset_frames: int = DEFAULT_OFFSET_FRAMES,
    ) -> None:
        if not 0 < offset_frames < BLOCK_FRAMES:
            raise ValueError(f"offset_frames must be in (0, {BLOCK_FRAMES}), got {offset_frames}")
        self.encoder = encoder
        self.offset_frames = int(offset_frames)
        kw = {"chunk_frames": chunk_frames, "layer": layer}
        self._streams = [BlockStream(encoder, **kw), BlockStream(encoder, **kw)]
        self._starts = [0, self.offset_frames * FRAME_SAMPLES]
        self.reset()

    def reset(self) -> None:
        for stream in self._streams:
            stream.reset()
        self._n = 0
        self._blocks = [0, 0]
        self._pending: dict[int, Tensor] = {}
        self._next = 0

    @property
    def frames_emitted(self) -> int:
        return self._next

    def position(self, t: int, source: int) -> int:
        """Position of frame t in `source`'s window, or -1 before that stream starts."""
        rel = t - source * self.offset_frames
        return rel % BLOCK_FRAMES if rel >= 0 else -1

    def chosen(self, t: int) -> int:
        """The stream frame t is read from: the one further into its window."""
        return 1 if self.position(t, 1) > self.position(t, 0) else 0

    def push(self, chunk: Tensor) -> Tensor:
        if chunk.dim() != 1:
            raise ValueError(f"chunk must be 1-D, got shape {tuple(chunk.shape)}")
        n0, n1 = self._n, self._n + chunk.shape[0]
        for source, start in enumerate(self._starts):
            if n1 > start:
                self._feed(source, chunk[max(0, start - n0) :])
        self._n = n1
        return self._drain()

    def flush(self) -> Tensor:
        for source, stream in enumerate(self._streams):
            before = stream.frames_emitted
            self._collect(source, stream.flush(), before)
        return self._drain()

    def _feed(self, source: int, samples: Tensor) -> None:
        stream = self._streams[source]
        while samples.shape[0]:
            room = BLOCK_SAMPLES - stream.n_samples
            piece, samples = samples[:room], samples[room:]
            before = stream.frames_emitted
            self._collect(source, stream.push(piece), before)
            if stream.n_samples == BLOCK_SAMPLES:
                before = stream.frames_emitted
                self._collect(source, stream.flush(), before)
                stream.reset()
                self._blocks[source] += 1

    def _collect(self, source: int, out: Tensor, local_first: int) -> None:
        base = source * self.offset_frames + self._blocks[source] * BLOCK_FRAMES
        for i in range(out.shape[0]):
            t = base + local_first + i
            if self.chosen(t) == source:
                self._pending[t] = out[i]

    def _drain(self) -> Tensor:
        rows = []
        while self._next in self._pending:
            rows.append(self._pending.pop(self._next))
            self._next += 1
        if rows:
            return torch.stack(rows)
        param = self.encoder.time_pos_embed
        return torch.zeros(0, self.encoder.embed_dim, device=param.device, dtype=param.dtype)


def sample_bound(t: int, chunk_frames: int) -> int:
    """The first sample the frame at index t cannot depend on, at any layer: the end of
    its chunk plus the front end's reach."""
    chunk_end = (t // chunk_frames + 1) * chunk_frames - 1
    return FRAME_SAMPLES * (chunk_end + 1) + FRAME_REACH


__all__ = [
    "BLOCK_FRAMES", "BLOCK_SAMPLES", "BlockStream", "Listener", "StaggeredStream",
    "chunk_mask", "n_frames", "sample_bound",
]  # fmt: skip
