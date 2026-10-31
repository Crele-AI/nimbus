"""The streaming encoder: the stream equals the batch forward, on tiny random weights."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from conftest import synth_audio, tiny_listener
from crele_nimbus.listener import (
    BLOCK_FRAMES,
    BLOCK_SAMPLES,
    BlockStream,
    StaggeredStream,
    n_frames,
    sample_bound,
)

FRAME = 640


def _collect(stream, wave: torch.Tensor, chunk: int) -> tuple[torch.Tensor, int]:
    parts = [stream.push(wave[s : s + chunk]) for s in range(0, wave.shape[0], chunk)]
    before = stream.frames_emitted
    parts.append(stream.flush())
    return torch.cat(parts), before


def _batch(model, wave: torch.Tensor, chunk_frames: int, layer: int) -> torch.Tensor:
    with torch.no_grad():
        return model(wave[None], chunk_frames=chunk_frames, layer=layer)[0]


@pytest.mark.parametrize("layer", [2, 4])
@pytest.mark.parametrize("chunk_frames", [2, 4])
@pytest.mark.parametrize("chunk", [FRAME, 7 * FRAME, 25 * FRAME, 1000])
def test_block_stream_equals_the_batch_forward(chunk, chunk_frames, layer):
    model = tiny_listener(0)
    wave = torch.from_numpy(synth_audio(1, 10.08))
    assert wave.shape[0] == BLOCK_SAMPLES
    expected = _batch(model, wave, chunk_frames, layer)
    stream = BlockStream(model, chunk_frames=chunk_frames, layer=layer)
    got, before = _collect(stream, wave, chunk)
    assert stream.frames_emitted == BLOCK_FRAMES
    # Before the flush the stream lacks the window's last frame (its last mel frame reads
    # past the window), so it lacks that frame's whole chunk.
    assert before == BLOCK_FRAMES - chunk_frames
    assert got.shape == (BLOCK_FRAMES, 32)
    torch.testing.assert_close(got, expected, atol=1e-4, rtol=0)


def test_a_partial_window_flushes_to_the_batch_forward_of_its_samples():
    model = tiny_listener(0)
    wave = torch.from_numpy(synth_audio(2, 3.3))
    stream = BlockStream(model, chunk_frames=2, layer=4)
    got, _ = _collect(stream, wave, 2000)
    assert got.shape[0] == n_frames(wave.shape[0])
    torch.testing.assert_close(got, _batch(model, wave, 2, 4), atol=1e-4, rtol=0)


def test_chunks_appear_whole_exactly_at_the_sample_bound():
    model = tiny_listener(0)
    wave = torch.from_numpy(synth_audio(3, 2.0))
    stream = BlockStream(model, chunk_frames=2, layer=4)
    emitted = 0
    for n in range(1, wave.shape[0] + 1, 97):
        out = stream.push(wave[emitted_samples(stream) : n])
        emitted += out.shape[0]
        # Frames 2k and 2k+1 are final once sample_bound(2k+1) samples have arrived.
        expected = max(0, sum(2 for k in range(0, 60, 2) if sample_bound(k + 1, 2) <= n))
        assert emitted == expected


def emitted_samples(stream) -> int:
    return stream.n_samples


def test_the_output_depends_only_on_samples_before_the_bound():
    model = tiny_listener(0)
    wave = torch.from_numpy(synth_audio(4, 2.0))
    base = _batch(model, wave, 2, 4)
    for t in (0, 3, 10, 21):
        bound = sample_bound(t, 2)
        other = wave.clone()
        other[bound:] = torch.from_numpy(synth_audio(99, 2.0))[bound:]
        changed = _batch(model, other, 2, 4)
        torch.testing.assert_close(changed[t], base[t], atol=1e-5, rtol=0)
        # ... and does depend on a sample inside the last mel frame before it (the
        # sample right at the edge sits under the window's near-zero tail).
        other = wave.clone()
        other[bound - 300] += 0.5
        assert not torch.allclose(_batch(model, other, 2, 4)[t], base[t], atol=1e-5)


def test_block_stream_refuses_overflow_and_pushes_after_flush():
    model = tiny_listener(0)
    stream = BlockStream(model, chunk_frames=2, layer=4)
    with pytest.raises(ValueError):
        stream.push(torch.zeros(BLOCK_SAMPLES + 1))
    stream.push(torch.zeros(1000))
    stream.flush()
    with pytest.raises(RuntimeError):
        stream.push(torch.zeros(10))
    stream.reset()
    assert stream.push(torch.zeros(100)).shape == (0, 32)
    with pytest.raises(ValueError):
        BlockStream(model, chunk_frames=2, layer=5)


def test_staggered_stream_over_25_seconds():
    model = tiny_listener(0)
    wave = torch.from_numpy(synth_audio(5, 25.0))
    offset = 126
    stream = StaggeredStream(model, chunk_frames=2, layer=4, offset_frames=offset)
    got, _ = _collect(stream, wave, 16000)
    T = got.shape[0]
    assert T == n_frames(wave.shape[0]) == stream.frames_emitted
    # Each frame equals the batch forward of the window it was read from, which from the
    # offset on always has more than `offset` frames of audio behind it.
    expected_by_block: dict[int, torch.Tensor] = {}
    for t in range(T):
        source = stream.chosen(t)
        rel = t - source * offset
        block, position = divmod(rel, BLOCK_FRAMES)
        start = source * offset * FRAME + block * BLOCK_SAMPLES
        if t >= offset:
            assert position + 1 > offset
        if start not in expected_by_block:
            expected_by_block[start] = _batch(model, wave[start : start + BLOCK_SAMPLES], 2, 4)
        torch.testing.assert_close(
            got[t], expected_by_block[start][position], atol=1e-4, rtol=0, msg=f"frame {t}"
        )
    assert set(expected_by_block) == {0, 80640, 161280, 241920}


def test_staggered_stream_emits_in_order_and_flushes_the_tail():
    model = tiny_listener(0)
    wave = torch.from_numpy(synth_audio(6, 12.0))
    stream = StaggeredStream(model, chunk_frames=2, layer=4)
    emitted = 0
    for start in range(0, wave.shape[0], 3333):
        out = stream.push(wave[start : start + 3333])
        emitted += out.shape[0]
        assert stream.frames_emitted == emitted
    tail = stream.flush()
    assert tail.shape[0] > 0
    assert emitted + tail.shape[0] == n_frames(wave.shape[0])
    with pytest.raises(ValueError):
        StaggeredStream(model, chunk_frames=2, layer=4, offset_frames=BLOCK_FRAMES)


def test_front_end_matches_torchaudio():
    """The torch-only front end against torchaudio's, bitwise, when torchaudio is around."""
    torchaudio = pytest.importorskip("torchaudio")
    from torchaudio import transforms

    ref = torch.nn.Sequential(
        transforms.MelSpectrogram(
            f_min=0, sample_rate=16000, win_length=512, center=True, n_fft=512, f_max=8000,
            hop_length=160, n_mels=64,
        ),
        transforms.AmplitudeToDB(top_db=None),
    )  # fmt: skip
    model = tiny_listener(0)
    with torch.no_grad():
        model.front_end.mel.mel_scale.fb.copy_(ref[0].mel_scale.fb)
    wave = torch.from_numpy(synth_audio(7, 1.0))[None]
    with torch.no_grad():
        theirs = ref(wave).unsqueeze(1)
        ours = model.front_end(wave)
    assert torch.equal(ours, theirs)
    assert torch.equal(model.front_end.mel.spectrogram.window, ref[0].spectrogram.window)
    # The uncentred streaming frames equal the centred frames on the mirrored signal.
    padded = torch.nn.functional.pad(wave, (256, 256), mode="reflect")[0]
    frames = model.front_end.frames(padded[160 * 3 : 160 * 9 + 512])
    assert torch.equal(frames, theirs[0, 0, :, 3:10])
    assert torchaudio is not None


def test_n_frames():
    assert [n_frames(n) for n in (0, 479, 480, 16000, 161280, 161281)] == [0, 0, 1, 25, 252, 252]
    with pytest.raises(ValueError):
        n_frames(-1)
    assert np.isfinite(sample_bound(0, 2))
