"""The voice model: CAM++ (vendored from 3D-Speaker, see PROVENANCE.md) with its Kaldi
filterbank front end. `VoiceModel.embed` turns a clip of speech into a 512-dim voice
vector; the session embeds the last 2 s of the stream every 0.4 s and the enrollment clip
once."""

from .embed import EMBEDDING_SIZE, MIN_SAMPLES, VoiceModel, kaldi_mel_banks

__all__ = ["EMBEDDING_SIZE", "MIN_SAMPLES", "VoiceModel", "kaldi_mel_banks"]
