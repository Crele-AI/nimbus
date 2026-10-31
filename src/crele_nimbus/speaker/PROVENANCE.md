# Provenance of the CAM++ code in this directory

`DTDNN.py` and `layers.py` are copies of two files from 3D-Speaker
(https://github.com/modelscope/3D-Speaker, Apache License 2.0), branch `main`, commit
`065629c313eaf1a01c65c640c46d77e61e9607b4` (2025-12-08), path `speakerlab/models/campplus/`.

| File here | Upstream path | sha256 of the upstream file |
|---|---|---|
| `DTDNN.py` | `speakerlab/models/campplus/DTDNN.py` | `52272359361ef9f03a8b4198f50139301ce712833e777396940a5b437177cd34` |
| `layers.py` | `speakerlab/models/campplus/layers.py` | `b3d28ad94d14b30fcd20c96d1d83516d4e44de733959460b358e90d97476adb8` |

`layers.py` is byte for byte the upstream file. `DTDNN.py` differs in one line: its import
of the layers is relative (`from .layers import ...`) instead of absolute, so the copy runs
inside this package without the `speakerlab` package. Both files keep their upstream
headers and licence notice.

**What it is.** CAM++, a speaker-verification network: a 2-D convolutional front, a densely
connected TDNN with context-aware masking, statistics pooling and a linear layer. Built as
`CAMPPlus(feat_dim=80, embedding_size=512)` (every other argument at the class default,
as 3D-Speaker's `infer_sv.py` builds it for the VoxCeleb model), it takes 80-dim log mel
filterbanks `[B, T, 80]` and gives one 512-dim embedding per row.

**The weights.** `speaker.safetensors` beside the model's config is the VoxCeleb checkpoint
of that model (ModelScope `iic/speech_campplus_sv_en_voxceleb_16k`, Apache License 2.0),
unchanged. The features it expects are reproduced in `embed.py`: Kaldi's 80-bin log
filterbank (25 ms frames every 10 ms, pre-emphasis 0.97, a Povey window, no dither) with
the mean over time removed, as 3D-Speaker's `FBank(80, sample_rate=16000, mean_nor=True)`
computes them.

**Policy.** The two copied files are not edited here and are excluded from the linter. A
change belongs upstream, then in a fresh copy with this file's commit and hashes updated.
