"""Where the weights come from: a local folder, or a Hugging Face repository id.

A model folder holds `config.json`, `listener.safetensors` (the streaming encoder),
`head.safetensors` (the decision head) and `speaker.safetensors` (the voice model).
`resolve` turns a repository id into such a folder through `huggingface_hub`, fetching
the four files into the Hugging Face cache on first use; `load_folder` builds the three
networks from one and loads their weights strictly.
"""

from __future__ import annotations

import os
from pathlib import Path

from crele_nimbus.config import CONFIG_FILE, WEIGHT_FILES, NimbusConfig
from crele_nimbus.head import DecisionHead
from crele_nimbus.listener import Listener
from crele_nimbus.speaker import VoiceModel


class WeightsError(RuntimeError):
    """The weights could not be found or do not match the code."""


def resolve(
    path_or_id: str | os.PathLike[str],
    *,
    revision: str | None = None,
    token: str | bool | None = None,
) -> Path:
    """A local folder as it is; anything else is a Hugging Face repository id, downloaded
    (or found in the cache) and returned as its local snapshot folder."""
    p = Path(path_or_id).expanduser()
    if p.is_dir():
        return p
    text = str(path_or_id)
    if os.sep in text and not text.count("/") == 1 or text.startswith("."):
        raise WeightsError(f"no folder at {p}")
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:  # pragma: no cover
        raise WeightsError("huggingface_hub is needed to load weights by repository id") from exc
    try:
        folder = snapshot_download(
            repo_id=text, revision=revision, token=token,
            allow_patterns=[CONFIG_FILE, *WEIGHT_FILES],
        )  # fmt: skip
    except Exception as exc:
        raise WeightsError(
            f"could not fetch {text!r} from Hugging Face ({exc}). Give a local folder with "
            f"{CONFIG_FILE} and the weights instead."
        ) from exc
    return Path(folder)


def load_folder(
    folder: str | os.PathLike[str],
) -> tuple[NimbusConfig, Listener, DecisionHead, VoiceModel]:
    """The config and the three networks of a model folder, weights loaded, on the CPU."""
    from safetensors.torch import load_file

    folder = Path(folder)
    missing = [f for f in (CONFIG_FILE, *WEIGHT_FILES) if not (folder / f).is_file()]
    if missing:
        raise WeightsError(f"{folder} lacks {missing}")
    config = NimbusConfig.load(folder)
    listener = Listener.from_config(config.listener)
    head = DecisionHead.from_config(config.head, config.calibration)
    voice = VoiceModel(embedding_size=int(config.head["spk_dim"]))
    try:
        listener.load_state_dict(load_file(folder / "listener.safetensors"), strict=True)
        head.load_weights(load_file(folder / "head.safetensors"))
        voice.model.load_state_dict(load_file(folder / "speaker.safetensors"), strict=True)
    except RuntimeError as exc:
        raise WeightsError(f"{folder}: the weights do not match the model ({exc})") from exc
    return config, listener.eval(), head.eval(), voice.eval()
