"""The model's configuration (`config.json` beside the weights) and the event settings.

`config.json` holds what building and running the model needs and nothing else: the time
grid, the sizes of the three networks, the text encoder's repository and pinned revision,
the calibration of the yes probability, the default event settings and the two voice
signals' texts. `NimbusConfig.load` reads and checks it.

`Settings` is the four event settings a developer may set per question, named as
voice-activity detectors name them: an event starts when the probability rises above
`threshold` and stays there for `start_secs`; it stops when the probability falls to
`stop_threshold` or below and stays there for `stop_secs`. The defaults come from the
config and apply to every question, the voice signals included; a question may override
any of the four. `threshold` never moves `stop_threshold`, and a `stop_threshold` at or
above `threshold` is refused.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

SAMPLE_RATE = 16000
"""The only sample rate the model takes."""
FRAME_SAMPLES = 640
"""Samples per answer frame: 40 ms."""
FRAME_HZ = SAMPLE_RATE // FRAME_SAMPLES
"""Answer frames per second: 25."""
CONFIG_FILE = "config.json"
WEIGHT_FILES = ("listener.safetensors", "head.safetensors", "speaker.safetensors")
SETTING_NAMES = ("threshold", "start_secs", "stop_threshold", "stop_secs")


class ConfigError(ValueError):
    """`config.json` is missing something the model needs, or holds a bad value."""


def min_frames(seconds: float, frame_hz: int = FRAME_HZ) -> int:
    """The smallest whole number of frames lasting at least `seconds`, and at least 1.
    The small slack keeps 0.3 s at 25 Hz (7.500000000000001) at 8 frames, not 9."""
    if seconds < 0:
        raise ValueError(f"a duration must be non-negative, got {seconds}")
    return max(1, math.ceil(seconds * frame_hz - 1e-9))


@dataclass(frozen=True)
class Settings:
    """One question's event settings (module docstring)."""

    threshold: float
    start_secs: float
    stop_threshold: float
    stop_secs: float

    def check(self, where: str = "settings") -> Settings:
        for name in SETTING_NAMES:
            v = getattr(self, name)
            if isinstance(v, bool) or not isinstance(v, int | float) or not math.isfinite(v):
                raise ValueError(f"{where}.{name}: not a finite number: {v!r}")
        if not 0.0 <= self.threshold <= 1.0 or not 0.0 <= self.stop_threshold <= 1.0:
            raise ValueError(f"{where}: threshold and stop_threshold must be between 0 and 1")
        if self.stop_threshold >= self.threshold:
            raise ValueError(
                f"{where}: stop_threshold {self.stop_threshold} must be under threshold "
                f"{self.threshold}"
            )
        if self.start_secs < 0 or self.stop_secs < 0:
            raise ValueError(f"{where}: start_secs and stop_secs must be non-negative")
        return self

    def override(self, where: str = "settings", **changes: float | None) -> Settings:
        """A copy with the given settings replaced; `None` leaves one as it is. An unknown
        name, a bad value or a stop threshold at or over the threshold raises `ValueError`."""
        unknown = sorted(set(changes) - set(SETTING_NAMES))
        if unknown:
            raise ValueError(f"{where}: unknown settings {unknown}; they are {SETTING_NAMES}")
        given = {
            k: float(v) for k, v in changes.items() if v is not None and not isinstance(v, bool)
        }
        bad = [k for k, v in changes.items() if v is not None and isinstance(v, bool)]
        if bad:
            raise ValueError(f"{where}: {bad[0]} is not a number")
        return replace(self, **given).check(where)

    def as_dict(self) -> dict[str, float]:
        return {k: float(getattr(self, k)) for k in SETTING_NAMES}

    @classmethod
    def from_mapping(cls, m: Mapping[str, Any], where: str = "settings") -> Settings:
        missing = [k for k in SETTING_NAMES if k not in m]
        if missing:
            raise ValueError(f"{where}: missing {missing}")
        return cls(**{k: float(m[k]) for k in SETTING_NAMES}).check(where)


@dataclass(frozen=True)
class NimbusConfig:
    """What `config.json` holds, checked. Nested parts stay as plain mappings; the modules
    that build the networks read the keys they need."""

    model: str
    sample_rate: int
    frame_ms: int
    chunk_frames: int
    listener: dict[str, Any]
    head: dict[str, Any]
    voice: dict[str, float]
    text_encoder: dict[str, Any]
    calibration: float
    defaults: Settings
    cant_tell_below: float
    voice_signals: dict[str, str]

    @property
    def chunk_samples(self) -> int:
        """Samples the session consumes per step: 1,280 at two frames per chunk (80 ms)."""
        return self.chunk_frames * FRAME_SAMPLES

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any], where: str = CONFIG_FILE) -> NimbusConfig:
        def need(key: str, kind: type | tuple[type, ...]) -> Any:
            if key not in data:
                raise ConfigError(f"{where}: missing {key!r}")
            v = data[key]
            if not isinstance(v, kind) or isinstance(v, bool):
                raise ConfigError(f"{where}: {key} has the wrong type ({type(v).__name__})")
            return v

        sample_rate = int(need("sample_rate", int))
        if sample_rate != SAMPLE_RATE:
            raise ConfigError(f"{where}: sample_rate {sample_rate} is not {SAMPLE_RATE}")
        frame_ms = int(need("frame_ms", int))
        if frame_ms * SAMPLE_RATE != FRAME_SAMPLES * 1000:
            raise ConfigError(
                f"{where}: frame_ms {frame_ms} does not match {FRAME_SAMPLES} samples"
            )
        chunk_frames = int(need("chunk_frames", int))
        if chunk_frames < 1:
            raise ConfigError(f"{where}: chunk_frames must be at least 1")
        defaults_raw = need("defaults", Mapping)
        try:
            defaults = Settings.from_mapping(defaults_raw, f"{where}: defaults")
        except ValueError as exc:
            raise ConfigError(str(exc)) from exc
        cant_tell_below = float(defaults_raw.get("cant_tell_below", 0.5))
        if not 0.0 <= cant_tell_below <= 1.0:
            raise ConfigError(f"{where}: defaults.cant_tell_below must be between 0 and 1")
        calibration = float(need("calibration", int | float))
        if not math.isfinite(calibration) or calibration <= 0:
            raise ConfigError(f"{where}: calibration must be a positive number")
        signals = dict(need("voice_signals", Mapping))
        if set(signals) != {"user", "other"} or not all(
            isinstance(v, str) for v in signals.values()
        ):
            raise ConfigError(
                f"{where}: voice_signals must name 'user' and 'other' with their texts"
            )
        voice = {k: float(v) for k, v in need("voice", Mapping).items()}
        for key in ("window_s", "hop_s"):
            if key not in voice:
                raise ConfigError(f"{where}: voice.{key} is missing")
        text = dict(need("text_encoder", Mapping))
        for key in ("repo", "revision", "instruction", "dim", "max_length"):
            if key not in text:
                raise ConfigError(f"{where}: text_encoder.{key} is missing")
        return cls(
            model=str(data.get("model", "")),
            sample_rate=sample_rate,
            frame_ms=frame_ms,
            chunk_frames=chunk_frames,
            listener=dict(need("listener", Mapping)),
            head=dict(need("head", Mapping)),
            voice=voice,
            text_encoder=text,
            calibration=calibration,
            defaults=defaults,
            cant_tell_below=cant_tell_below,
            voice_signals={"user": signals["user"], "other": signals["other"]},
        )

    @classmethod
    def load(cls, folder: str | Path) -> NimbusConfig:
        path = Path(folder) / CONFIG_FILE
        if not path.is_file():
            raise ConfigError(f"no {CONFIG_FILE} in {Path(folder)}")
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ConfigError(f"{path}: not JSON ({exc})") from exc
        if not isinstance(data, Mapping):
            raise ConfigError(f"{path}: expected a JSON object")
        return cls.from_mapping(data, str(path))
