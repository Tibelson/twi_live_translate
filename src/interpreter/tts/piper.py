"""Twi speech output via a Piper voice (Kasanoma).

Quality caveat worth knowing before you judge the output: the Kasanoma project
states its voices are trained for a limited time on donated GPU capacity, and
publishes training checkpoints precisely so they can be fine-tuned further.
Expect intelligible but rough prosody, and expect tone -- which is lexically
meaningful in Twi -- to be the weakest part. Fine-tuning from the published
checkpoints is the path to improving it.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from ..config import TtsConfig
from .base import Utterance


class PiperSynthesizer:
    def __init__(self, config: TtsConfig, language: str = "tw") -> None:
        from piper import PiperVoice, SynthesisConfig

        voice_path = Path(config.twi_voice)
        if not voice_path.is_file():
            raise FileNotFoundError(
                f"{voice_path} not found. Run `python scripts/fetch_models.py` first."
            )
        self.language = language
        self._voice = PiperVoice.load(voice_path)
        # Piper's length_scale stretches duration, so it is the reciprocal of
        # a speaking rate.
        self._syn_config = SynthesisConfig(
            length_scale=1.0 / config.speaking_rate if config.speaking_rate else None
        )

    def speak(self, text: str) -> Utterance:
        started = time.monotonic()
        chunks = list(self._voice.synthesize(text, syn_config=self._syn_config))
        if not chunks:
            return Utterance(np.zeros(0, dtype=np.float32), 22_050, 0.0)
        audio = np.concatenate([chunk.audio_float_array for chunk in chunks])
        return Utterance(
            audio=audio.astype(np.float32),
            sample_rate=chunks[0].sample_rate,
            elapsed=time.monotonic() - started,
        )
