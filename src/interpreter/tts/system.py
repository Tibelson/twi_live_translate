"""English speech output via the OS voice (macOS `say`).

Chosen as the default English backend because it needs no model download and
no setup, which keeps the first run of the interpreter to one fetch step. Swap
in a Piper English voice via TWI_EN_TTS=piper when output quality matters more
than setup cost.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np

from ..config import TtsConfig
from .base import Utterance


class SystemSynthesizer:
    language = "en"

    def __init__(self, config: TtsConfig) -> None:
        if shutil.which("say") is None:
            raise RuntimeError(
                "`say` not found (macOS only). Set TWI_EN_TTS=piper and supply an "
                "English Piper voice instead."
            )
        self._voice = config.english_voice
        # `say` takes words per minute; 175 is roughly its natural pace.
        self._rate = int(175 * config.speaking_rate)

    def speak(self, text: str) -> Utterance:
        import soundfile as sf

        started = time.monotonic()
        with tempfile.TemporaryDirectory() as workdir:
            path = Path(workdir) / "speech.aiff"
            subprocess.run(
                ["say", "-v", self._voice, "-r", str(self._rate), "-o", str(path), text],
                check=True,
                capture_output=True,
            )
            audio, sample_rate = sf.read(path, dtype="float32", always_2d=False)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        return Utterance(
            audio=np.asarray(audio, dtype=np.float32),
            sample_rate=int(sample_rate),
            elapsed=time.monotonic() - started,
        )
