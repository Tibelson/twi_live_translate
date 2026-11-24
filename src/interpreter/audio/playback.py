"""Blocking playback of synthesised speech."""

from __future__ import annotations

import numpy as np

from ..config import AudioConfig


class Speaker:
    def __init__(self, config: AudioConfig) -> None:
        self.config = config

    def play(self, audio: np.ndarray, sample_rate: int) -> None:
        """Play one utterance and return when it has finished.

        Blocking is deliberate: in face-to-face interpreting, overlapping the
        translated voice with the next incoming utterance makes the
        conversation unintelligible for both parties.
        """
        if audio.size == 0:
            return
        import sounddevice as sd

        sd.play(
            np.asarray(audio, dtype=np.float32),
            samplerate=sample_rate,
            device=self.config.output_device,
        )
        sd.wait()
