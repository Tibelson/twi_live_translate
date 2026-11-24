"""Recognizer protocol shared by the Twi and English front ends."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass
class Transcript:
    text: str
    language: str
    # Seconds of wall clock spent in recognition, for the latency readout.
    elapsed: float = 0.0


class Recognizer(Protocol):
    language: str

    def transcribe(self, audio: np.ndarray) -> Transcript:
        """Transcribe one 16 kHz mono float32 utterance."""
        ...
