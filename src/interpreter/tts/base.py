"""Synthesizer protocol."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass
class Utterance:
    audio: np.ndarray  # mono float32
    sample_rate: int
    elapsed: float = 0.0


class Synthesizer(Protocol):
    language: str

    def speak(self, text: str) -> Utterance:
        """Synthesise one utterance."""
        ...
