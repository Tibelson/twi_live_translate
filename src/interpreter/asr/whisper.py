"""English recognition for the English -> Twi direction.

Griot's vocabulary covers Ghanaian English, so it can stand in for this when
faster-whisper is not installed, but it is trained for Ghanaian languages and
will be weaker on general English than a dedicated model.
"""

from __future__ import annotations

import time

import numpy as np

from ..config import AsrConfig
from .base import Transcript


class WhisperRecognizer:
    language = "en"

    def __init__(self, config: AsrConfig) -> None:
        from faster_whisper import WhisperModel

        self._model = WhisperModel(
            config.whisper_model, device="cpu", compute_type=config.whisper_compute
        )

    def transcribe(self, audio: np.ndarray) -> Transcript:
        started = time.monotonic()
        segments, _ = self._model.transcribe(
            np.asarray(audio, dtype=np.float32),
            language="en",
            beam_size=1,  # greedy: this is a latency-sensitive path
            vad_filter=False,  # already segmented upstream
        )
        text = " ".join(segment.text.strip() for segment in segments).strip()
        return Transcript(text=text, language="en", elapsed=time.monotonic() - started)


def load_english_recognizer(config: AsrConfig):
    """Prefer faster-whisper; fall back to Griot with a warning."""
    try:
        return WhisperRecognizer(config)
    except ImportError:
        import sys

        from .griot import GriotRecognizer

        print(
            "[asr] faster-whisper not installed (`pip install '.[en-asr]'`);\n"
            "      using Griot for English, which is trained for Ghanaian languages.",
            file=sys.stderr,
        )
        recognizer = GriotRecognizer(config)
        recognizer.language = "en"
        return recognizer
