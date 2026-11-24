"""Wires capture -> VAD -> ASR -> repair+translate -> TTS -> playback."""

from __future__ import annotations

import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .asr.base import Recognizer, Transcript
from .audio import MicrophoneStream, SegmentChunker, Speaker
from .config import Config
from .translate import Interpretation, Translator
from .tts.base import Synthesizer

# What to say when an utterance came through too garbled to interpret.
REPEAT_PROMPT = {
    "tw": "Mepa wo kyɛw, ka bio.",
    "en": "Sorry, could you say that again?",
}


@dataclass
class Turn:
    source: str
    target: str
    transcript: Transcript
    audio_seconds: float
    # None in ASR-only mode, where no translation is attempted.
    interpretation: Interpretation | None = None
    tts_seconds: float = 0.0
    # Peak sample level of the utterance, for setting microphone gain.
    peak: float = 0.0

    @property
    def total_seconds(self) -> float:
        interpret = self.interpretation.elapsed if self.interpretation else 0.0
        return self.transcript.elapsed + interpret + self.tts_seconds

    @property
    def realtime_factor(self) -> float:
        """Recognition seconds per second of audio.

        Below 1.0 means the recogniser finishes faster than the speech
        arrives, which is the condition for keeping up in a live conversation.
        """
        if self.audio_seconds <= 0:
            return 0.0
        return self.transcript.elapsed / self.audio_seconds


class Interpreter:
    """One direction at a time; `both` flips on demand."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.speaker = Speaker(config.audio)
        self._translator: Translator | None = None
        self._recognizers: dict[str, Recognizer] = {}
        self._synthesizers: dict[str, Synthesizer] = {}
        self._direction_lock = threading.Lock()
        self.source, self.target = self._parse_direction(config.direction)
        self._turn_index = 0

    # -- lazily built backends, so a one-direction run loads one ASR model,
    #    and ASR-only mode builds no API client and no synthesiser at all ----

    @property
    def translator(self) -> Translator:
        if self._translator is None:
            self._translator = Translator(self.config.translate)
        return self._translator

    def recognizer(self, language: str) -> Recognizer:
        if language not in self._recognizers:
            if language == "tw":
                from .asr.griot import GriotRecognizer

                self._recognizers[language] = GriotRecognizer(self.config.asr)
            else:
                from .asr.whisper import load_english_recognizer

                self._recognizers[language] = load_english_recognizer(self.config.asr)
        return self._recognizers[language]

    def synthesizer(self, language: str) -> Synthesizer:
        if language not in self._synthesizers:
            if language == "tw" or self.config.tts.english_backend == "piper":
                from .tts.piper import PiperSynthesizer

                self._synthesizers[language] = PiperSynthesizer(self.config.tts, language)
            else:
                from .tts.system import SystemSynthesizer

                self._synthesizers[language] = SystemSynthesizer(self.config.tts)
        return self._synthesizers[language]

    # -- direction handling -------------------------------------------------

    @staticmethod
    def _parse_direction(direction: str) -> tuple[str, str]:
        if direction in {"tw-en", "both"}:
            return "tw", "en"
        if direction == "en-tw":
            return "en", "tw"
        raise ValueError(f"unknown direction {direction!r}")

    def flip(self) -> tuple[str, str]:
        with self._direction_lock:
            self.source, self.target = self.target, self.source
            return self.source, self.target

    def current_direction(self) -> tuple[str, str]:
        with self._direction_lock:
            return self.source, self.target

    # -- the actual work ----------------------------------------------------

    def handle(
        self, audio: np.ndarray, *, speak: bool = True, translate: bool = True
    ) -> Turn:
        source, target = self.current_direction()
        turn = Turn(
            source=source,
            target=target,
            transcript=self.recognizer(source).transcribe(audio),
            audio_seconds=len(audio) / self.config.audio.sample_rate,
            peak=float(np.max(np.abs(audio))) if audio.size else 0.0,
        )

        if not translate:
            # ASR-only: no API call, no synthesis, no playback.
            if self.config.save_audio is not None:
                self._save(audio, turn.transcript, None)
            return turn

        turn.interpretation = self.translator.interpret(
            turn.transcript.text, source, target
        )
        if self.config.save_audio is not None:
            self._save(audio, turn.transcript, turn.interpretation)

        if speak:
            # On a low-confidence turn, prompting the speaker to repeat is more
            # useful than reading out a guess the listener cannot trust.
            if turn.interpretation.needs_repeat:
                utterance = self.synthesizer(source).speak(REPEAT_PROMPT[source])
            else:
                utterance = self.synthesizer(target).speak(turn.interpretation.translation)
            turn.tts_seconds = utterance.elapsed
            self.speaker.play(utterance.audio, utterance.sample_rate)

        return turn

    def _save(
        self,
        audio: np.ndarray,
        transcript: Transcript,
        interpretation: Interpretation | None,
    ) -> None:
        """Persist utterance audio plus its transcript.

        This is the seed corpus for fine-tuning: real conversational audio with
        a reviewed reconstruction beside the raw recogniser output. In ASR-only
        mode only the raw output exists, which is still the half that matters
        for measuring and improving recognition.
        """
        import json

        import soundfile as sf

        directory = Path(self.config.save_audio)
        directory.mkdir(parents=True, exist_ok=True)
        self._turn_index += 1
        stem = f"{int(time.time())}_{self._turn_index:04d}"
        sf.write(directory / f"{stem}.wav", audio, self.config.audio.sample_rate)
        record: dict[str, object] = {
            "raw_asr": transcript.text,
            "language": transcript.language,
        }
        if interpretation is not None:
            record.update(
                {
                    "repaired_source": interpretation.repaired_source,
                    "translation": interpretation.translation,
                    "confidence": interpretation.confidence,
                    "notes": interpretation.notes,
                }
            )
        (directory / f"{stem}.json").write_text(
            json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def run(self, on_turn, on_segment_start=None, *, translate: bool = True) -> None:
        """Listen until interrupted, calling `on_turn(turn)` per utterance."""
        chunker = SegmentChunker(self.config.audio)
        with MicrophoneStream(self.config.audio) as microphone:
            for segment in chunker.segments(microphone.frames()):
                if on_segment_start is not None:
                    on_segment_start(len(segment) / self.config.audio.sample_rate)
                try:
                    on_turn(self.handle(segment, translate=translate))
                except Exception as exc:  # one bad turn must not end the session
                    print(f"[pipeline] {exc.__class__.__name__}: {exc}", file=sys.stderr)


def read_wav(path: Path, sample_rate: int) -> np.ndarray:
    """Load a file as 16 kHz mono float32, for testing without a microphone."""
    import soundfile as sf
    from scipy.signal import resample_poly

    audio, file_rate = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if file_rate != sample_rate:
        from math import gcd

        divisor = gcd(int(file_rate), sample_rate)
        audio = resample_poly(audio, sample_rate // divisor, int(file_rate) // divisor)
    return np.asarray(audio, dtype=np.float32)
