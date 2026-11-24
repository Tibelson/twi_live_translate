"""Runtime configuration, resolved from environment then CLI overrides."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent.parent
SAMPLE_RATE = 16_000


@dataclass
class AudioConfig:
    sample_rate: int = SAMPLE_RATE
    # webrtcvad accepts 10/20/30 ms frames only.
    frame_ms: int = 20
    # VAD aggressiveness 0-3; 2 is a reasonable default for conversational rooms.
    aggressiveness: int = 2
    # Speech must persist this long before a segment opens, which rejects coughs
    # and door slams without adding latency to real speech.
    speech_onset_ms: int = 160
    # Silence that closes a segment. Twi speakers pause mid-sentence, so too
    # short a value chops utterances in half and starves the repair stage of
    # context; too long adds dead air to every turn.
    silence_hangover_ms: int = 700
    min_segment_ms: int = 400
    max_segment_ms: int = 15_000
    input_device: int | None = None
    output_device: int | None = None


@dataclass
class AsrConfig:
    griot_dir: Path = field(
        default_factory=lambda: ROOT / os.getenv("TWI_GRIOT_DIR", "models/griot-edge")
    )
    # 0 = fully causal (lowest latency). 13 frames of lookahead measured
    # WER 0.349 vs 0.374 causal on the vendor's internal test set.
    lookahead_frames: int = int(os.getenv("TWI_LOOKAHEAD_FRAMES", "13"))
    num_threads: int = int(os.getenv("TWI_ASR_THREADS", "0"))
    # English ASR for the English->Twi direction.
    whisper_model: str = os.getenv("TWI_WHISPER_MODEL", "small.en")
    whisper_compute: str = os.getenv("TWI_WHISPER_COMPUTE", "int8")


@dataclass
class TranslateConfig:
    model: str = os.getenv("TWI_MODEL", "claude-opus-5")
    effort: str = os.getenv("TWI_EFFORT", "low")
    max_tokens: int = 1024
    # Thinking is on by default on Claude Opus 5. It costs latency the
    # interpreter can feel, so `fast` turns it off; see translate/llm.py.
    fast: bool = os.getenv("TWI_FAST", "").lower() in {"1", "true", "yes"}
    # Turns of prior dialogue handed to the repair stage as context.
    context_turns: int = 6
    glossary_path: Path | None = None


@dataclass
class TtsConfig:
    twi_voice: Path = field(
        default_factory=lambda: ROOT
        / os.getenv("TWI_PIPER_VOICE", "models/kasanoma-twi/kasanoma-twi.onnx")
    )
    # macOS `say` needs no setup and is adequate for English output.
    english_backend: str = os.getenv("TWI_EN_TTS", "system")
    english_voice: str = os.getenv("TWI_EN_VOICE", "Samantha")
    speaking_rate: float = float(os.getenv("TWI_TTS_RATE", "1.0"))


@dataclass
class Config:
    audio: AudioConfig = field(default_factory=AudioConfig)
    asr: AsrConfig = field(default_factory=AsrConfig)
    translate: TranslateConfig = field(default_factory=TranslateConfig)
    tts: TtsConfig = field(default_factory=TtsConfig)
    # "tw-en", "en-tw", or "both" (language detected per utterance).
    direction: str = "tw-en"
    save_audio: Path | None = None
