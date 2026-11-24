"""Asante Twi recognition via GhanaNLP / Qlerqly Griot Edge (Conformer CTC).

We load the vendor's own `onnx_runtime.py` out of the model directory rather
than reimplementing log-mel extraction. The model was trained against that
exact extractor (400-sample Hann window, 160 hop, 80 Slaney mels, log floor
1e-5); a plausible-looking reimplementation that differs in any of those
degrades accuracy without raising an error, which is the worst failure mode
available to us.

Accuracy to expect: the vendor reports CER 0.158 / WER 0.374 fully causal and
CER 0.150 / WER 0.349 with 13 frames of lookahead, on Asante Twi. Roughly one
word in three comes out wrong, which is why the translate stage is also a
repair stage.
"""

from __future__ import annotations

import importlib.util
import sys
import time
import unicodedata
from pathlib import Path
from typing import Any

import numpy as np

from ..config import AsrConfig
from .base import Transcript

_MODULE_NAME = "_griot_onnx_runtime"


def _load_vendor_runtime(model_dir: Path) -> Any:
    """Import `onnx_runtime.py` from the downloaded model directory."""
    module_path = model_dir / "onnx_runtime.py"
    if not module_path.is_file():
        raise FileNotFoundError(
            f"{module_path} not found. Run `python scripts/fetch_models.py` first."
        )
    if _MODULE_NAME in sys.modules:
        return sys.modules[_MODULE_NAME]
    spec = importlib.util.spec_from_file_location(_MODULE_NAME, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[_MODULE_NAME] = module
    # The vendor module imports `ctc_decode` by bare name for KenLM beam
    # search, so its directory has to be importable.
    sys.path.insert(0, str(model_dir))
    spec.loader.exec_module(module)
    return module


class GriotRecognizer:
    language = "tw"

    def __init__(self, config: AsrConfig) -> None:
        self.config = config
        runtime_module = _load_vendor_runtime(config.griot_dir)
        self._runtime = runtime_module.OnnxRuntime.load(
            config.griot_dir, threads=config.num_threads
        )
        self.lookahead_frames = (
            config.lookahead_frames if self._runtime.supports_lookahead else 0
        )
        if config.lookahead_frames and not self._runtime.supports_lookahead:
            print(
                "[asr] this export has no lookahead input; running fully causal",
                file=sys.stderr,
            )

    def transcribe(self, audio: np.ndarray) -> Transcript:
        started = time.monotonic()
        text = self._runtime.transcribe(
            np.asarray(audio, dtype=np.float32),
            lookahead_frames=self.lookahead_frames,
            # One utterance at a time, so no windowing is needed. Passing 0
            # keeps the whole segment in a single forward pass.
            chunk_seconds=0.0,
        )
        return Transcript(
            text=normalize(text), language=self.language, elapsed=time.monotonic() - started
        )


def normalize(text: str) -> str:
    """Match the vendor's evaluation normalisation: NFC, lowercase.

    Tone marks are not produced by this model at all, so there is nothing to
    strip; we keep combining marks that are part of Twi orthography (ɛ, ɔ).
    """
    return unicodedata.normalize("NFC", text).strip().lower()
