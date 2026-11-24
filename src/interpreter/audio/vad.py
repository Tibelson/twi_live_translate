"""Turn a frame stream into utterance-sized segments using WebRTC VAD.

Segmentation is the part of a cascaded interpreter that most affects perceived
quality. Cut too eagerly and the ASR sees half a clause, which the repair stage
cannot fix because the missing words were never spoken into the microphone. Cut
too late and the listener waits. The onset/hangover pair below is the knob.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Iterator

import numpy as np
import webrtcvad

from ..config import AudioConfig


def _to_pcm16(frame: np.ndarray) -> bytes:
    clipped = np.clip(frame, -1.0, 1.0)
    return (clipped * 32767.0).astype(np.int16).tobytes()


class SegmentChunker:
    """Emit one float32 array per detected utterance."""

    def __init__(self, config: AudioConfig) -> None:
        self.config = config
        self._vad = webrtcvad.Vad(config.aggressiveness)
        self.frame_samples = int(config.sample_rate * config.frame_ms / 1000)
        self._onset_frames = max(1, config.speech_onset_ms // config.frame_ms)
        self._hangover_frames = max(1, config.silence_hangover_ms // config.frame_ms)
        self._min_frames = max(1, config.min_segment_ms // config.frame_ms)
        self._max_frames = max(1, config.max_segment_ms // config.frame_ms)

    def segments(self, frames: Iterable[np.ndarray]) -> Iterator[np.ndarray]:
        # Keep a short rolling window so the segment includes the speech that
        # triggered the onset, not just everything after the trigger fired.
        preroll: deque[np.ndarray] = deque(maxlen=self._onset_frames)
        voiced: list[np.ndarray] = []
        in_speech = False
        onset_run = 0
        silence_run = 0

        for frame in frames:
            if len(frame) != self.frame_samples:
                continue
            is_speech = self._vad.is_speech(_to_pcm16(frame), self.config.sample_rate)

            if not in_speech:
                preroll.append(frame)
                onset_run = onset_run + 1 if is_speech else 0
                if onset_run >= self._onset_frames:
                    in_speech = True
                    voiced = list(preroll)
                    preroll.clear()
                    silence_run = 0
                continue

            voiced.append(frame)
            silence_run = 0 if is_speech else silence_run + 1

            ended = silence_run >= self._hangover_frames
            overlong = len(voiced) >= self._max_frames
            if not (ended or overlong):
                continue

            # Drop the trailing silence we only collected to detect the end.
            payload = voiced[: len(voiced) - silence_run] if ended else voiced
            if len(payload) >= self._min_frames:
                yield np.concatenate(payload)
            in_speech = False
            onset_run = 0
            silence_run = 0
            voiced = []
            # An over-long segment is almost always someone still talking, so
            # carry the tail forward as preroll instead of discarding it.
            if overlong:
                preroll.extend(payload[-self._onset_frames :])
