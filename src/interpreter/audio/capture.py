"""Continuous microphone capture as a queue of fixed-size float32 frames.

The point of streaming capture here is that nothing is ever written to a file:
the VAD consumes frames as they arrive and emits an utterance the moment the
speaker stops, so recognition starts while the speaker is still in the room.
"""

from __future__ import annotations

import queue
import sys
from collections.abc import Iterator
from types import TracebackType

import numpy as np

from ..config import AudioConfig


class MicrophoneStream:
    """Yield mono float32 frames of exactly `frame_ms` milliseconds."""

    def __init__(self, config: AudioConfig) -> None:
        self.config = config
        self.frame_samples = int(config.sample_rate * config.frame_ms / 1000)
        self._queue: queue.Queue[np.ndarray] = queue.Queue()
        self._stream = None
        self._residual = np.zeros(0, dtype=np.float32)

    def __enter__(self) -> MicrophoneStream:
        import sounddevice as sd

        def callback(indata, frames, time_info, status) -> None:  # noqa: ANN001
            del frames, time_info
            if status:
                print(f"[audio] {status}", file=sys.stderr)
            self._queue.put(indata[:, 0].copy())

        self._stream = sd.InputStream(
            samplerate=self.config.sample_rate,
            channels=1,
            dtype="float32",
            blocksize=self.frame_samples,
            device=self.config.input_device,
            callback=callback,
        )
        self._stream.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def frames(self) -> Iterator[np.ndarray]:
        """Re-chunk whatever the device hands us into exact VAD-sized frames.

        PortAudio honours `blocksize` loosely on some devices, and webrtcvad
        rejects anything that is not exactly 10/20/30 ms, so normalise here.
        """
        while True:
            block = self._queue.get()
            if block is None:
                return
            self._residual = np.concatenate([self._residual, block])
            while len(self._residual) >= self.frame_samples:
                yield self._residual[: self.frame_samples]
                self._residual = self._residual[self.frame_samples :]

    def stop(self) -> None:
        self._queue.put(None)  # type: ignore[arg-type]
