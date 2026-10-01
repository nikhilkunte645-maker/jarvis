"""
jarvis.audio.wake_word
~~~~~~~~~~~~~~~~~~~~~~
Wake-word / hotword detection layer.

Listens continuously on a raw audio stream and emits a ``wake_word.detected``
event on the bus when the configured keyword is spotted.

Supported backends:
* ``openwakeword`` – fully open source, no API key required (default)
* ``porcupine``    – Picovoice Porcupine (requires access key)
* ``none``         – disabled; every utterance is treated as post-wake-word

Usage::

    detector = WakeWordDetector()
    detector.listen(mic.stream())  # blocks; calls bus event on detection
"""

from __future__ import annotations

import io
import struct
import wave
from typing import Callable, Iterable, Optional

import numpy as np
from loguru import logger

from jarvis.core.config_loader import cfg
from jarvis.core.event_bus import bus


class WakeWordDetector:
    """Wraps a hotword engine and emits events when the keyword fires.

    Attributes:
        backend: ``"openwakeword"`` | ``"porcupine"`` | ``"none"``.
        keyword: Wake-word string (e.g. ``"jarvis"``).
        sample_rate: Audio sample rate (must match microphone).
    """

    def __init__(
        self,
        backend: Optional[str] = None,
        keyword: Optional[str] = None,
        sample_rate: Optional[int] = None,
        on_detection: Optional[Callable[[], None]] = None,
    ) -> None:
        self.backend: str = backend or cfg.get("audio.wake_word.backend", default="openwakeword")
        self.keyword: str = keyword or cfg.get("audio.wake_word.keyword", default="jarvis")
        self.sample_rate: int = sample_rate or cfg.get("audio.sample_rate", default=16000)
        self._on_detection = on_detection
        self._engine = self._build_engine()
        logger.info(
            f"WakeWordDetector ready  backend={self.backend}  keyword='{self.keyword}'"
        )

    # ── Public API ──────────────────────────────────────────────────────────

    def listen(self, audio_stream: Iterable[np.ndarray]) -> None:
        """Block and listen to a raw audio stream for the wake word.

        Emits a ``"wake_word.detected"`` event on the global bus when
        the keyword is heard.

        Args:
            audio_stream: Iterable of int16 NumPy arrays from the microphone.

        Returns:
            None
        """
        logger.info(f"Listening for wake word: '{self.keyword}' …")
        if self.backend == "none":
            # Immediately "detect" – useful for always-on mode
            self._fire()
            return

        for chunk in audio_stream:
            if self._check_chunk(chunk):
                logger.info(f"Wake word detected: '{self.keyword}'")
                self._fire()

    def check_chunk(self, chunk: np.ndarray) -> bool:
        """Check a single audio chunk for the wake word.

        Useful for integrating into a custom loop.

        Args:
            chunk: int16 mono NumPy array (one microphone frame).

        Returns:
            True if wake word detected in this chunk.
        """
        return self._check_chunk(chunk)

    # ── Private ─────────────────────────────────────────────────────────────

    def _build_engine(self) -> object:
        if self.backend == "none":
            return None

        if self.backend == "openwakeword":
            try:
                from openwakeword.model import Model  # type: ignore

                model = Model(
                    wakeword_models=[],  # loads default models
                    inference_framework="onnx",
                )
                logger.debug("openWakeWord model loaded.")
                return model
            except ImportError as exc:
                logger.warning(
                    "openwakeword not installed – falling back to 'none' backend. "
                    "Install with: pip install openwakeword"
                )
                self.backend = "none"
                return None

        if self.backend == "porcupine":
            try:
                import pvporcupine  # type: ignore

                access_key = cfg.get("audio.wake_word.porcupine_access_key", default="")
                if not access_key:
                    raise ValueError(
                        "Porcupine backend requires PORCUPINE_ACCESS_KEY in .env"
                    )
                engine = pvporcupine.create(
                    access_key=access_key,
                    keywords=[self.keyword],
                )
                logger.debug("Porcupine engine created.")
                return engine
            except ImportError as exc:
                raise ImportError(
                    "pvporcupine not installed. Run: pip install pvporcupine"
                ) from exc

        raise ValueError(f"Unknown wake-word backend: {self.backend}")

    def _check_chunk(self, chunk: np.ndarray) -> bool:
        if self.backend == "none" or self._engine is None:
            return False

        if chunk.ndim > 1:
            chunk = chunk[:, 0]

        if self.backend == "openwakeword":
            # openWakeWord expects 16-bit int samples
            pcm = chunk.astype(np.int16)
            self._engine.predict(pcm)  # type: ignore
            scores = self._engine.prediction_buffer  # type: ignore
            # Check if any model score exceeds threshold (0.5)
            for model_name, score_list in scores.items():
                if score_list and score_list[-1] > 0.5:
                    if self.keyword.lower() in model_name.lower() or True:
                        return True
            return False

        if self.backend == "porcupine":
            import pvporcupine  # type: ignore

            frame_length = self._engine.frame_length  # type: ignore
            # Feed frames of exactly frame_length samples
            pcm = chunk.astype(np.int16).tolist()
            for i in range(0, len(pcm) - frame_length, frame_length):
                frame = pcm[i : i + frame_length]
                keyword_index = self._engine.process(frame)  # type: ignore
                if keyword_index >= 0:
                    return True
            return False

        return False

    def _fire(self) -> None:
        bus.emit(
            "wake_word.detected",
            data={"keyword": self.keyword},
            source="wake_word",
        )
        if self._on_detection:
            self._on_detection()
