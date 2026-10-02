"""
jarvis/audio/vad.py
~~~~~~~~~~~~~~~~~~~
Voice Activity Detection using Silero VAD v5 (ONNX).

Silero VAD v5 requires a 64-sample context prepended to each 512-sample
window, making the effective input shape ``(1, 576)`` at 16 kHz.
After each inference the last 64 samples of the window become the context
for the next call.  The LSTM state ``(2, 1, 128)`` is also carried forward.

``VADSegmenter.is_speech()`` accepts **any** chunk size by:
  1. Prepending buffered leftover from the previous call.
  2. Consuming as many complete 512-sample windows as possible.
  3. Storing the remaining tail as leftover for the next call.

``reset_state()`` zeroes both the LSTM state and the context buffer.

Usage::

    vad = VADSegmenter(sample_rate=16000, threshold=0.3)
    for chunk_bytes in mic_stream:          # any length, int16 PCM
        if vad.is_speech(chunk_bytes):
            ...                             # speech detected in this chunk
"""

from __future__ import annotations

import os
import urllib.request

import numpy as np
import onnxruntime as ort
from loguru import logger

_WINDOW_SAMPLES_16K: int = 512   # required window size at 16 kHz
_WINDOW_SAMPLES_8K:  int = 256   # required window size at  8 kHz
_CONTEXT_SAMPLES:    int = 64    # v5 context prepended to every window


class VADSegmenter:
    """Wraps Silero VAD v5 (ONNX) and exposes a simple ``is_speech()`` API.

    Attributes:
        sample_rate:    Audio sample rate in Hz (8000 or 16000).
        threshold:      Speech-probability threshold (0.0–1.0).
        window_samples: Samples per ONNX inference call (512 @ 16 k).
    """

    MODEL_URL  = (
        "https://github.com/snakers4/silero-vad/raw/master/"
        "src/silero_vad/data/silero_vad.onnx"
    )
    MODEL_PATH = os.path.join(os.path.dirname(__file__), "silero_vad.onnx")

    def __init__(self, sample_rate: int = 16000, threshold: float = 0.3) -> None:
        if sample_rate not in (8000, 16000):
            raise ValueError(
                f"Silero VAD supports 8000 or 16000 Hz, got {sample_rate}"
            )
        self.sample_rate    = sample_rate
        self.threshold      = threshold
        self.window_samples = (
            _WINDOW_SAMPLES_16K if sample_rate == 16000 else _WINDOW_SAMPLES_8K
        )
        self._sr_tensor = np.array(sample_rate, dtype=np.int64)

        self._ensure_model_exists()

        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        self._sess = ort.InferenceSession(
            self.MODEL_PATH,
            providers=["CPUExecutionProvider"],
            sess_options=opts,
        )

        self.reset_state()
        logger.info(
            f"VADSegmenter ready  sr={sample_rate}  threshold={threshold}  "
            f"window={self.window_samples} samples ({self.window_samples / sample_rate * 1000:.0f} ms)  "
            f"context={_CONTEXT_SAMPLES} samples (v5)"
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def is_speech(self, chunk: bytes) -> bool:
        """Return True if any 512-sample window in *chunk* contains speech.

        Accepts any chunk length; leftovers are carried to the next call.

        Args:
            chunk: Raw int16 PCM bytes at ``self.sample_rate``.

        Returns:
            True if speech probability exceeds ``self.threshold``.
        """
        if not chunk:
            return False

        # int16 bytes → float32 [-1.0, 1.0]
        new_samples = (
            np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32768.0
        )

        # Prepend leftover from the previous call
        audio = (
            np.concatenate([self._leftover, new_samples])
            if self._leftover.size
            else new_samples
        )

        detected = False
        offset   = 0
        while offset + self.window_samples <= len(audio):
            window   = audio[offset : offset + self.window_samples]
            prob     = self._infer_window(window)
            if prob > self.threshold:
                detected = True
            offset += self.window_samples

        # Save unconsumed tail for the next call
        self._leftover = (
            audio[offset:].copy() if offset < len(audio)
            else np.empty(0, dtype=np.float32)
        )
        return detected

    def reset_state(self) -> None:
        """Reset LSTM hidden state, context buffer, and leftover tail."""
        self._state    = np.zeros((2, 1, 128), dtype=np.float32)
        self._context  = np.zeros(_CONTEXT_SAMPLES, dtype=np.float32)
        self._leftover = np.empty(0, dtype=np.float32)

    # ── Private ───────────────────────────────────────────────────────────────

    def _infer_window(self, window: np.ndarray) -> float:
        """Run one 512-sample window through the model (with 64-sample context).

        Silero VAD v5 requires the input to be ``[context(64) + window(512)]``
        = 576 samples total, shaped ``(1, 576)``.

        Args:
            window: float32 array of exactly ``self.window_samples`` samples.

        Returns:
            Speech probability as a Python float.
        """
        # v5: prepend 64-sample context → (1, 576)
        inp = np.concatenate([self._context, window])[np.newaxis, :].astype(np.float32)

        out, state_n = self._sess.run(
            None,
            {
                "input": inp,
                "state": self._state,
                "sr":    self._sr_tensor,
            },
        )
        # Carry forward state and context
        self._state   = state_n                 # shape (2, 1, 128)
        self._context = window[-_CONTEXT_SAMPLES:].copy()   # last 64 samples
        return float(out[0, 0])

    def _ensure_model_exists(self) -> None:
        if not os.path.exists(self.MODEL_PATH):
            logger.info("Downloading Silero VAD ONNX model…")
            urllib.request.urlretrieve(self.MODEL_URL, self.MODEL_PATH)
            logger.info("Silero VAD model downloaded.")
