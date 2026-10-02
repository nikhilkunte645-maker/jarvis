"""
jarvis/tests/test_vad.py
~~~~~~~~~~~~~~~~~~~~~~~~
Unit tests for VADSegmenter (Silero VAD v5, ONNX).

All tests use the REAL model — no mocks — so tensor shapes are validated
against the actual inference session.  silero_vad.onnx is auto-downloaded
on first run (cached in jarvis/audio/).

Key fact about Silero v5:
  * A 512-sample window WITHOUT context scores < 0.01 even for real speech.
  * The same window WITH a 64-sample context prepended can score > 0.99.
  * Pure sine waves and white noise score < 0.01 (model is speech-specific).
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from jarvis.audio.vad import VADSegmenter, _WINDOW_SAMPLES_16K, _CONTEXT_SAMPLES


# ── helpers ──────────────────────────────────────────────────────────────────

def _silence_bytes(n_samples: int) -> bytes:
    return np.zeros(n_samples, dtype=np.int16).tobytes()


def _noise_bytes(n_samples: int, amplitude: float = 0.4, seed: int = 42) -> bytes:
    rng = np.random.default_rng(seed)
    return (rng.uniform(-1, 1, n_samples) * amplitude * 32767).astype(np.int16).tobytes()


def _speech_like_bytes(n_samples: int, sample_rate: int = 16000) -> bytes:
    """
    Synthesise a formant-filtered noise burst that mimics voiced speech energy.
    We prime the model with a full second of this signal before asserting so
    the LSTM state can accumulate context.
    """
    rng = np.random.default_rng(7)
    noise = rng.standard_normal(n_samples).astype(np.float32)

    # Simple low-pass + resonance to add formant-like energy
    from scipy.signal import butter, lfilter  # type: ignore
    b, a = butter(4, 3500 / (sample_rate / 2), btype="low")
    filtered = lfilter(b, a, noise).astype(np.float32)

    # Normalise to ~70 % peak
    filtered /= np.abs(filtered).max() + 1e-9
    filtered *= 0.7
    return (filtered * 32767).astype(np.int16).tobytes()


# ── fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def vad() -> VADSegmenter:
    return VADSegmenter(sample_rate=16000, threshold=0.3)


# ── tests: speech probability ─────────────────────────────────────────────────

class TestVADProbability:
    def test_silence_prob_is_low(self, vad: VADSegmenter):
        """Extended silence should score < 0.2 in every window."""
        vad.reset_state()
        for _ in range(20):
            window = np.zeros(_WINDOW_SAMPLES_16K, dtype=np.float32)
            prob = vad._infer_window(window)
            assert prob < 0.2, f"Silence scored {prob:.4f}, expected < 0.2"

    def test_speech_like_signal_prob_exceeds_threshold(self, vad: VADSegmenter):
        """
        A formant-filtered noise burst (speech-like) should score > 0.5 in
        at least one window after the LSTM has been primed for ~0.5 s.
        Uses scipy for filtering — if scipy is absent, skip gracefully.
        """
        pytest.importorskip("scipy")
        vad.reset_state()
        # Feed ~1.5 s of speech-like audio to prime the LSTM context
        n_prime = 16000 * 2  # 2 seconds
        prime_bytes = _speech_like_bytes(n_prime)
        vad.is_speech(prime_bytes)  # primes state; result not checked

        # Now check individual window scores
        max_prob = 0.0
        window_bytes = _speech_like_bytes(_WINDOW_SAMPLES_16K, seed=99)
        for _ in range(20):
            window = np.frombuffer(window_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            prob = vad._infer_window(window)
            max_prob = max(max_prob, prob)

        assert max_prob > 0.5, (
            f"Speech-like signal max prob was {max_prob:.4f}, expected > 0.5 "
            "after LSTM priming. Check model version or signal generation."
        )

    def test_white_noise_does_not_trigger(self, vad: VADSegmenter):
        """Stationary white noise should NOT exceed threshold 0.3 (not speech)."""
        vad.reset_state()
        any_detected = any(
            vad.is_speech(_noise_bytes(_WINDOW_SAMPLES_16K, seed=i))
            for i in range(20)
        )
        assert not any_detected, "White noise triggered VAD — threshold may be too low"

    def test_pure_tone_prob_is_low(self, vad: VADSegmenter):
        """A 440 Hz sine tone should score < 0.15 (not speech)."""
        vad.reset_state()
        t = np.arange(_WINDOW_SAMPLES_16K) / 16000
        tone = (np.sin(2 * np.pi * 440 * t) * 0.6).astype(np.float32)
        for _ in range(10):
            prob = vad._infer_window(tone)
        assert prob < 0.15, f"440 Hz tone scored {prob:.4f}, expected < 0.15"


# ── tests: chunk-size robustness ─────────────────────────────────────────────

class TestVADChunkHandling:
    def test_exact_512_samples_does_not_crash(self, vad: VADSegmenter):
        vad.reset_state()
        result = vad.is_speech(_silence_bytes(_WINDOW_SAMPLES_16K))
        assert isinstance(result, bool)

    def test_1024_samples_does_not_crash(self, vad: VADSegmenter):
        """Double window (1024 samples) must be handled without raising."""
        vad.reset_state()
        result = vad.is_speech(_silence_bytes(1024))
        assert isinstance(result, bool)

    def test_odd_chunk_does_not_crash(self, vad: VADSegmenter):
        """700-sample chunk: 512 consumed + 188 leftover, no crash."""
        vad.reset_state()
        result = vad.is_speech(_silence_bytes(700))
        assert isinstance(result, bool)

    def test_leftover_carries_to_next_call(self, vad: VADSegmenter):
        """512 + 100 = 612 → 512 consumed, 100 leftover; next 412 → one more window."""
        vad.reset_state()
        vad.is_speech(_silence_bytes(612))   # 512 consumed, 100 leftover
        result = vad.is_speech(_silence_bytes(412))   # 100 + 412 = 512 → one window
        assert isinstance(result, bool)
        assert vad._leftover.size == 0       # nothing left after exact window

    def test_small_chunks_accumulate(self, vad: VADSegmenter):
        """100-sample chunks accumulate until 512 are ready."""
        vad.reset_state()
        results = [vad.is_speech(_silence_bytes(100)) for _ in range(6)]
        # After 600 samples: 1 window consumed (False), 88 leftover
        assert isinstance(results[-1], bool)
        assert vad._leftover.size == 88

    def test_empty_chunk_returns_false(self, vad: VADSegmenter):
        vad.reset_state()
        assert vad.is_speech(b"") is False

    def test_reset_clears_state_and_context(self, vad: VADSegmenter):
        vad.is_speech(_silence_bytes(100))  # create some leftover
        vad.reset_state()
        assert vad._leftover.size == 0
        assert np.all(vad._state == 0)
        assert np.all(vad._context == 0)


# ── tests: config & construction ─────────────────────────────────────────────

class TestVADConfig:
    def test_invalid_sample_rate_raises(self):
        with pytest.raises(ValueError, match="8000 or 16000"):
            VADSegmenter(sample_rate=44100)

    def test_window_size_16k(self, vad: VADSegmenter):
        assert vad.window_samples == 512

    def test_window_size_8k(self):
        v = VADSegmenter(sample_rate=8000, threshold=0.3)
        assert v.window_samples == 256

    def test_context_constant_is_64(self):
        assert _CONTEXT_SAMPLES == 64

    def test_infer_window_updates_context(self, vad: VADSegmenter):
        """After _infer_window(), context must equal the last 64 samples of the window."""
        vad.reset_state()
        window = np.arange(_WINDOW_SAMPLES_16K, dtype=np.float32) / 32768.0
        vad._infer_window(window)
        np.testing.assert_array_equal(vad._context, window[-64:])
