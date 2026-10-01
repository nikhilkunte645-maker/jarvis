"""
jarvis/audio/stt.py
~~~~~~~~~~~~~~~~~~~
Speech-to-Text using faster-whisper on CPU (int8).
"""

import os
import tempfile
import wave
from loguru import logger
from faster_whisper import WhisperModel

class STTEngine:
    """
    Transcribes audio using faster-whisper.
    """

    def __init__(self, model_size: str = "tiny.en", device: str = "cpu", compute_type: str = "int8"):
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self._model = None
        logger.info(f"Initialized STTEngine (model={model_size}, device={device})")

    def _ensure_model_loaded(self) -> None:
        """Lazy load the Whisper model to save memory until actually needed."""
        if self._model is None:
            logger.info("Loading faster-whisper model...")
            self._model = WhisperModel(
                self.model_size,
                device=self.device,
                compute_type=self.compute_type,
                download_root=os.path.join(os.path.dirname(__file__), "..", "..", "models")
            )
            logger.info("STT model loaded successfully.")

    def transcribe(self, audio_bytes: bytes, sample_rate: int = 16000) -> str:
        """
        Takes raw 16kHz int16 PCM audio bytes, saves to a temp WAV, and transcribes.
        """
        self._ensure_model_loaded()
        
        # Write to temporary file for faster-whisper
        # (faster-whisper can also take numpy arrays, but file avoids shape wrangling)
        fd, temp_path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        
        try:
            with wave.open(temp_path, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2) # 16-bit
                wf.setframerate(sample_rate)
                wf.writeframes(audio_bytes)

            segments, info = self._model.transcribe(temp_path, beam_size=1) # type: ignore
            
            text = "".join(segment.text for segment in segments).strip()
            return text
            
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
