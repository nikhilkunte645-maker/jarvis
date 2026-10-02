"""
jarvis.ui_hud.tts
~~~~~~~~~~~~~~~~~
Text-to-Speech voice feedback engine.

Supports three offline-capable backends:
* ``edge_tts``  – Microsoft Edge TTS (requires internet for synthesis but
                  very natural; can be used offline with pre-cached audio)
* ``pyttsx3``   – 100% offline, uses OS speech engines
* ``piper``     – fully local neural TTS (best quality offline)

Usage::

    from jarvis.ui_hud.tts import TTSEngine

    tts = TTSEngine()
    tts.speak("Command executed successfully.")
"""

from __future__ import annotations

import asyncio
import io
import tempfile
from pathlib import Path
from typing import Optional

from loguru import logger

from jarvis.core.config_loader import cfg


class TTSEngine:
    """Converts text to speech using the configured backend.

    Attributes:
        backend: ``"edge_tts"`` | ``"pyttsx3"`` | ``"piper"`` | ``"none"``.
        voice: TTS voice identifier.
    """

    def __init__(
        self,
        backend: Optional[str] = None,
        voice: Optional[str] = None,
    ) -> None:
        tts_engine = cfg.get("tts.tts_engine", default="offline")
        if not backend:
            backend = "edge_tts" if tts_engine == "edge" else "pyttsx3"
            
        self.backend: str = backend
        self.voice: str = voice or cfg.get("tts.voice", default="en-US-GuyNeural")
        self.rate: str = cfg.get("tts.rate", default="+0%")
        self.volume: str = cfg.get("tts.volume", default="+0%")
        self.piper_model: str = cfg.get("tts.piper_model", default="")

        self._engine = None  # lazy init for pyttsx3
        logger.info(f"TTSEngine configured  backend={self.backend}  voice={self.voice}")

    # ── Public API ──────────────────────────────────────────────────────────

    def speak(self, text: str) -> None:
        """Synthesise and play *text* via the configured backend.

        Non-blocking for edge_tts (runs async internally); blocking for
        pyttsx3 and piper.

        Args:
            text: The text to speak aloud.

        Returns:
            None
        """
        if not text or self.backend == "none":
            return

        logger.debug(f"TTS speak: '{text[:80]}'")
        try:
            if self.backend == "edge_tts":
                self._speak_edge(text)
            elif self.backend == "pyttsx3":
                self._speak_pyttsx3(text)
            elif self.backend == "piper":
                self._speak_piper(text)
        except Exception as exc:
            logger.warning(f"TTS error ({self.backend}): {exc}")

    # ── Backend implementations ─────────────────────────────────────────────

    def _speak_edge(self, text: str) -> None:
        """Synthesise with edge-tts and play via playsound.

        Args:
            text: Text to synthesise.

        Returns:
            None
        """
        try:
            import edge_tts  # type: ignore
            import playsound  # type: ignore
        except ImportError:
            logger.warning("edge_tts or playsound not installed; falling back to pyttsx3")
            self.backend = "pyttsx3"
            self._speak_pyttsx3(text)
            return

        async def _run() -> None:
            communicate = edge_tts.Communicate(
                text, self.voice, rate=self.rate, volume=self.volume
            )
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
                tmp_path = f.name
            await communicate.save(tmp_path)
            playsound.playsound(tmp_path)
            Path(tmp_path).unlink(missing_ok=True)

        asyncio.run(_run())

    def _speak_pyttsx3(self, text: str) -> None:
        """Synthesise with pyttsx3 (100% offline).

        Args:
            text: Text to synthesise.

        Returns:
            None
        """
        try:
            import pyttsx3  # type: ignore
        except ImportError as exc:
            raise ImportError("pyttsx3 not installed. Run: pip install pyttsx3") from exc

        if self._engine is None:
            self._engine = pyttsx3.init()
            voices = self._engine.getProperty("voices")
            # Try to find an English voice
            for v in voices:
                if "en" in v.id.lower() or "english" in v.name.lower():
                    self._engine.setProperty("voice", v.id)
                    break
            self._engine.setProperty("rate", 175)

        self._engine.say(text)
        self._engine.runAndWait()

    def _speak_piper(self, text: str) -> None:
        """Synthesise with Piper TTS (offline neural, best quality).

        Args:
            text: Text to synthesise.

        Returns:
            None
        """
        import subprocess

        if not self.piper_model:
            raise ValueError("piper_model path not set in config/tts.piper_model")

        model_path = Path(self.piper_model).expanduser()
        if not model_path.exists():
            raise FileNotFoundError(f"Piper model not found: {model_path}")

        proc = subprocess.Popen(
            ["piper", "--model", str(model_path), "--output-raw"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
        )
        raw_audio, _ = proc.communicate(input=text.encode())

        # Play raw PCM via aplay (Linux) or sox / afplay (macOS)
        import platform

        if platform.system() == "Linux":
            subprocess.run(
                ["aplay", "-r", "22050", "-f", "S16_LE", "-t", "raw"],
                input=raw_audio,
                check=True,
            )
        elif platform.system() == "Darwin":
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                import wave

                with wave.open(f.name, "wb") as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(22050)
                    wf.writeframes(raw_audio)
            subprocess.run(["afplay", f.name])
            Path(f.name).unlink(missing_ok=True)
