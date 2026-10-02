"""
jarvis/audio/pipeline.py
~~~~~~~~~~~~~~~~~~~~~~~~
High-level audio pipeline combining Mic, VAD, and STT.
"""

from typing import Callable, Optional
from loguru import logger

from jarvis.core.config_loader import cfg
from .microphone import MicrophoneStream
from .vad import VADSegmenter
from .stt import STTEngine

class AudioPipeline:
    """
    Listens to the microphone, segments speech using VAD, 
    and transcribes the utterance using STT.
    """

    def __init__(self, 
                 stt_model: str = "tiny.en", 
                 on_transcription: Optional[Callable[[str], None]] = None):
        self.sample_rate = cfg.get("audio.sample_rate", default=16000)
        self.chunk_size = cfg.get("audio.chunk_size", default=512)
        device_index = cfg.get("audio.device_index", default=None)
        vad_threshold = cfg.get("audio.vad.threshold", default=0.3)  # lower = more sensitive
        silence_ms = cfg.get("audio.vad.silence_timeout_ms", default=800)
        # Convert silence_timeout_ms → number of chunks
        self._max_silence_chunks = max(1, int(silence_ms / 1000 * self.sample_rate / self.chunk_size))

        self.mic = MicrophoneStream(
            sample_rate=self.sample_rate,
            chunk_size=self.chunk_size,
            device_index=device_index,
        )
        self.vad = VADSegmenter(sample_rate=self.sample_rate, threshold=vad_threshold)
        stt_device  = cfg.get("audio.stt.device", default="cpu")
        stt_compute = cfg.get("audio.stt.compute_type", default="int8")
        self.stt = STTEngine(model_size=stt_model, device=stt_device, compute_type=stt_compute)
        
        self.on_transcription = on_transcription
        self._is_running = False
        logger.info(
            f"AudioPipeline ready  sr={self.sample_rate}  chunk={self.chunk_size}  "
            f"vad_threshold={vad_threshold}  silence_chunks={self._max_silence_chunks}  "
            f"device_index={device_index}"
        )

    def run(self) -> None:
        """Runs the continuous listening loop. Blocks the current thread."""
        logger.info("Starting Audio Pipeline...")
        self._is_running = True
        
        speech_buffer = bytearray()
        silence_chunks = 0
        is_speaking = False
        _dbg_counter = 0
        
        try:
            for chunk in self.mic.listen():
                if not self._is_running:
                    break
                    
                is_speech_now = self.vad.is_speech(chunk)
                _dbg_counter += 1
                if _dbg_counter % 50 == 0:  # log every ~1.6s
                    logger.debug(f"VAD heartbeat  speaking={is_speaking}  buffer={len(speech_buffer)}B")
                
                if is_speech_now:
                    if not is_speaking:
                        is_speaking = True
                        logger.info("🎤 Speech started")
                    speech_buffer.extend(chunk)
                    silence_chunks = 0
                else:
                    if is_speaking:
                        silence_chunks += 1
                        speech_buffer.extend(chunk)  # trailing silence for context
                        
                        if silence_chunks > self._max_silence_chunks:
                            # End of utterance
                            is_speaking = False
                            logger.info(f"🔇 Speech ended  buffer={len(speech_buffer)}B  silence_chunks={silence_chunks}")
                            self._process_utterance(bytes(speech_buffer))
                            speech_buffer = bytearray()
                            silence_chunks = 0
                            self.vad.reset_state()
                            
        except KeyboardInterrupt:
            logger.info("Audio Pipeline stopped via KeyboardInterrupt.")
        finally:
            self.stop()

    def _process_utterance(self, audio_data: bytes) -> None:
        """Transcribes the utterance and triggers the callback."""
        # Only process if it's long enough (e.g. > 0.5s)
        if len(audio_data) < self.sample_rate * 2 * 0.5: # 2 bytes per sample
            return
            
        logger.debug(f"Transcribing utterance of {len(audio_data)} bytes...")
        text = self.stt.transcribe(audio_data, sample_rate=self.sample_rate)
        
        if text:
            logger.info(f"[STT] User: {text}")
            if self.on_transcription:
                self.on_transcription(text)

    def stop(self) -> None:
        """Stops the pipeline."""
        self._is_running = False
        self.mic.stop()
