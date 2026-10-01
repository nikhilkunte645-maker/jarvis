"""
jarvis/audio/pipeline.py
~~~~~~~~~~~~~~~~~~~~~~~~
High-level audio pipeline combining Mic, VAD, and STT.
"""

import time
from typing import Callable, Optional
from loguru import logger

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
        self.sample_rate = 16000
        self.chunk_size = 512
        
        self.mic = MicrophoneStream(sample_rate=self.sample_rate, chunk_size=self.chunk_size)
        self.vad = VADSegmenter(sample_rate=self.sample_rate)
        self.stt = STTEngine(model_size=stt_model)
        
        self.on_transcription = on_transcription
        self._is_running = False

    def run(self) -> None:
        """Runs the continuous listening loop. Blocks the current thread."""
        logger.info("Starting Audio Pipeline...")
        self._is_running = True
        
        speech_buffer = bytearray()
        silence_chunks = 0
        max_silence_chunks = 30 # roughly ~1 second of silence at 512 samples/chunk
        is_speaking = False
        
        try:
            for chunk in self.mic.listen():
                if not self._is_running:
                    break
                    
                is_speech_now = self.vad.is_speech(chunk)
                
                if is_speech_now:
                    if not is_speaking:
                        is_speaking = True
                        logger.debug("Speech detected...")
                    speech_buffer.extend(chunk)
                    silence_chunks = 0
                else:
                    if is_speaking:
                        silence_chunks += 1
                        speech_buffer.extend(chunk) # add trailing silence for context
                        
                        if silence_chunks > max_silence_chunks:
                            # End of utterance
                            is_speaking = False
                            self._process_utterance(bytes(speech_buffer))
                            speech_buffer = bytearray()
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
