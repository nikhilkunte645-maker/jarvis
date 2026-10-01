"""
jarvis/audio/microphone.py
~~~~~~~~~~~~~~~~~~~~~~~~~~
Handles real-time audio capture using sounddevice.
"""

import queue
import sounddevice as sd
from typing import Generator
from loguru import logger

class MicrophoneStream:
    """
    Captures raw audio from the default microphone and yields chunks.
    """

    def __init__(self, sample_rate: int = 16000, chunk_size: int = 512):
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self._queue: queue.Queue[bytes] = queue.Queue()
        self._is_running = False
        self._stream = None

    def _callback(self, indata, frames, time, status):
        """Called by sounddevice for each audio block."""
        if status:
            logger.warning(f"Microphone status: {status}")
        self._queue.put(bytes(indata))

    def start(self) -> None:
        """Starts the audio stream."""
        logger.info(f"Starting microphone stream ({self.sample_rate}Hz)...")
        self._is_running = True
        self._stream = sd.RawInputStream(
            samplerate=self.sample_rate,
            blocksize=self.chunk_size,
            dtype='int16',
            channels=1,
            callback=self._callback
        )
        self._stream.start()

    def stop(self) -> None:
        """Stops the audio stream."""
        if self._is_running:
            logger.info("Stopping microphone stream.")
            self._is_running = False
            if self._stream:
                self._stream.stop()
                self._stream.close()

    def listen(self) -> Generator[bytes, None, None]:
        """Yields audio chunks continuously."""
        if not self._is_running:
            self.start()
            
        try:
            while self._is_running:
                # Block until we get a chunk
                chunk = self._queue.get()
                yield chunk
        finally:
            self.stop()
