"""
jarvis/audio/vad.py
~~~~~~~~~~~~~~~~~~~
Voice Activity Detection using Silero VAD via ONNXRuntime.
"""

import os
import urllib.request
import numpy as np
import onnxruntime as ort
from loguru import logger

class VADSegmenter:
    """
    Detects speech in audio chunks and groups them into utterances.
    """

    MODEL_URL = "https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx"
    MODEL_PATH = os.path.join(os.path.dirname(__file__), "silero_vad.onnx")

    def __init__(self, sample_rate: int = 16000, threshold: float = 0.5):
        self.sample_rate = sample_rate
        self.threshold = threshold
        self._ensure_model_exists()
        
        # Load ONNX model
        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        self.session = ort.InferenceSession(self.MODEL_PATH, providers=['CPUExecutionProvider'], sess_options=opts)
        
        self.reset_state()

    def _ensure_model_exists(self) -> None:
        """Downloads the Silero VAD ONNX model if not present."""
        if not os.path.exists(self.MODEL_PATH):
            logger.info("Downloading Silero VAD ONNX model...")
            urllib.request.urlretrieve(self.MODEL_URL, self.MODEL_PATH)
            logger.info("Download complete.")

    def reset_state(self) -> None:
        """Resets the internal RNN state of the VAD model."""
        self._state = np.zeros((2, 1, 128), dtype=np.float32)

    def is_speech(self, chunk: bytes) -> bool:
        """
        Takes 16kHz int16 PCM bytes and returns True if speech is detected.
        Chunk must be exactly 512 samples (1024 bytes) for 16kHz.
        """
        # Convert bytes to float32 normalized [-1.0, 1.0]
        audio_int16 = np.frombuffer(chunk, np.int16)
        audio_float32 = (audio_int16 / 32768.0).astype(np.float32)
        
        # Add batch dimension [1, seq_len]
        input_data = np.expand_dims(audio_float32, axis=0)
        
        # ONNX inputs
        inputs = {
            'input': input_data,
            'sr': np.array(self.sample_rate, dtype=np.int64),
            'state': self._state
        }
        
        # Run inference
        out, self._state = self.session.run(None, inputs)
        prob = out[0][0]
        return bool(prob > self.threshold)
