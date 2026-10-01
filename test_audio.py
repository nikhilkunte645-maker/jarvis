import sys
import threading
from loguru import logger
from jarvis.audio import AudioPipeline

logger.remove()
logger.add(sys.stderr, level="DEBUG")

def handle_text(text: str) -> None:
    print(f"\n✅ YOU SAID: {text}\n")

if __name__ == "__main__":
    pipeline = AudioPipeline(stt_model="tiny.en", on_transcription=handle_text)
    
    # Run pipeline in a background thread so we can stop it easily
    t = threading.Thread(target=pipeline.run)
    t.start()
    
    print("[Mic] Speak into your microphone. Press Enter to stop.")
    input()
    
    pipeline.stop()
    t.join()
    print("Test finished.")
