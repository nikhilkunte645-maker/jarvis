import sys
import json
import argparse
import logging
import requests
from pathlib import Path

from jarvis.core.config_loader import cfg
from jarvis.core.orchestrator import Orchestrator

logger = logging.getLogger(__name__)

def check_ollama(model: str):
    """Check if Ollama is reachable."""
    try:
        resp = requests.get("http://localhost:11434/api/tags", timeout=3)
        if resp.status_code != 200:
            raise requests.RequestException("Non-200 response")
    except requests.RequestException:
        print(f"Error: Ollama is not reachable at http://localhost:11434.")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="JARVIS Main Entrypoint")
    parser.add_argument("--text", action="store_true", help="typed input instead of the microphone")
    parser.add_argument("--dry-run", action="store_true", help="print the parsed JSON without executing it")
    parser.add_argument("--voice", action="store_true", help="full voice mode (wake word, STT, TTS) - default if no flag is given")
    parser.add_argument("--laptop-only", action="store_true", help="skip the phone bridge")
    parser.add_argument("--config", type=str, help="override the settings.yaml path")

    args = parser.parse_args()

    # If config override is provided, update and reload
    if args.config:
        cfg._path = Path(args.config)
        cfg.reload()
        
    if args.laptop_only:
        # Prevent phone bridge reconnections
        if "phone" not in cfg._data:
            cfg._data["phone"] = {}
        cfg._data["phone"]["enabled"] = False

    # Ollama check
    model = cfg.get("brain.model", default="llama3.2")
    check_ollama(model)

    orch = Orchestrator()
    
    original_connect = orch.phone.connect

    def safe_connect():
        if args.laptop_only:
            return False
        try:
            return original_connect()
        except FileNotFoundError:
            logger.warning("ADB not found or phone unavailable. Continuing with laptop only.")
            return False
        except Exception as e:
            logger.warning(f"Phone connection failed: {e}. Continuing with laptop only.")
            return False

    orch.phone.connect = safe_connect
    
    # setup handles graceful phone/adb failure automatically with safe_connect
    orch.setup()

    if args.text:
        # Text loop
        while True:
            try:
                user_input = input("You> ").strip()
                if not user_input:
                    continue
                if user_input.lower() in ("quit", "exit"):
                    break
                
                if args.dry_run:
                    cmd = orch.brain.parse(user_input)
                    # Convert enums to strings
                    intent_str = str(cmd.intent.value) if hasattr(cmd.intent, "value") else str(cmd.intent)
                    target_str = str(cmd.target.value) if hasattr(cmd.target, "value") else str(cmd.target)
                    print(json.dumps({
                        "intent": intent_str,
                        "target": target_str,
                        "action": cmd.action,
                        "parameters": cmd.parameters,
                        "confidence": cmd.confidence
                    }, indent=2))
                    print(f"WOULD RUN: {cmd.action} on {target_str}")
                else:
                    orch.process(user_input)
            except KeyboardInterrupt:
                break
    else:
        # Voice mode loop
        from jarvis.audio.pipeline import AudioPipeline
        
        stt_model = cfg.get("audio.stt.model_size", default="base.en")
        wake_word = cfg.get("audio.wake_word.keyword", default="jarvis").lower()
        
        def on_transcription(text: str):
            text = text.strip()
            if not text:
                return
                
            # Check wake word on transcribed text
            lower_text = text.lower()
            # Handle cases like "Jarvis, open notepad" or "Jarvis open notepad"
            if not lower_text.startswith(wake_word):
                logger.info(f"Ignored (no wake word): {text}")
                return
                
            # Strip the wake word
            command = text[len(wake_word):].strip()
            # Strip leading punctuation if present
            while command and command[0] in ".,;:!?- ":
                command = command[1:].strip()
                
            if not command:
                return
                
            if args.dry_run:
                cmd = orch.brain.parse(command)
                intent_str = str(cmd.intent.value) if hasattr(cmd.intent, "value") else str(cmd.intent)
                target_str = str(cmd.target.value) if hasattr(cmd.target, "value") else str(cmd.target)
                print(json.dumps({
                    "intent": intent_str,
                    "target": target_str,
                    "action": cmd.action,
                    "parameters": cmd.parameters,
                    "confidence": cmd.confidence
                }, indent=2))
                print(f"WOULD RUN: {cmd.action} on {target_str}")
            else:
                orch.process(command)

        pipeline = AudioPipeline(stt_model=stt_model, on_transcription=on_transcription)
        print(f"Voice mode activated. Listening for '{wake_word}'...")
        pipeline.run()

if __name__ == "__main__":
    main()
