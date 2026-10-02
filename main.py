import re
import sys
import json
import argparse
import requests
from pathlib import Path
from loguru import logger

from jarvis.core.config_loader import cfg
from jarvis.core.orchestrator import Orchestrator


def check_ollama(model: str):
    """Check if Ollama is reachable."""
    try:
        resp = requests.get("http://localhost:11434/api/tags", timeout=3)
        if resp.status_code != 200:
            raise requests.RequestException("Non-200 response")
    except requests.RequestException:
        print("Error: Ollama is not reachable at http://localhost:11434.")
        sys.exit(1)


def print_dry_run(orch, command: str):
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


def main():
    parser = argparse.ArgumentParser(description="JARVIS Main Entrypoint")
    parser.add_argument("--text", action="store_true", help="typed input instead of the microphone")
    parser.add_argument("--dry-run", action="store_true", help="print the parsed JSON without executing it")
    parser.add_argument("--voice", action="store_true", help="full voice mode (wake word, STT, TTS) - default if no flag is given")
    parser.add_argument("--laptop-only", action="store_true", help="skip the phone bridge")
    parser.add_argument("--config", type=str, help="override the settings.yaml path")

    args = parser.parse_args()

    if args.config:
        cfg._path = Path(args.config)
        cfg.reload()

    if args.laptop_only:
        if "phone" not in cfg._data:
            cfg._data["phone"] = {}
        cfg._data["phone"]["enabled"] = False

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
    orch.setup()

    if args.text:
        while True:
            try:
                user_input = input("You> ").strip()
                if not user_input:
                    continue
                if user_input.lower() in ("quit", "exit"):
                    break

                if args.dry_run:
                    print_dry_run(orch, user_input)
                else:
                    orch.process(user_input)
            except KeyboardInterrupt:
                break
    else:
        from jarvis.audio.pipeline import AudioPipeline

        stt_model = cfg.get("audio.stt.model_size", default="base.en")
        wake_word = cfg.get("audio.wake_word.keyword", default="jarvis").lower()

        wake_re = re.compile(
            rf"^\W*(?:hey|ok|okay)?\W*(?:{re.escape(wake_word)}|travis|jarvus|jervis|jairus|jarvas|jarves|jervas|gervais)\b[\s,.:;!?-]*",
            re.IGNORECASE,
        )

        def on_transcription(text: str):
            text = text.strip()
            if not text:
                return

            m = wake_re.match(text)
            if not m:
                logger.info(f"Ignored (no wake word): {text}")
                return

            command = text[m.end():].strip()
            if not command:
                return

            logger.info(f"Command: {command}")
            if args.dry_run:
                print_dry_run(orch, command)
            else:
                orch.process(command)

        pipeline = AudioPipeline(stt_model=stt_model, on_transcription=on_transcription)
        print(f"Voice mode activated. Listening for '{wake_word}'...")
        pipeline.run()


if __name__ == "__main__":
    main()