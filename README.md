# JARVIS – Just A Rather Very Intelligent System

> A modular, **completely local-first**, voice-controlled AI assistant that
> controls your laptop **and** Android phone via a single natural-language
> voice interface.

---

## ✨ Features

| Capability | Details |
|---|---|
| 🎤 **Voice input** | Continuous or wake-word–triggered listening |
| 🧠 **Local LLM brain** | Ollama (llama3.2, phi-3, qwen2.5 …) – no cloud required |
| 💻 **Laptop control** | Open apps, search web, type, volume, brightness, screenshot |
| 📱 **Phone control** | ADB + uiautomator2 – tap, swipe, launch apps, send WhatsApp |
| 🔊 **Voice feedback** | edge-tts / pyttsx3 / piper TTS |
| 🖥️ **Live HUD** | Rich terminal dashboard |
| 🔒 **Privacy** | Everything runs offline after model download |
| 🔌 **Modular** | Swap any component independently |

---

## 📁 Project Structure

```
jarvis/
├── audio/              # Microphone, VAD, STT, wake-word, pipeline
├── brain/              # LLM client, prompt templates, memory, intent parser
├── laptop_bridge/      # Desktop automation (pyautogui, OS helpers)
├── phone_bridge/       # ADB manager, uiautomator2 executor
├── ui_hud/             # Terminal HUD, TTS engine
├── core/               # Config loader, logger, event bus, schemas, orchestrator
├── skills/             # Pluggable skill modules (future expansion)
├── config/             # settings.yaml, device profiles
├── tests/              # Unit + integration tests
main.py                 # Entry point (thin orchestrator wire-up)
requirements.txt
.env.example
```

---

## 🚀 Quick Start

### 1. Prerequisites

- **Python 3.11+**
- **Ollama** installed and running: https://ollama.ai
- **Android Platform Tools** (for phone control): https://developer.android.com/tools/releases/platform-tools
- A microphone connected to your laptop

### 2. Clone & Install

```bash
git clone <repo-url>
cd "jarvis clone"

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS/Linux

pip install -r requirements.txt
```

### 3. Download an Ollama Model

```bash
ollama pull llama3.2          # ~2 GB, recommended
# OR
ollama pull phi3              # ~2.3 GB, faster on CPU
# OR
ollama pull qwen2.5:3b        # ~2 GB, multilingual
```

### 4. Configure

```bash
# Copy the environment template
cp .env.example .env

# Edit jarvis/config/settings.yaml to set:
# - audio.stt.model_size  (tiny.en = fastest, base.en = balanced)
# - brain.model           (must match what you pulled in Ollama)
# - tts.backend           (pyttsx3 = fully offline, edge_tts = higher quality)
```

### 5. Run JARVIS

```bash
# Full voice mode (wake-word + mic)
python main.py

# Always-on mode (no wake word needed)
python main.py --no-wake-word

# One-shot text command (no microphone needed – great for testing)
python main.py --text "open Chrome and search SpaceX"

# Laptop-only (no phone required)
python main.py --no-phone --no-wake-word
```

---

## 📱 Android Phone Setup (ADB)

### USB Connection

1. Enable **Developer Options** on your phone:
   - `Settings → About Phone → tap "Build Number" 7 times`
2. Enable **USB Debugging**:
   - `Settings → Developer Options → USB Debugging ✓`
3. Connect phone via USB cable
4. Accept the "Allow USB Debugging?" prompt on your phone
5. Verify: `adb devices` should show your device

### Wireless ADB (Android 11+)

1. `Settings → Developer Options → Wireless debugging → Pair device with pairing code`
2. Note the pairing IP:port and code
3. Run: `adb pair <ip:port> <code>`
4. Then: `adb connect <ip:port>`
5. Set `phone.wifi_host` in `settings.yaml` and `phone.connection_mode: wifi`

### Verify Phone Control

```bash
python main.py --text "take a screenshot on my phone"
```

---

## 🗣️ Voice Command Examples

### Laptop Commands

| You say | What happens |
|---|---|
| `"Open Chrome"` | Launches Google Chrome |
| `"Search for SpaceX latest news"` | Opens browser with Google search |
| `"Set volume to 40 percent"` | Adjusts system volume |
| `"Take a screenshot"` | Saves screenshot to Desktop |
| `"Type hello world"` | Types text at cursor position |
| `"Press Control C"` | Sends keyboard shortcut |
| `"Open the file manager"` | Launches file explorer |
| `"Lock the screen"` | Locks workstation |

### Phone Commands

| You say | What happens |
|---|---|
| `"Send a WhatsApp message to Mom saying I'll be home in 20 minutes"` | Opens WhatsApp, finds contact, drafts message |
| `"Open Instagram on my phone"` | Launches Instagram via ADB |
| `"Take a screenshot on my phone"` | Captures phone screen, copies to laptop |
| `"Turn on Do Not Disturb on the phone"` | Sets phone DND mode |
| `"Turn off WiFi on the phone"` | Disables phone WiFi |
| `"Go back on my phone"` | Presses Android Back button |
| `"Open YouTube on my phone"` | Launches YouTube app |
| `"Call Mom"` | Opens phone dialer for contact |

### Multi-Step Commands

| You say | What happens |
|---|---|
| `"Open Chrome and search for the latest SpaceX launch"` | Opens Chrome then searches |
| `"Take a screenshot on my phone and tell me where it was saved"` | Screenshot + reports path |
| `"Turn on DND on the phone and set laptop volume to 30 percent"` | Both devices simultaneously |

---

## 🔧 Configuration Reference

Edit `jarvis/config/settings.yaml`:

| Key | Default | Description |
|---|---|---|
| `audio.wake_word.enabled` | `true` | Require "jarvis" before commands |
| `audio.wake_word.keyword` | `jarvis` | Wake word phrase |
| `audio.stt.model_size` | `base.en` | Whisper model (tiny/base/small/medium) |
| `audio.stt.device` | `cpu` | Use `cuda` if you have an NVIDIA GPU |
| `brain.backend` | `ollama` | LLM provider |
| `brain.model` | `llama3.2` | Must match a pulled Ollama model |
| `phone.serial` | `""` | ADB device serial (blank = auto-detect) |
| `tts.backend` | `edge_tts` | TTS engine (edge_tts/pyttsx3/piper) |
| `tts.voice` | `en-US-GuyNeural` | edge-tts voice name |

---

## 🧪 Running Tests

```bash
pytest jarvis/tests/ -v
```

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────┐
│                    main.py                          │
│              (entry point / wire-up)                │
└───────────────────┬─────────────────────────────────┘
                    │
      ┌─────────────▼──────────────┐
      │       AudioPipeline        │
      │  Mic → VAD → WakeWord → STT│
      └─────────────┬──────────────┘
                    │ text
      ┌─────────────▼──────────────┐
      │        Orchestrator        │  ← Event Bus
      │  (routes, confirms, logs)  │
      └──────┬──────────────┬──────┘
             │              │
    ┌────────▼─────┐  ┌─────▼────────┐
    │ IntentParser │  │    Safety     │
    │  (brain/LLM) │  │ Confirmation  │
    └────────┬─────┘  └──────────────┘
             │ JarvisCommand
    ┌────────▼─────────────────────┐
    │         Dispatch             │
    ├─────────────┬────────────────┤
    │             │                │
┌───▼───┐   ┌────▼────┐      ┌────▼────┐
│Laptop │   │  Phone  │      │   Both  │
│Bridge │   │ Bridge  │      │         │
│pyauto │   │ADB+u2   │      │ (serial)│
└───────┘   └─────────┘      └─────────┘
    │              │
    └──────┬───────┘
           │ ExecutionResult
    ┌──────▼──────┐
    │  TTS + HUD  │
    └─────────────┘
```

---

## 🔮 First 10 Voice Commands Demo

These 10 commands exercise the full JARVIS stack end-to-end:

1. **"Jarvis, open Chrome"**
   → Laptop: launches Chrome browser

2. **"Search for SpaceX latest launch on YouTube"**
   → Laptop: opens YouTube search results

3. **"Set my laptop volume to 30 percent"**
   → Laptop: adjusts system volume via OS API

4. **"Take a screenshot on my laptop"**
   → Laptop: saves PNG to Desktop

5. **"Open WhatsApp on my phone"**
   → Phone: ADB launches com.whatsapp

6. **"Send a WhatsApp message to Mom saying I'll be home in 20 minutes"**
   → Phone: opens chat, types message (requires confirmation)

7. **"Take a screenshot on my phone and tell me where it was saved"**
   → Phone: screencap + pull → reports local path

8. **"Turn on Do Not Disturb on the phone and set laptop volume to zero"**
   → Both: multi-step across two devices simultaneously

9. **"Open Instagram on my phone, swipe up twice"**
   → Phone: launches Instagram, performs two upward swipes

10. **"Lock my screen"**
    → Laptop: locks the workstation session

---

## 🛡️ Privacy & Security

- **All processing is local** – audio never leaves your machine (when using Ollama)
- **Cloud LLM** is opt-in only (`brain.backend: openai`) and off by default
- **High-risk actions** (send_message, make_call, delete) require explicit confirmation
- **ADB** only connects to physically/wirelessly paired devices you've already trusted
- **No telemetry** – JARVIS collects nothing

---

## 🤝 Contributing

1. Fork the repo
2. Create a feature branch: `git checkout -b feat/my-skill`
3. Add your skill in `jarvis/skills/`
4. Write tests in `jarvis/tests/`
5. Run `ruff check . && mypy jarvis/ && pytest`
6. Open a PR

---

## 📄 License

MIT – See [LICENSE](LICENSE) for details.
