"""
jarvis.core.orchestrator
~~~~~~~~~~~~~~~~~~~~~~~~~
The central command router of JARVIS.

Receives a transcribed text string, hands it to the brain for intent
parsing, then dispatches the resulting ``JarvisCommand`` to the correct
execution bridge(s).  Handles multi-step workflows, confirmation gates,
and graceful phone-disconnection fallback.

Usage (called by main.py)::

    from jarvis.core.orchestrator import Orchestrator

    orch = Orchestrator()
    orch.setup()
    orch.process("open Chrome and search SpaceX")
"""

from __future__ import annotations

import threading
from typing import Optional

from loguru import logger

from jarvis.brain.intent_parser import IntentParser
from jarvis.core.config_loader import cfg
from jarvis.core.event_bus import bus
from jarvis.core.schemas import (
    ExecutionResult,
    IntentType,
    JarvisCommand,
    TargetDevice,
)
from jarvis.laptop_bridge.executor import LaptopExecutor
from jarvis.phone_bridge.executor import PhoneExecutor
from jarvis.ui_hud.terminal_hud import TerminalHUD
from jarvis.ui_hud.tts import TTSEngine


class Orchestrator:
    """Coordinates all JARVIS subsystems end-to-end.

    Attributes:
        brain: Intent parser (LLM wrapper).
        laptop: Desktop automation executor.
        phone: Android ADB executor.
        hud: Terminal status display.
        tts: Voice feedback engine.
    """

    def __init__(
        self,
        brain: Optional[IntentParser] = None,
        laptop: Optional[LaptopExecutor] = None,
        phone: Optional[PhoneExecutor] = None,
        hud: Optional[TerminalHUD] = None,
        tts: Optional[TTSEngine] = None,
    ) -> None:
        self.brain = brain or IntentParser()
        self.laptop = laptop or LaptopExecutor()
        self.phone = phone or PhoneExecutor()
        self.hud = hud or TerminalHUD()
        self.tts = tts or TTSEngine()

        self._safety_actions: list[str] = cfg.get(
            "safety.require_confirmation", default=[]
        )
        self._confirmation_timeout: int = cfg.get(
            "safety.confirmation_timeout_seconds", default=10
        )

        # Register event bus listeners
        bus.subscribe("wake_word.detected", self._on_wake_word)
        bus.subscribe("stt.transcription", self._on_transcription)

    # ── Public API ──────────────────────────────────────────────────────────

    def setup(self) -> None:
        """Initialise all subsystems (call once before the listen loop).

        Returns:
            None
        """
        logger.info("Orchestrator: setting up subsystems …")

        # Connect phone (non-fatal if not present)
        phone_ok = self.phone.connect()
        self.hud.set_phone_status(phone_ok)
        if phone_ok:
            self.tts.speak("Phone connected.")
            logger.info("Phone bridge ready.")
        else:
            logger.warning("Phone not connected – phone actions will be skipped.")

        self.tts.speak("JARVIS online. How can I help you?")
        logger.info("Orchestrator ready.")

    def process(self, text: str) -> list[ExecutionResult]:
        """Full pipeline: text → intent → execute → feedback.

        This is the main entry point for a single voice command.

        Args:
            text: Transcribed voice command string.

        Returns:
            List of :class:`ExecutionResult` objects (one per executed step).
        """
        if not text.strip():
            return []

        self.hud.show_transcription(text)
        logger.info(f"Processing: '{text}'")

        # 1. Parse intent
        cmd = self.brain.parse(text)
        self.hud.show_intent(
            intent=cmd.intent,
            target=cmd.target,
            action=cmd.action,
            confidence=cmd.confidence,
        )

        if cmd.intent == IntentType.UNKNOWN:
            msg = "I didn't understand that. Could you rephrase?"
            self.tts.speak(msg)
            self.hud.show_error(msg)
            return []

        # 2. Safety confirmation gate
        if cmd.requires_confirmation or cmd.action in self._safety_actions:
            confirmed = self._confirm_action(cmd)
            if not confirmed:
                self.tts.speak("Action cancelled.")
                self.hud.show_result(success=False, message="Cancelled by user.")
                return []

        # 3. Execute
        results = self._execute(cmd)

        # 4. Feedback
        for result in results:
            self.hud.show_result(
                success=result.success,
                message=result.message,
                data=result.data if result.data else None,
            )
            if result.success:
                self.tts.speak(result.message)
            else:
                self.tts.speak(f"Something went wrong. {result.message}")

        # Re-check phone health after every command
        if not self.phone.available and cfg.get("phone.enabled", default=True):
            logger.warning("Phone disconnected after command – attempting reconnect.")
            self.phone.connect()

        return results

    # ── Execution router ────────────────────────────────────────────────────

    def _execute(self, cmd: JarvisCommand) -> list[ExecutionResult]:
        """Route a command to one or both bridges.

        Args:
            cmd: Validated :class:`JarvisCommand`.

        Returns:
            List of :class:`ExecutionResult`.
        """
        if cmd.intent == IntentType.MULTI_STEP:
            return self._execute_multi(cmd)

        results: list[ExecutionResult] = []

        if cmd.target in (TargetDevice.LAPTOP, TargetDevice.BOTH):
            results.append(self.laptop.run(cmd))

        if cmd.target in (TargetDevice.PHONE, TargetDevice.BOTH):
            if self.phone.available:
                results.append(self.phone.run(cmd))
            else:
                results.append(
                    ExecutionResult(
                        success=False,
                        message="Phone not connected.",
                        device=TargetDevice.PHONE,
                    )
                )

        if not results:
            results.append(
                ExecutionResult(
                    success=False,
                    message=f"No executor for target '{cmd.target}'",
                )
            )

        return results

    def _execute_multi(self, cmd: JarvisCommand) -> list[ExecutionResult]:
        """Execute each step of a multi-step command in sequence.

        Args:
            cmd: :class:`JarvisCommand` with ``intent=multi_step``.

        Returns:
            Aggregated list of results, one per sub-step.
        """
        all_results: list[ExecutionResult] = []
        for i, step in enumerate(cmd.steps, start=1):
            logger.info(f"Multi-step {i}/{len(cmd.steps)}: {step.action}")
            step_results = self._execute(step)
            all_results.extend(step_results)
            # Abort on first failure
            if any(not r.success for r in step_results):
                logger.warning(f"Multi-step aborted at step {i}")
                break
        return all_results

    # ── Safety confirmation ─────────────────────────────────────────────────

    def _confirm_action(self, cmd: JarvisCommand) -> bool:
        """Prompt user to confirm a high-risk action.

        Uses the terminal HUD (keyboard input as fallback since voice
        confirmation requires the audio pipeline to still be active).

        Args:
            cmd: The command that needs confirmation.

        Returns:
            True if confirmed.
        """
        description = (
            f"Action: {cmd.action}  |  Target: {cmd.target}\n"
            f"Params: {cmd.parameters}"
        )
        self.tts.speak(
            f"This action requires confirmation: {cmd.action}. "
            "Please confirm or cancel."
        )
        return self.hud.show_confirmation_prompt(description)

    # ── Event bus handlers ──────────────────────────────────────────────────

    def _on_wake_word(self, event: object) -> None:
        """Handle wake-word detection event from the bus.

        Args:
            event: Event object (ignored).

        Returns:
            None
        """
        self.hud.show_wake_word()
        self.tts.speak("Yes?")

    def _on_transcription(self, event: object) -> None:
        """Handle live transcription events for UI updates.

        Args:
            event: Bus event with ``data.text``.

        Returns:
            None
        """
        # The full process() is called by the audio pipeline loop in main.py;
        # this just updates the HUD in real-time if needed.
        pass
