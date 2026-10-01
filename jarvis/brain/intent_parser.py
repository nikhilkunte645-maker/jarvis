"""
jarvis.brain.intent_parser
~~~~~~~~~~~~~~~~~~~~~~~~~~~
Orchestrates the LLM call → JSON parse → Pydantic validation pipeline.

Given a raw transcribed string this module returns a validated
``JarvisCommand`` ready for the execution bridges.

Usage::

    from jarvis.brain.intent_parser import IntentParser

    parser = IntentParser()
    cmd = parser.parse("open Chrome and search SpaceX")
    print(cmd.intent, cmd.steps)
"""

from __future__ import annotations

import json
import re
from typing import Optional

from loguru import logger
from pydantic import ValidationError

from jarvis.brain.llm_client import LLMClient
from jarvis.brain.memory import ConversationMemory
from jarvis.brain.prompt_templates import SYSTEM_PROMPT, build_user_prompt
from jarvis.core.schemas import IntentType, JarvisCommand, TargetDevice


class IntentParser:
    """Converts transcribed text into a validated :class:`JarvisCommand`.

    Uses the configured LLM to produce structured JSON, then validates it
    against the Pydantic schema.  Falls back gracefully on parse errors.

    Attributes:
        llm: The LLM client used for inference.
        memory: Short-term conversation memory.
    """

    def __init__(
        self,
        llm: Optional[LLMClient] = None,
        memory: Optional[ConversationMemory] = None,
    ) -> None:
        self.llm = llm or LLMClient()
        self.memory = memory or ConversationMemory()

    # ── Public API ──────────────────────────────────────────────────────────

    def parse(self, text: str, _is_retry: bool = False) -> JarvisCommand:
        """Parse a voice command string into a structured JarvisCommand.

        Args:
            text: Raw transcribed voice input from the STT engine.
            _is_retry: Internal flag to prevent infinite retry loops.

        Returns:
            A validated :class:`JarvisCommand`.  On parse failure returns
            a ``JarvisCommand`` with ``intent=unknown`` or clarification.
        """
        history = self.memory.format_history()
        user_prompt = build_user_prompt(user_input=text, history=history)

        try:
            raw_response = self.llm.complete(system=SYSTEM_PROMPT, user=user_prompt)
            logger.debug(f"LLM raw response: {raw_response[:300]}")
        except RuntimeError as exc:
            logger.error(f"LLM call failed: {exc}")
            return self._fallback(text, f"LLM error: {exc}")

        json_str = self._extract_json(raw_response)
        if not json_str:
            logger.warning(f"No JSON found in LLM response: {raw_response[:200]}")
            return self._fallback(text, "No JSON in response")

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as exc:
            logger.warning(f"JSON decode error: {exc}  raw='{json_str[:200]}'")
            return self._fallback(text, f"JSON decode error: {exc}")

        try:
            cmd = JarvisCommand(**data, raw_text=text)
            logger.info(
                f"Parsed intent='{cmd.intent}'  target='{cmd.target}'  "
                f"action='{cmd.action}'  confidence={cmd.confidence:.2f}"
            )
            
            # Validation for required parameters
            missing_param = None
            if cmd.intent == IntentType.OPEN_APP and "app" not in cmd.parameters:
                missing_param = "app"
            elif cmd.intent == IntentType.SEARCH_WEB and "query" not in cmd.parameters:
                missing_param = "query"
            elif cmd.intent == IntentType.SYSTEM_CONTROL and cmd.action == "set_volume" and "level" not in cmd.parameters:
                missing_param = "level"
                
            if missing_param:
                if not _is_retry:
                    logger.warning(f"Missing required parameter '{missing_param}' for {cmd.intent}. Retrying...")
                    error_msg = f"ERROR: Your previous JSON was missing the required '{missing_param}' parameter in the 'parameters' object. Please try again and include it."
                    return self.parse(text + "\n\n" + error_msg, _is_retry=True)
                else:
                    logger.warning(f"Still missing '{missing_param}' after retry. Returning clarification.")
                    return JarvisCommand(
                        intent=IntentType.UNKNOWN,
                        target=TargetDevice.LAPTOP,
                        action="clarify",
                        parameters={"message": f"Which {missing_param}?"},
                        confidence=1.0,
                        requires_confirmation=False,
                        raw_text=text
                    )
                    
        except ValidationError as exc:
            logger.warning(f"Schema validation failed: {exc}")
            return self._fallback(text, f"Schema error: {exc}")

        # Store to memory only on success
        self.memory.add_turn(user=text, assistant=json_str)
        return cmd

    # ── Private helpers ─────────────────────────────────────────────────────

    @staticmethod
    def _extract_json(text: str) -> str:
        """Extract the first JSON object from the model's response.

        Handles cases where the model wraps JSON in markdown code fences.

        Args:
            text: Raw model response string.

        Returns:
            Extracted JSON string, or empty string if none found.
        """
        # Strip markdown code fences
        text = re.sub(r"```(?:json)?", "", text).strip()

        # Find first { … } block
        start = text.find("{")
        if start == -1:
            return ""

        # Walk to find balanced closing brace
        depth = 0
        for i, ch in enumerate(text[start:], start=start):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]
        return ""

    @staticmethod
    def _fallback(raw_text: str, reason: str) -> JarvisCommand:
        """Build a safe 'unknown' command when parsing fails.

        Args:
            raw_text: Original transcribed text.
            reason: Human-readable explanation of failure.

        Returns:
            A :class:`JarvisCommand` with ``intent=unknown``.
        """
        logger.warning(f"Falling back to unknown intent: {reason}")
        return JarvisCommand(
            intent=IntentType.UNKNOWN,
            target=TargetDevice.LAPTOP,
            action="unknown",
            parameters={"reason": reason},
            confidence=0.0,
            requires_confirmation=False,
            raw_text=raw_text,
        )
