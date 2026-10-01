"""
jarvis.brain.memory
~~~~~~~~~~~~~~~~~~~
Short-term conversation memory for the intent-parsing LLM.

Keeps the last N exchanges so the brain understands pronoun references and
multi-turn commands like "now do the same on my phone".

Usage::

    from jarvis.brain.memory import ConversationMemory

    mem = ConversationMemory()
    mem.add_turn(user="open Spotify", assistant='{"intent":"open_app",...}')
    ctx = mem.format_history()
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Optional

from loguru import logger

from jarvis.core.config_loader import cfg


@dataclass
class Turn:
    """Single user ↔ assistant exchange.

    Attributes:
        user: User's transcribed speech.
        assistant: JSON string the LLM returned.
    """

    user: str
    assistant: str


class ConversationMemory:
    """Ring-buffer of recent conversation turns.

    Attributes:
        max_history: Maximum number of turns to retain.
    """

    def __init__(self, max_history: Optional[int] = None) -> None:
        self.max_history: int = max_history or cfg.get(
            "brain.memory.max_history", default=8
        )
        self._turns: deque[Turn] = deque(maxlen=self.max_history)

    # ── Public API ──────────────────────────────────────────────────────────

    def add_turn(self, user: str, assistant: str) -> None:
        """Record a completed exchange.

        Args:
            user: User's voice command.
            assistant: LLM's JSON response (as string).

        Returns:
            None
        """
        self._turns.append(Turn(user=user, assistant=assistant))
        logger.debug(f"Memory: {len(self._turns)}/{self.max_history} turns stored.")

    def format_history(self) -> str:
        """Format the turn history for inclusion in the LLM prompt.

        Returns:
            Multi-line string of alternating User/Assistant lines,
            or empty string if no history exists.
        """
        if not self._turns:
            return ""
        lines = []
        for turn in self._turns:
            lines.append(f"User: {turn.user}")
            lines.append(f"Assistant: {turn.assistant}")
        return "\n".join(lines)

    def clear(self) -> None:
        """Wipe all stored turns.

        Returns:
            None
        """
        self._turns.clear()
        logger.debug("Memory cleared.")

    def __len__(self) -> int:
        return len(self._turns)
