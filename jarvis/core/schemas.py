"""
jarvis.core.schemas
~~~~~~~~~~~~~~~~~~~
Pydantic data models for the structured JSON that the brain emits and the
execution bridges consume.

These models are the contract between every module in JARVIS.  Any module
that produces or consumes a command MUST use these types.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator


# ── Enumerations ────────────────────────────────────────────────────────────


class IntentType(str, Enum):
    """Supported intent categories that the brain can classify."""

    OPEN_APP = "open_app"
    CLOSE_APP = "close_app"
    TYPE_TEXT = "type_text"
    SEARCH_WEB = "search_web"
    SEND_MESSAGE = "send_message"
    MAKE_CALL = "make_call"
    TAKE_SCREENSHOT = "take_screenshot"
    SYSTEM_CONTROL = "system_control"
    PHONE_ACTION = "phone_action"
    FILE_OPERATION = "file_operation"
    BROWSER_CONTROL = "browser_control"
    MEDIA_CONTROL = "media_control"
    MULTI_STEP = "multi_step"
    UNKNOWN = "unknown"


class TargetDevice(str, Enum):
    """Which device should execute the command."""

    LAPTOP = "laptop"
    PHONE = "phone"
    BOTH = "both"


# ── Main command schema ─────────────────────────────────────────────────────


class JarvisCommand(BaseModel):
    """Structured command emitted by the brain after intent parsing.

    This is the canonical data contract passed from brain → orchestrator
    → execution bridges.

    Attributes:
        intent: High-level intent classification.
        target: Which device(s) should act.
        action: Specific action string for the skill dispatcher.
        parameters: Arbitrary key-value pairs for the action.
        app_package: Android package name when relevant.
        confidence: 0.0–1.0 brain confidence score.
        requires_confirmation: Whether to ask user before executing.
        steps: Ordered list of sub-commands for multi-step intents.
        raw_text: Original transcribed utterance (for logging).
    """

    intent: IntentType = Field(..., description="Intent classification")
    target: TargetDevice = Field(TargetDevice.LAPTOP, description="Execution target")
    action: str = Field(..., description="Specific action identifier")
    parameters: dict[str, Any] = Field(
        default_factory=dict, description="Action-specific parameters"
    )
    app_package: Optional[str] = Field(
        None, description="Android package name (e.g. com.whatsapp)"
    )
    confidence: float = Field(
        1.0, ge=0.0, le=1.0, description="Brain confidence score"
    )
    requires_confirmation: bool = Field(
        False, description="Prompt user before executing"
    )
    steps: list["JarvisCommand"] = Field(
        default_factory=list, description="Sub-steps for multi_step intents"
    )
    raw_text: Optional[str] = Field(None, description="Original transcribed text")

    @field_validator("confidence", mode="before")
    @classmethod
    def clamp_confidence(cls, v: Any) -> float:
        """Ensure confidence is within [0.0, 1.0].

        Args:
            v: Raw value from JSON.

        Returns:
            Float clamped to [0.0, 1.0].
        """
        return max(0.0, min(1.0, float(v)))

    model_config = {"use_enum_values": True}


# ── Execution result ────────────────────────────────────────────────────────


class ExecutionResult(BaseModel):
    """Result returned by an execution bridge after running a command.

    Attributes:
        success: Whether the command completed without error.
        message: Human-readable description of the outcome.
        data: Optional structured data (e.g. screenshot path).
        device: Which device executed the command.
    """

    success: bool
    message: str = ""
    data: dict[str, Any] = Field(default_factory=dict)
    device: TargetDevice = TargetDevice.LAPTOP
