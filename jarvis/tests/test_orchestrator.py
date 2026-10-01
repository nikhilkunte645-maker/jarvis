"""
tests/test_orchestrator.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~
Integration tests for the Orchestrator using mocked subsystems.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from jarvis.brain.intent_parser import IntentParser
from jarvis.core.orchestrator import Orchestrator
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


@pytest.fixture
def mock_orch() -> Orchestrator:
    brain = MagicMock(spec=IntentParser)
    laptop = MagicMock(spec=LaptopExecutor)
    phone = MagicMock(spec=PhoneExecutor)
    hud = MagicMock(spec=TerminalHUD)
    tts = MagicMock(spec=TTSEngine)
    phone.available = True

    orch = Orchestrator(
        brain=brain, laptop=laptop, phone=phone, hud=hud, tts=tts
    )
    return orch


def _make_cmd(
    intent: IntentType = IntentType.OPEN_APP,
    target: TargetDevice = TargetDevice.LAPTOP,
    action: str = "open_app",
    requires_confirmation: bool = False,
    steps: list | None = None,
) -> JarvisCommand:
    return JarvisCommand(
        intent=intent,
        target=target,
        action=action,
        parameters={"app_name": "Chrome"},
        confidence=0.95,
        requires_confirmation=requires_confirmation,
        steps=steps or [],
    )


def test_laptop_command_dispatched(mock_orch: Orchestrator) -> None:
    cmd = _make_cmd(target=TargetDevice.LAPTOP)
    mock_orch.brain.parse.return_value = cmd
    mock_orch.laptop.run.return_value = ExecutionResult(
        success=True, message="Opened Chrome", device=TargetDevice.LAPTOP
    )

    results = mock_orch.process("open Chrome")

    mock_orch.laptop.run.assert_called_once_with(cmd)
    mock_orch.phone.run.assert_not_called()
    assert results[0].success is True


def test_phone_command_dispatched(mock_orch: Orchestrator) -> None:
    cmd = _make_cmd(target=TargetDevice.PHONE, action="launch_app")
    mock_orch.brain.parse.return_value = cmd
    mock_orch.phone.run.return_value = ExecutionResult(
        success=True, message="Launched WhatsApp", device=TargetDevice.PHONE
    )

    results = mock_orch.process("open WhatsApp on my phone")

    mock_orch.phone.run.assert_called_once_with(cmd)
    mock_orch.laptop.run.assert_not_called()


def test_both_target_dispatched(mock_orch: Orchestrator) -> None:
    cmd = _make_cmd(target=TargetDevice.BOTH, action="take_screenshot")
    mock_orch.brain.parse.return_value = cmd
    mock_orch.laptop.run.return_value = ExecutionResult(success=True, message="L", device=TargetDevice.LAPTOP)
    mock_orch.phone.run.return_value = ExecutionResult(success=True, message="P", device=TargetDevice.PHONE)

    results = mock_orch.process("take screenshot on both")

    assert len(results) == 2
    mock_orch.laptop.run.assert_called_once()
    mock_orch.phone.run.assert_called_once()


def test_unknown_intent_skips_execution(mock_orch: Orchestrator) -> None:
    cmd = _make_cmd(intent=IntentType.UNKNOWN, action="unknown")
    mock_orch.brain.parse.return_value = cmd

    results = mock_orch.process("asdfghjkl gibberish")

    mock_orch.laptop.run.assert_not_called()
    mock_orch.phone.run.assert_not_called()
    assert results == []


def test_multi_step_executes_in_order(mock_orch: Orchestrator) -> None:
    step1 = _make_cmd(intent=IntentType.OPEN_APP, action="open_app")
    step2 = JarvisCommand(
        intent=IntentType.SEARCH_WEB,
        target=TargetDevice.LAPTOP,
        action="search_web",
        parameters={"query": "SpaceX"},
        confidence=0.9,
    )
    cmd = _make_cmd(intent=IntentType.MULTI_STEP, action="multi", steps=[step1, step2])
    mock_orch.brain.parse.return_value = cmd
    mock_orch.laptop.run.return_value = ExecutionResult(success=True, message="ok", device=TargetDevice.LAPTOP)

    results = mock_orch.process("open Chrome and search SpaceX")
    assert mock_orch.laptop.run.call_count == 2


def test_phone_disconnected_graceful_fallback(mock_orch: Orchestrator) -> None:
    mock_orch.phone.available = False
    cmd = _make_cmd(target=TargetDevice.PHONE, action="launch_app")
    mock_orch.brain.parse.return_value = cmd

    results = mock_orch.process("open WhatsApp")

    mock_orch.phone.run.assert_not_called()
    assert results[0].success is False
    assert "not connected" in results[0].message.lower()


def test_empty_text_returns_early(mock_orch: Orchestrator) -> None:
    results = mock_orch.process("  ")
    mock_orch.brain.parse.assert_not_called()
    assert results == []
