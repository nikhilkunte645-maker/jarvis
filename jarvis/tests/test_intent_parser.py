"""
tests/test_intent_parser.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Unit tests for IntentParser using a mocked LLM client.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from jarvis.brain.intent_parser import IntentParser
from jarvis.brain.llm_client import LLMClient
from jarvis.core.schemas import IntentType, TargetDevice


def _make_llm_response(intent: str, target: str, action: str, **kwargs) -> str:
    data = {
        "intent": intent,
        "target": target,
        "action": action,
        "parameters": kwargs.get("parameters", {}),
        "app_package": kwargs.get("app_package"),
        "confidence": kwargs.get("confidence", 0.95),
        "requires_confirmation": kwargs.get("requires_confirmation", False),
        "steps": kwargs.get("steps", []),
    }
    return json.dumps(data)


@pytest.fixture
def mock_llm() -> MagicMock:
    return MagicMock(spec=LLMClient)


def test_parse_open_app(mock_llm: MagicMock) -> None:
    mock_llm.complete.return_value = _make_llm_response(
        intent="open_app",
        target="laptop",
        action="open_app",
        parameters={"app": "Chrome"},
    )
    parser = IntentParser(llm=mock_llm)
    cmd = parser.parse("open Chrome")

    assert cmd.intent == IntentType.OPEN_APP
    assert cmd.target == TargetDevice.LAPTOP
    assert cmd.parameters["app"] == "Chrome"
    assert cmd.confidence == 0.95


def test_parse_send_message_requires_confirmation(mock_llm: MagicMock) -> None:
    mock_llm.complete.return_value = _make_llm_response(
        intent="send_message",
        target="phone",
        action="whatsapp_send",
        parameters={"recipient": "Mom", "message": "Be home soon"},
        app_package="com.whatsapp",
        requires_confirmation=True,
    )
    parser = IntentParser(llm=mock_llm)
    cmd = parser.parse("send a WhatsApp message to Mom saying be home soon")

    assert cmd.requires_confirmation is True
    assert cmd.app_package == "com.whatsapp"
    assert cmd.parameters["recipient"] == "Mom"


def test_parse_multi_step(mock_llm: MagicMock) -> None:
    steps = [
        {
            "intent": "open_app", "target": "laptop", "action": "open_app",
            "parameters": {"app": "Chrome"}, "app_package": None,
            "confidence": 0.99, "requires_confirmation": False, "steps": [],
        },
        {
            "intent": "search_web", "target": "laptop", "action": "search_web",
            "parameters": {"query": "SpaceX"}, "app_package": None,
            "confidence": 0.97, "requires_confirmation": False, "steps": [],
        },
    ]
    mock_llm.complete.return_value = _make_llm_response(
        intent="multi_step",
        target="laptop",
        action="open_then_search",
        steps=steps,
    )
    parser = IntentParser(llm=mock_llm)
    cmd = parser.parse("open Chrome and search SpaceX")

    assert cmd.intent == IntentType.MULTI_STEP
    assert len(cmd.steps) == 2
    assert cmd.steps[0].parameters["app"] == "Chrome"
    assert cmd.steps[1].parameters["query"] == "SpaceX"


def test_parse_llm_failure_returns_unknown(mock_llm: MagicMock) -> None:
    mock_llm.complete.side_effect = RuntimeError("Ollama not running")
    parser = IntentParser(llm=mock_llm)
    cmd = parser.parse("do something")

    assert cmd.intent == IntentType.UNKNOWN
    assert cmd.confidence == 0.0


def test_parse_malformed_json_returns_unknown(mock_llm: MagicMock) -> None:
    mock_llm.complete.return_value = "Sorry, I cannot do that."
    parser = IntentParser(llm=mock_llm)
    cmd = parser.parse("whatever")

    assert cmd.intent == IntentType.UNKNOWN


def test_memory_stores_turns(mock_llm: MagicMock) -> None:
    mock_llm.complete.return_value = _make_llm_response(
        intent="open_app", target="laptop", action="open_app",
        parameters={"app": "Notepad"},
    )
    parser = IntentParser(llm=mock_llm)
    parser.parse("open Notepad")
    assert len(parser.memory) == 1

    parser.parse("close it")
    assert len(parser.memory) == 2


def test_confidence_clamped(mock_llm: MagicMock) -> None:
    resp = json.loads(
        _make_llm_response(intent="open_app", target="laptop", action="open_app", parameters={"app": "something"})
    )
    resp["confidence"] = 99.9  # out-of-range
    mock_llm.complete.return_value = json.dumps(resp)
    parser = IntentParser(llm=mock_llm)
    cmd = parser.parse("anything")
    assert cmd.confidence <= 1.0


def test_validation_retry_and_clarification(mock_llm: MagicMock) -> None:
    # First response: missing "app" parameter
    first_response = _make_llm_response(
        intent="open_app",
        target="laptop",
        action="open_app",
        parameters={},  # Missing "app"
    )
    # Second response: still missing "app"
    second_response = _make_llm_response(
        intent="open_app",
        target="laptop",
        action="open_app",
        parameters={},  # Still missing "app"
    )
    mock_llm.complete.side_effect = [first_response, second_response]
    
    parser = IntentParser(llm=mock_llm)
    cmd = parser.parse("open something")
    
    # Should result in clarification command
    assert cmd.intent == IntentType.UNKNOWN
    assert cmd.action == "clarify"
    assert cmd.parameters["message"] == "Which app?"
    assert mock_llm.complete.call_count == 2
