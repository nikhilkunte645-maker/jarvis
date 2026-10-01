"""
tests/test_event_bus.py
~~~~~~~~~~~~~~~~~~~~~~~
Unit tests for the EventBus pub-sub system.
"""

from __future__ import annotations

import pytest

from jarvis.core.event_bus import Event, EventBus


def test_basic_emit_and_receive() -> None:
    bus = EventBus()
    received: list[Event] = []

    bus.subscribe("test.topic", lambda e: received.append(e))
    bus.emit("test.topic", data={"value": 42})

    assert len(received) == 1
    assert received[0].data["value"] == 42


def test_wildcard_handler() -> None:
    bus = EventBus()
    received: list[str] = []

    bus.subscribe("*", lambda e: received.append(e.topic))
    bus.emit("stt.transcription", data={})
    bus.emit("wake_word.detected", data={})

    assert "stt.transcription" in received
    assert "wake_word.detected" in received


def test_unsubscribe() -> None:
    bus = EventBus()
    calls: list[int] = []

    def handler(e: Event) -> None:
        calls.append(1)

    bus.subscribe("foo.bar", handler)
    bus.emit("foo.bar")
    assert len(calls) == 1

    bus.unsubscribe("foo.bar", handler)
    bus.emit("foo.bar")
    assert len(calls) == 1  # still 1 – handler removed


def test_decorator_registration() -> None:
    bus = EventBus()
    received: list[Event] = []

    @bus.on("decorated.topic")
    def handle(event: Event) -> None:
        received.append(event)

    bus.emit("decorated.topic", data={"x": "y"})
    assert received[0].data["x"] == "y"


def test_handler_exception_does_not_propagate() -> None:
    bus = EventBus()
    after: list[bool] = []

    bus.subscribe("err.topic", lambda e: (_ for _ in ()).throw(ValueError("boom")))  # type: ignore
    bus.subscribe("err.topic", lambda e: after.append(True))

    # Should not raise
    bus.emit("err.topic")
    assert after == [True]


def test_no_handlers_is_silent() -> None:
    bus = EventBus()
    # Should not raise
    bus.emit("orphan.topic", data={"a": 1})
