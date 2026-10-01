"""
jarvis.core.event_bus
~~~~~~~~~~~~~~~~~~~~~
A simple synchronous + async pub-sub event bus that decouples modules.

Producers emit events; consumers register handlers.  No external broker
required – everything runs in-process.

Usage::

    from jarvis.core.event_bus import bus, Event

    # Register a handler
    @bus.on("stt.transcription")
    def handle_text(event: Event) -> None:
        print(event.data["text"])

    # Emit an event
    bus.emit("stt.transcription", data={"text": "open Chrome"})
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Callable

from loguru import logger

Handler = Callable[["Event"], None]
AsyncHandler = Callable[["Event"], "asyncio.Future[None]"]


@dataclass
class Event:
    """Container for a single bus event.

    Attributes:
        topic: Dot-separated topic string, e.g. ``"stt.transcription"``.
        data:  Arbitrary payload dict.
        source: Optional module that emitted the event.
    """

    topic: str
    data: dict[str, Any] = field(default_factory=dict)
    source: str = "unknown"


class EventBus:
    """In-process synchronous publish-subscribe bus.

    All handlers registered for a topic are called synchronously in
    registration order when :meth:`emit` is called.

    Example:
        >>> bus = EventBus()
        >>> received = []
        >>> bus.on("test.topic")(lambda e: received.append(e.data))
        >>> bus.emit("test.topic", data={"x": 1})
        >>> assert received[0]["x"] == 1
    """

    def __init__(self) -> None:
        self._handlers: dict[str, list[Handler]] = {}

    # ── Registration ────────────────────────────────────────────────────────

    def on(self, topic: str) -> Callable[[Handler], Handler]:
        """Decorator that registers *func* as a handler for *topic*.

        Args:
            topic: Event topic string (supports ``"*"`` wildcard).

        Returns:
            The original function unchanged (for chaining).
        """

        def decorator(func: Handler) -> Handler:
            self._handlers.setdefault(topic, []).append(func)
            logger.debug(f"EventBus: registered '{func.__name__}' on '{topic}'")
            return func

        return decorator

    def subscribe(self, topic: str, handler: Handler) -> None:
        """Programmatically subscribe *handler* to *topic*.

        Args:
            topic: Event topic string.
            handler: Callable accepting a single :class:`Event` argument.

        Returns:
            None
        """
        self._handlers.setdefault(topic, []).append(handler)

    def unsubscribe(self, topic: str, handler: Handler) -> None:
        """Remove a previously registered handler.

        Args:
            topic: Event topic.
            handler: The exact handler callable to remove.

        Returns:
            None
        """
        handlers = self._handlers.get(topic, [])
        if handler in handlers:
            handlers.remove(handler)

    # ── Emission ────────────────────────────────────────────────────────────

    def emit(
        self,
        topic: str,
        *,
        data: dict[str, Any] | None = None,
        source: str = "unknown",
    ) -> None:
        """Emit an event synchronously to all matching handlers.

        Handlers are invoked in registration order.  Exceptions inside a
        handler are caught and logged so other handlers still execute.

        Args:
            topic: Event topic string.
            data: Optional payload dict.
            source: Module name for traceability.

        Returns:
            None
        """
        event = Event(topic=topic, data=data or {}, source=source)
        matched: list[Handler] = []

        for registered_topic, handlers in self._handlers.items():
            if registered_topic == topic or registered_topic == "*":
                matched.extend(handlers)

        if not matched:
            logger.trace(f"EventBus: no handlers for '{topic}'")
            return

        for handler in matched:
            try:
                handler(event)
            except Exception as exc:
                logger.exception(
                    f"EventBus: handler '{handler.__name__}' failed "
                    f"on topic '{topic}': {exc}"
                )


# ── Module-level singleton ─────────────────────────────────────────────────
bus = EventBus()
