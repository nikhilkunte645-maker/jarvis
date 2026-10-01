"""
jarvis.brain.llm_client
~~~~~~~~~~~~~~~~~~~~~~~~
Unified LLM client that dispatches to Ollama, OpenAI, or Anthropic.

The caller never needs to know which backend is active.  All backends
return a plain string (the model's response).

Usage::

    from jarvis.brain.llm_client import LLMClient

    client = LLMClient()
    response = client.complete(system=SYSTEM_PROMPT, user="open Chrome")
"""

from __future__ import annotations

from typing import Optional

from loguru import logger

from jarvis.core.config_loader import cfg


class LLMClient:
    """Thin adapter that routes LLM calls to the configured backend.

    Attributes:
        backend: ``"ollama"`` | ``"openai"`` | ``"anthropic"``.
        model: Model identifier string.
        temperature: Sampling temperature (low = deterministic).
        max_tokens: Maximum output tokens.
        timeout: Request timeout in seconds.
    """

    def __init__(
        self,
        backend: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        timeout: Optional[int] = None,
    ) -> None:
        self.backend: str = backend or cfg.get("brain.backend", default="ollama")
        self.model: str = model or cfg.get("brain.model", default="llama3.2")
        self.temperature: float = temperature if temperature is not None else cfg.get(
            "brain.temperature", default=0.1
        )
        self.max_tokens: int = max_tokens or cfg.get("brain.max_tokens", default=512)
        self.timeout: int = timeout or cfg.get("brain.timeout_seconds", default=30)

        logger.info(
            f"LLMClient initialised  backend={self.backend}  model={self.model}"
        )

    # ── Public API ──────────────────────────────────────────────────────────

    def complete(self, system: str, user: str) -> str:
        """Send a system+user prompt and return the model's raw text response.

        Args:
            system: System/context prompt (the JSON schema instructions).
            user: User message (the transcribed command + history).

        Returns:
            Raw string response from the model.

        Raises:
            RuntimeError: If the backend call fails after retries.
        """
        logger.debug(f"LLM request  backend={self.backend}  user='{user[:80]}…'")

        try:
            if self.backend == "ollama":
                return self._call_ollama(system, user)
            elif self.backend == "openai":
                return self._call_openai(system, user)
            elif self.backend == "anthropic":
                return self._call_anthropic(system, user)
            else:
                raise ValueError(f"Unknown LLM backend: {self.backend}")
        except Exception as exc:
            logger.error(f"LLM call failed: {exc}")
            raise RuntimeError(f"LLM backend '{self.backend}' error: {exc}") from exc

    # ── Backend implementations ─────────────────────────────────────────────

    def _call_ollama(self, system: str, user: str) -> str:
        """Call a local Ollama instance.

        Args:
            system: System prompt text.
            user: User message text.

        Returns:
            Model response string.
        """
        try:
            import ollama  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "ollama Python SDK not installed. Run: pip install ollama"
            ) from exc

        host = cfg.get("brain.ollama_host", default="http://localhost:11434")

        client = ollama.Client(host=host, timeout=self.timeout)
        response = client.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            options={
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
            },
        )
        return response["message"]["content"]

    def _call_openai(self, system: str, user: str) -> str:
        """Call OpenAI ChatCompletion API.

        Args:
            system: System prompt text.
            user: User message text.

        Returns:
            Model response string.
        """
        try:
            from openai import OpenAI  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "openai SDK not installed. Run: pip install openai"
            ) from exc

        import os

        api_key = cfg.get("brain.cloud.openai_api_key") or os.environ.get("OPENAI_API_KEY")
        openai_model = cfg.get("brain.cloud.openai_model", default="gpt-4o-mini")

        client = OpenAI(api_key=api_key, timeout=self.timeout)
        resp = client.chat.completions.create(
            model=openai_model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return resp.choices[0].message.content or ""

    def _call_anthropic(self, system: str, user: str) -> str:
        """Call Anthropic Claude API.

        Args:
            system: System prompt text.
            user: User message text.

        Returns:
            Model response string.
        """
        try:
            import anthropic  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "anthropic SDK not installed. Run: pip install anthropic"
            ) from exc

        import os

        api_key = cfg.get("brain.cloud.anthropic_api_key") or os.environ.get("ANTHROPIC_API_KEY")
        anthropic_model = cfg.get(
            "brain.cloud.anthropic_model", default="claude-3-haiku-20240307"
        )

        client = anthropic.Anthropic(api_key=api_key)
        message = client.messages.create(
            model=anthropic_model,
            max_tokens=self.max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return message.content[0].text if message.content else ""
