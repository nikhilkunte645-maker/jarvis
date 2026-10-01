"""
jarvis.core.config_loader
~~~~~~~~~~~~~~~~~~~~~~~~~
Loads and validates the JARVIS YAML configuration file, merging any
environment-variable overrides from a .env file.

All other modules should import the singleton `cfg` rather than
reading files themselves:

    from jarvis.core.config_loader import cfg
    model_size = cfg.get("audio.stt.model_size", default="base.en")
"""

from __future__ import annotations

import os
from functools import reduce
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from loguru import logger

# ── Resolve paths ──────────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent.parent  # repo root
_DEFAULT_CONFIG = _ROOT / "jarvis" / "config" / "settings.yaml"

# Load .env so os.environ is populated before we read anything
load_dotenv(_ROOT / ".env", override=False)


class ConfigLoader:
    """Singleton wrapper around a YAML configuration file.

    Supports dot-notation key access and environment-variable overrides.

    Example:
        >>> cfg = ConfigLoader()
        >>> cfg.get("audio.stt.model_size", default="base.en")
        'base.en'
    """

    def __init__(self, config_path: Path | None = None) -> None:
        env_path = os.environ.get("JARVIS_CONFIG")
        self._path: Path = (
            Path(env_path) if env_path else (config_path or _DEFAULT_CONFIG)
        )
        self._data: dict[str, Any] = self._load()
        logger.debug(f"Config loaded from {self._path}")

    # ── Public API ──────────────────────────────────────────────────────────

    def get(self, dotted_key: str, *, default: Any = None) -> Any:
        """Retrieve a value by dot-notation key.

        Args:
            dotted_key: E.g. ``"audio.stt.model_size"``.
            default: Returned when key is absent.

        Returns:
            The config value or *default*.
        """
        try:
            return reduce(lambda d, k: d[k], dotted_key.split("."), self._data)
        except (KeyError, TypeError):
            return default

    def section(self, key: str) -> dict[str, Any]:
        """Return an entire top-level section as a dict.

        Args:
            key: Top-level YAML key (e.g. ``"audio"``).

        Returns:
            Dict of that section, or empty dict if absent.
        """
        return dict(self._data.get(key, {}))

    def all(self) -> dict[str, Any]:
        """Return the full configuration dictionary (read-only copy).

        Returns:
            Deep copy of the entire config.
        """
        import copy

        return copy.deepcopy(self._data)

    def reload(self) -> None:
        """Reload the YAML file from disk (useful for live updates).

        Returns:
            None
        """
        self._data = self._load()
        logger.info("Config reloaded.")

    # ── Private helpers ─────────────────────────────────────────────────────

    def _load(self) -> dict[str, Any]:
        if not self._path.exists():
            raise FileNotFoundError(
                f"JARVIS config file not found: {self._path}\n"
                "Copy jarvis/config/settings.yaml to configure JARVIS."
            )
        with open(self._path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        # Apply env-var overrides defined in .env.example
        data = self._apply_env_overrides(data)
        return data

    @staticmethod
    def _apply_env_overrides(data: dict[str, Any]) -> dict[str, Any]:
        """Patch specific config keys from environment variables.

        Args:
            data: The raw YAML data dict.

        Returns:
            Patched data dict.
        """
        overrides: dict[str, tuple[str, ...]] = {
            "JARVIS_LOG_LEVEL": ("logging", "level"),
            "JARVIS_PHONE_SERIAL": ("phone", "serial"),
            "OPENAI_API_KEY": ("brain", "cloud", "openai_api_key"),
            "ANTHROPIC_API_KEY": ("brain", "cloud", "anthropic_api_key"),
            "PORCUPINE_ACCESS_KEY": ("audio", "wake_word", "porcupine_access_key"),
        }
        for env_var, key_path in overrides.items():
            value = os.environ.get(env_var)
            if value:
                node = data
                for k in key_path[:-1]:
                    node = node.setdefault(k, {})
                node[key_path[-1]] = value
        return data


# ── Module-level singleton ─────────────────────────────────────────────────
cfg = ConfigLoader()
