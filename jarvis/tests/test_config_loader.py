"""
tests/test_config_loader.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Unit tests for the ConfigLoader singleton.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from jarvis.core.config_loader import ConfigLoader


FIXTURE_YAML = """\
jarvis:
  name: "TestBot"
audio:
  sample_rate: 16000
  stt:
    model_size: "tiny.en"
brain:
  backend: "ollama"
  model: "llama3.2"
logging:
  level: "DEBUG"
phone:
  serial: ""
"""


@pytest.fixture
def tmp_config(tmp_path: Path) -> Path:
    cfg_file = tmp_path / "test_settings.yaml"
    cfg_file.write_text(FIXTURE_YAML, encoding="utf-8")
    return cfg_file


def test_get_nested_key(tmp_config: Path) -> None:
    loader = ConfigLoader(config_path=tmp_config)
    assert loader.get("audio.stt.model_size") == "tiny.en"


def test_get_default_when_missing(tmp_config: Path) -> None:
    loader = ConfigLoader(config_path=tmp_config)
    assert loader.get("nonexistent.key", default="fallback") == "fallback"


def test_section(tmp_config: Path) -> None:
    loader = ConfigLoader(config_path=tmp_config)
    section = loader.section("brain")
    assert section["backend"] == "ollama"
    assert section["model"] == "llama3.2"


def test_env_override(tmp_config: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("JARVIS_LOG_LEVEL", "WARNING")
    loader = ConfigLoader(config_path=tmp_config)
    assert loader.get("logging.level") == "WARNING"


def test_missing_file_raises() -> None:
    with pytest.raises(FileNotFoundError):
        ConfigLoader(config_path=Path("/nonexistent/settings.yaml"))


def test_reload(tmp_config: Path) -> None:
    loader = ConfigLoader(config_path=tmp_config)
    assert loader.get("audio.sample_rate") == 16000
    # Modify file
    content = tmp_config.read_text()
    tmp_config.write_text(content.replace("16000", "22050"))
    loader.reload()
    assert loader.get("audio.sample_rate") == 22050
