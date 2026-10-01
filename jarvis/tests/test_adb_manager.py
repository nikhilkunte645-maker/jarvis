"""
jarvis/tests/test_adb_manager.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Unit tests for ADBManager using mocked subprocess calls.
"""

from __future__ import annotations

from subprocess import CompletedProcess
from unittest.mock import MagicMock, patch

import pytest

from jarvis.phone_bridge.adb_manager import ADBError, ADBManager


@pytest.fixture
def adb() -> ADBManager:
    mgr = ADBManager(serial="test123", adb_path="adb")
    mgr.connected = True
    return mgr


def _cp(stdout: str, rc: int = 0) -> CompletedProcess:
    return CompletedProcess(args=[], returncode=rc, stdout=stdout, stderr="")


def test_connect_auto_selects_first_device() -> None:
    mgr = ADBManager(adb_path="adb")
    with patch.object(mgr, "_run_raw", return_value=_cp("List of devices attached\nabc123\tdevice\n")):
        result = mgr.connect()
    assert result is True
    assert mgr.serial == "abc123"
    assert mgr.connected is True


def test_connect_no_devices_returns_false() -> None:
    mgr = ADBManager(adb_path="adb")
    with patch.object(mgr, "_run_raw", return_value=_cp("List of devices attached\n")):
        result = mgr.connect()
    assert result is False


def test_shell_requires_connected() -> None:
    mgr = ADBManager()
    mgr.connected = False
    with pytest.raises(ADBError, match="No device connected"):
        mgr.shell("echo hello")


def test_tap_calls_correct_shell(adb: ADBManager) -> None:
    with patch.object(adb, "_run", return_value=_cp("")) as mock_run:
        adb.tap(500, 1000)
        args = mock_run.call_args[0][0]
        assert "input tap 500 1000" in " ".join(args)


def test_swipe_calls_correct_shell(adb: ADBManager) -> None:
    with patch.object(adb, "_run", return_value=_cp("")) as mock_run:
        adb.swipe(100, 200, 300, 400, 500)
        args = mock_run.call_args[0][0]
        assert "input swipe 100 200 300 400 500" in " ".join(args)


def test_input_text_escapes_spaces(adb: ADBManager) -> None:
    with patch.object(adb, "_run", return_value=_cp("")) as mock_run:
        adb.input_text("hello world")
        args = mock_run.call_args[0][0]
        assert "%s" in " ".join(args)


def test_get_screen_size_parses_output(adb: ADBManager) -> None:
    with patch.object(adb, "_run", return_value=_cp("Physical size: 1080x2340")):
        w, h = adb.get_screen_size()
    assert w == 1080
    assert h == 2340


def test_get_screen_size_fallback_on_bad_output(adb: ADBManager) -> None:
    with patch.object(adb, "_run", return_value=_cp("garbage")):
        w, h = adb.get_screen_size()
    assert (w, h) == (1080, 1920)


def test_launch_app_calls_monkey(adb: ADBManager) -> None:
    with patch.object(adb, "_run", return_value=_cp("")) as mock_run:
        adb.launch_app("com.whatsapp")
        args = mock_run.call_args[0][0]
        assert "monkey" in " ".join(args)
        assert "com.whatsapp" in " ".join(args)


def test_shell_raises_on_nonzero(adb: ADBManager) -> None:
    with patch.object(adb, "_run", return_value=_cp("err", rc=1)) as mock_run:
        # _run raises ADBError on non-zero rc
        mock_run.side_effect = ADBError("command failed")
        with pytest.raises(ADBError):
            adb.shell("bad_command")
