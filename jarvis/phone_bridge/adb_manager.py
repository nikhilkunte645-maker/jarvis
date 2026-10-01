"""
jarvis.phone_bridge.adb_manager
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Low-level ADB command wrapper.

Provides a thin Python API over ``adb shell`` / ``adb`` commands.
All methods raise :class:`ADBError` on failure so callers can handle
device disconnects gracefully.

Usage::

    from jarvis.phone_bridge.adb_manager import ADBManager

    adb = ADBManager()
    adb.connect()
    adb.tap(500, 1000)
    adb.press_home()
"""

from __future__ import annotations

import shutil
import subprocess
import time
from pathlib import Path
from typing import Optional

from loguru import logger

from jarvis.core.config_loader import cfg


class ADBError(Exception):
    """Raised when an ADB command fails."""


class ADBManager:
    """Manages ADB connections and raw command execution.

    Attributes:
        serial: Device serial number (``""`` = auto-detect).
        adb_path: Path to the ``adb`` binary.
        connected: Whether a device is currently reachable.
    """

    def __init__(
        self,
        serial: Optional[str] = None,
        adb_path: Optional[str] = None,
    ) -> None:
        self.serial: str = serial or cfg.get("phone.serial", default="")
        self.adb_path: str = adb_path or cfg.get("phone.adb_path", default="adb")
        self.connected: bool = False
        self._wifi_host: str = cfg.get("phone.wifi_host", default="")
        self._mode: str = cfg.get("phone.connection_mode", default="usb")

    # ── Connection ──────────────────────────────────────────────────────────

    def connect(self) -> bool:
        """Detect or connect to an Android device.

        For USB mode: auto-selects the first connected device.
        For WiFi mode: runs ``adb connect <host>``.

        Returns:
            True if a device is now reachable.
        """
        if self._mode == "wifi" and self._wifi_host:
            logger.info(f"ADB: connecting over WiFi to {self._wifi_host}")
            self._run_raw(["adb", "connect", self._wifi_host])
            time.sleep(1)

        devices = self._list_devices()
        if not devices:
            logger.warning("ADB: no devices found.")
            self.connected = False
            return False

        if not self.serial:
            self.serial = devices[0]
            logger.info(f"ADB: auto-selected device '{self.serial}'")

        if self.serial not in devices:
            logger.warning(f"ADB: serial '{self.serial}' not in device list {devices}")
            self.connected = False
            return False

        self.connected = True
        logger.info(f"ADB: connected to '{self.serial}'")
        return True

    def disconnect(self) -> None:
        """Disconnect WiFi ADB (no-op for USB).

        Returns:
            None
        """
        if self._mode == "wifi" and self._wifi_host:
            self._run_raw(["adb", "disconnect", self._wifi_host])
        self.connected = False

    # ── Shell commands ──────────────────────────────────────────────────────

    def shell(self, command: str, check: bool = True) -> str:
        """Run an ``adb shell`` command and return stdout.

        Args:
            command: Shell command string to execute on the device.
            check: Raise :class:`ADBError` if command exits non-zero.

        Returns:
            Decoded stdout string (trailing whitespace stripped).

        Raises:
            ADBError: If the command fails and *check* is True.
        """
        self._require_connected()
        args = self._base_args() + ["shell", command]
        result = self._run(args, check=check)
        return result.stdout.strip()

    def pull(self, remote: str, local: str) -> None:
        """Pull a file from device to local disk.

        Args:
            remote: Path on the Android device.
            local: Destination path on the host.

        Returns:
            None
        """
        self._require_connected()
        self._run(self._base_args() + ["pull", remote, local])
        logger.info(f"ADB pull: {remote} → {local}")

    def push(self, local: str, remote: str) -> None:
        """Push a local file to the device.

        Args:
            local: Source path on the host.
            remote: Destination path on the Android device.

        Returns:
            None
        """
        self._require_connected()
        self._run(self._base_args() + ["push", local, remote])
        logger.info(f"ADB push: {local} → {remote}")

    # ── Input actions ───────────────────────────────────────────────────────

    def tap(self, x: int, y: int) -> None:
        """Simulate a tap at screen coordinates.

        Args:
            x: X coordinate in pixels.
            y: Y coordinate in pixels.

        Returns:
            None
        """
        self.shell(f"input tap {x} {y}")

    def swipe(
        self,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        duration_ms: int = 300,
    ) -> None:
        """Simulate a swipe gesture.

        Args:
            x1: Start X coordinate.
            y1: Start Y coordinate.
            x2: End X coordinate.
            y2: End Y coordinate.
            duration_ms: Swipe duration in milliseconds.

        Returns:
            None
        """
        self.shell(f"input swipe {x1} {y1} {x2} {y2} {duration_ms}")

    def input_text(self, text: str) -> None:
        """Type text on the device (escapes spaces).

        Args:
            text: Text string to type.

        Returns:
            None
        """
        escaped = text.replace(" ", "%s").replace("'", "\\'")
        self.shell(f"input text '{escaped}'")

    def press_key(self, keycode: int | str) -> None:
        """Send a keyevent to the device.

        Args:
            keycode: Android keycode integer or name (e.g. ``3`` or ``"KEYCODE_HOME"``).

        Returns:
            None
        """
        self.shell(f"input keyevent {keycode}")

    def press_home(self) -> None:
        """Press the Home button.

        Returns:
            None
        """
        self.press_key("KEYCODE_HOME")

    def press_back(self) -> None:
        """Press the Back button.

        Returns:
            None
        """
        self.press_key("KEYCODE_BACK")

    def press_recents(self) -> None:
        """Open the Recents / App Switcher.

        Returns:
            None
        """
        self.press_key("KEYCODE_APP_SWITCH")

    # ── App management ──────────────────────────────────────────────────────

    def launch_app(self, package: str, activity: Optional[str] = None) -> None:
        """Launch an app by package name (and optional activity).

        Args:
            package: Android package name, e.g. ``"com.whatsapp"``.
            activity: Optional full activity name. If omitted, uses the
                      launcher activity.

        Returns:
            None
        """
        if activity:
            self.shell(f"am start -n {package}/{activity}")
        else:
            self.shell(
                f"monkey -p {package} -c android.intent.category.LAUNCHER 1"
            )
        time.sleep(1.5)

    def force_stop(self, package: str) -> None:
        """Force-stop an application.

        Args:
            package: Android package name.

        Returns:
            None
        """
        self.shell(f"am force-stop {package}")

    def take_screenshot(self, remote_path: Optional[str] = None) -> str:
        """Capture a device screenshot and pull it to the host.

        Args:
            remote_path: Optional device-side path. Defaults to a temp path.

        Returns:
            Local path where the screenshot was saved.
        """
        from jarvis.core.config_loader import cfg

        remote = remote_path or "/sdcard/jarvis_screen.png"
        pull_dir = Path(
            cfg.get("phone.pull_dir", default="~/Desktop/jarvis_phone")
        ).expanduser()
        pull_dir.mkdir(parents=True, exist_ok=True)

        local = str(pull_dir / f"phone_{int(time.time())}.png")
        self.shell(f"screencap -p {remote}")
        self.pull(remote, local)
        logger.info(f"Phone screenshot saved: {local}")
        return local

    # ── System toggles ──────────────────────────────────────────────────────

    def toggle_wifi(self, enabled: bool) -> None:
        """Enable or disable WiFi.

        Args:
            enabled: True to enable, False to disable.

        Returns:
            None
        """
        state = "enable" if enabled else "disable"
        self.shell(f"svc wifi {state}")

    def toggle_bluetooth(self, enabled: bool) -> None:
        """Enable or disable Bluetooth.

        Args:
            enabled: True to enable, False to disable.

        Returns:
            None
        """
        value = "true" if enabled else "false"
        self.shell(
            f"am start -a android.bluetooth.adapter.action.REQUEST_ENABLE"
            if enabled
            else f"am broadcast -a android.bluetooth.adapter.action.STATE_CHANGED"
        )

    def set_airplane_mode(self, enabled: bool) -> None:
        """Toggle airplane mode.

        Args:
            enabled: True to enable, False to disable.

        Returns:
            None
        """
        val = "1" if enabled else "0"
        self.shell(f"settings put global airplane_mode_on {val}")
        self.shell("am broadcast -a android.intent.action.AIRPLANE_MODE")

    def set_dnd(self, enabled: bool) -> None:
        """Toggle Do Not Disturb mode.

        Args:
            enabled: True to enable DND, False to disable.

        Returns:
            None
        """
        # DND requires notification policy access; uses settings workaround
        ringer_mode = "0" if enabled else "2"  # 0=silent, 2=normal
        self.shell(f"cmd notification set-interruption-filter {ringer_mode}")

    def open_notifications(self) -> None:
        """Open the notification shade.

        Returns:
            None
        """
        self.shell("cmd statusbar expand-notifications")

    def set_volume(self, stream: int, level: int) -> None:
        """Set device media volume.

        Args:
            stream: Android audio stream index (3 = STREAM_MUSIC).
            level: Volume level (device-specific max, usually 0-15).

        Returns:
            None
        """
        self.shell(f"media volume --stream {stream} --set {level}")

    # ── Device info ─────────────────────────────────────────────────────────

    def get_screen_size(self) -> tuple[int, int]:
        """Return the device display resolution.

        Returns:
            Tuple of (width, height) in pixels.
        """
        out = self.shell("wm size")
        # Output: "Physical size: 1080x2400"
        try:
            size_part = out.split(":")[-1].strip()
            w, h = size_part.split("x")
            return int(w), int(h)
        except Exception:
            return 1080, 1920  # sensible default

    def get_installed_packages(self) -> list[str]:
        """List all installed package names.

        Returns:
            List of package name strings.
        """
        out = self.shell("pm list packages -3")  # -3 = third-party only
        packages = []
        for line in out.splitlines():
            if line.startswith("package:"):
                packages.append(line[8:])
        return packages

    # ── Private helpers ─────────────────────────────────────────────────────

    def _base_args(self) -> list[str]:
        args = [self.adb_path]
        if self.serial:
            args += ["-s", self.serial]
        return args

    def _list_devices(self) -> list[str]:
        result = self._run_raw([self.adb_path, "devices"])
        lines = result.stdout.strip().splitlines()
        devices = []
        for line in lines[1:]:
            if "\t" in line:
                serial, state = line.split("\t", 1)
                if state.strip() == "device":
                    devices.append(serial.strip())
        return devices

    def _require_connected(self) -> None:
        if not self.connected:
            raise ADBError("No device connected. Call connect() first.")

    def _run(
        self, args: list[str], check: bool = True
    ) -> subprocess.CompletedProcess:
        try:
            result = subprocess.run(
                args,
                capture_output=True,
                text=True,
                timeout=15,
            )
            if check and result.returncode != 0:
                raise ADBError(
                    f"ADB command failed (rc={result.returncode}): "
                    f"{' '.join(args)}\nstderr: {result.stderr}"
                )
            return result
        except subprocess.TimeoutExpired as exc:
            raise ADBError(f"ADB command timed out: {' '.join(args)}") from exc

    def _run_raw(self, args: list[str]) -> subprocess.CompletedProcess:
        return subprocess.run(args, capture_output=True, text=True, timeout=10)
