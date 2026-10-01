"""
jarvis.laptop_bridge.executor
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Desktop automation executor: routes ``JarvisCommand`` actions to the
correct OS-level helpers using PyAutoGUI, keyboard, and platform APIs.

Cross-platform support:
* Windows  – PowerShell + win32api (optional) + pyautogui
* macOS    – AppleScript via ``osascript`` + pyautogui
* Linux    – xdotool / wmctrl + pyautogui

Usage::

    from jarvis.laptop_bridge.executor import LaptopExecutor

    exec = LaptopExecutor()
    result = exec.run(command)
"""

from __future__ import annotations

import platform
import subprocess
import time
from pathlib import Path
from typing import Optional

from loguru import logger

from jarvis.core.schemas import ExecutionResult, JarvisCommand, TargetDevice

_OS = platform.system()  # "Windows" | "Darwin" | "Linux"


class LaptopExecutor:
    """Routes laptop-targeted commands to the appropriate OS helper.

    Attributes:
        screenshot_dir: Directory where screenshots are saved.
    """

    def __init__(self, screenshot_dir: Optional[str] = None) -> None:
        from jarvis.core.config_loader import cfg

        self.screenshot_dir = Path(
            screenshot_dir or cfg.get("laptop.screenshot_dir", default="~/Desktop")
        ).expanduser()
        self.screenshot_dir.mkdir(parents=True, exist_ok=True)

    # ── Public API ──────────────────────────────────────────────────────────

    def run(self, command: JarvisCommand) -> ExecutionResult:
        """Dispatch a command to the appropriate handler.

        Args:
            command: Validated :class:`JarvisCommand` with target laptop/both.

        Returns:
            :class:`ExecutionResult` describing success/failure.
        """
        action = command.action
        intent = command.intent
        params = command.parameters
        logger.info(f"LaptopExecutor: intent='{intent}' action='{action}' params={params}")

        dispatch = {
            "open_app": self._open_app,
            "open": self._open_app,
            "close_app": self._close_app,
            "close": self._close_app,
            "search_web": self._search_web,
            "type_text": self._type_text,
            "take_screenshot": self._take_screenshot,
            "set_volume": self._set_volume,
            "set_brightness": self._set_brightness,
            "press_key": self._press_key,
            "move_mouse": self._move_mouse,
            "click": self._click,
            "open_file": self._open_file,
            "lock_screen": self._lock_screen,
            "browser_navigate": self._browser_navigate,
            "media_play_pause": self._media_play_pause,
            "media_next": self._media_next,
            "media_prev": self._media_prev,
        }

        handler = dispatch.get(action) or dispatch.get(intent)
        if handler is None:
            return ExecutionResult(
                success=False,
                message=f"Unknown laptop action: '{action}'",
                device=TargetDevice.LAPTOP,
            )

        try:
            return handler(params)
        except Exception as exc:
            logger.exception(f"LaptopExecutor error in '{action}': {exc}")
            return ExecutionResult(
                success=False,
                message=str(exc),
                device=TargetDevice.LAPTOP,
            )

    # ── Action handlers ─────────────────────────────────────────────────────

    def _open_app(self, params: dict) -> ExecutionResult:
        """Open an application by name.

        Args:
            params: Must contain ``app_name``.

        Returns:
            ExecutionResult
        """
        app_name: str = params.get("app") or params.get("app_name", "")
        if not app_name:
            return ExecutionResult(success=False, message="app_name required")

        try:
            if _OS == "Windows":
                subprocess.Popen(["start", app_name], shell=True)
            elif _OS == "Darwin":
                subprocess.Popen(["open", "-a", app_name])
            else:
                subprocess.Popen([app_name.lower().replace(" ", "-")])
            time.sleep(1.5)
            return ExecutionResult(
                success=True,
                message=f"Opened {app_name}",
                device=TargetDevice.LAPTOP,
            )
        except FileNotFoundError:
            return ExecutionResult(
                success=False,
                message=f"App not found: {app_name}",
                device=TargetDevice.LAPTOP,
            )

    def _close_app(self, params: dict) -> ExecutionResult:
        """Kill an application by name.

        Args:
            params: Must contain ``app_name``.

        Returns:
            ExecutionResult
        """
        app_name: str = params.get("app_name", "")
        if _OS == "Windows":
            subprocess.run(["taskkill", "/F", "/IM", f"{app_name}.exe"], shell=True)
        elif _OS == "Darwin":
            subprocess.run(["pkill", "-x", app_name])
        else:
            subprocess.run(["pkill", "-f", app_name])
        return ExecutionResult(success=True, message=f"Closed {app_name}", device=TargetDevice.LAPTOP)

    def _search_web(self, params: dict) -> ExecutionResult:
        """Open the default browser with a search query.

        Args:
            params: Must contain ``query``.

        Returns:
            ExecutionResult
        """
        import webbrowser

        query: str = params.get("query", "")
        url: str = params.get("url") or f"https://www.google.com/search?q={query.replace(' ', '+')}"
        webbrowser.open(url)
        return ExecutionResult(
            success=True, message=f"Searching: {query or url}", device=TargetDevice.LAPTOP
        )

    def _type_text(self, params: dict) -> ExecutionResult:
        """Type text at the current cursor position.

        Args:
            params: Must contain ``text``.

        Returns:
            ExecutionResult
        """
        import pyautogui  # type: ignore

        text: str = params.get("text", "")
        pyautogui.write(text, interval=0.03)
        return ExecutionResult(success=True, message=f"Typed: '{text}'", device=TargetDevice.LAPTOP)

    def _take_screenshot(self, params: dict) -> ExecutionResult:
        """Capture a screenshot and save it.

        Args:
            params: Optional ``filename`` override.

        Returns:
            ExecutionResult with ``data.path`` set to the file path.
        """
        import pyautogui  # type: ignore

        filename = params.get("filename") or f"jarvis_{int(time.time())}.png"
        path = self.screenshot_dir / filename
        img = pyautogui.screenshot()
        img.save(str(path))
        logger.info(f"Screenshot saved: {path}")
        return ExecutionResult(
            success=True,
            message=f"Screenshot saved to {path}",
            data={"path": str(path)},
            device=TargetDevice.LAPTOP,
        )

    def _set_volume(self, params: dict) -> ExecutionResult:
        """Set system volume to a percentage.

        Args:
            params: Must contain ``level`` (0-100).

        Returns:
            ExecutionResult
        """
        level: int = int(params.get("level", 50))
        level = max(0, min(100, level))

        if _OS == "Windows":
            from ctypes import cast, POINTER
            from comtypes import CLSCTX_ALL  # type: ignore
            from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume  # type: ignore

            devices = AudioUtilities.GetSpeakers()
            interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
            volume = cast(interface, POINTER(IAudioEndpointVolume))
            volume.SetMasterVolumeLevelScalar(level / 100.0, None)

        elif _OS == "Darwin":
            subprocess.run(["osascript", "-e", f"set volume output volume {level}"])

        else:
            subprocess.run(
                ["amixer", "-D", "pulse", "sset", "Master", f"{level}%"],
                capture_output=True,
            )

        return ExecutionResult(
            success=True, message=f"Volume set to {level}%", device=TargetDevice.LAPTOP
        )

    def _set_brightness(self, params: dict) -> ExecutionResult:
        """Set screen brightness.

        Args:
            params: Must contain ``level`` (0-100).

        Returns:
            ExecutionResult
        """
        level: int = int(params.get("level", 70))
        level = max(0, min(100, level))

        if _OS == "Windows":
            script = (
                f"(Get-WmiObject -Namespace root/WMI -Class "
                f"WmiMonitorBrightnessMethods).WmiSetBrightness(1,{level})"
            )
            subprocess.run(["powershell", "-Command", script], capture_output=True)
        elif _OS == "Darwin":
            subprocess.run(
                ["brightness", str(level / 100.0)], capture_output=True
            )
        else:
            subprocess.run(
                ["brightnessctl", "s", f"{level}%"], capture_output=True
            )

        return ExecutionResult(
            success=True, message=f"Brightness set to {level}%", device=TargetDevice.LAPTOP
        )

    def _press_key(self, params: dict) -> ExecutionResult:
        """Simulate a keyboard key press.

        Args:
            params: Must contain ``key`` (e.g. ``"ctrl+c"``).

        Returns:
            ExecutionResult
        """
        import pyautogui  # type: ignore

        key: str = params.get("key", "")
        keys = [k.strip() for k in key.split("+")]
        if len(keys) > 1:
            pyautogui.hotkey(*keys)
        else:
            pyautogui.press(key)
        return ExecutionResult(
            success=True, message=f"Pressed key: {key}", device=TargetDevice.LAPTOP
        )

    def _move_mouse(self, params: dict) -> ExecutionResult:
        """Move the mouse cursor to absolute coordinates.

        Args:
            params: Must contain ``x`` and ``y``.

        Returns:
            ExecutionResult
        """
        import pyautogui  # type: ignore

        x: int = int(params.get("x", 0))
        y: int = int(params.get("y", 0))
        pyautogui.moveTo(x, y, duration=0.3)
        return ExecutionResult(
            success=True, message=f"Mouse moved to ({x},{y})", device=TargetDevice.LAPTOP
        )

    def _click(self, params: dict) -> ExecutionResult:
        """Click at coordinates or current position.

        Args:
            params: Optional ``x``, ``y``, ``button`` (left/right/middle).

        Returns:
            ExecutionResult
        """
        import pyautogui  # type: ignore

        x = params.get("x")
        y = params.get("y")
        button: str = params.get("button", "left")
        if x is not None and y is not None:
            pyautogui.click(int(x), int(y), button=button)
        else:
            pyautogui.click(button=button)
        return ExecutionResult(
            success=True, message="Clicked", device=TargetDevice.LAPTOP
        )

    def _open_file(self, params: dict) -> ExecutionResult:
        """Open a file with its default application.

        Args:
            params: Must contain ``file_path``.

        Returns:
            ExecutionResult
        """
        import os

        path = Path(params.get("file_path", "")).expanduser()
        if not path.exists():
            return ExecutionResult(
                success=False,
                message=f"File not found: {path}",
                device=TargetDevice.LAPTOP,
            )
        os.startfile(str(path)) if _OS == "Windows" else subprocess.Popen(
            ["open" if _OS == "Darwin" else "xdg-open", str(path)]
        )
        return ExecutionResult(
            success=True, message=f"Opened {path}", device=TargetDevice.LAPTOP
        )

    def _lock_screen(self, _: dict) -> ExecutionResult:
        """Lock the workstation.

        Returns:
            ExecutionResult
        """
        if _OS == "Windows":
            import ctypes

            ctypes.windll.user32.LockWorkStation()  # type: ignore
        elif _OS == "Darwin":
            subprocess.run(
                ["/System/Library/CoreServices/Menu Extras/User.menu/Contents/Resources/CGSession", "-suspend"]
            )
        else:
            subprocess.run(["loginctl", "lock-session"])
        return ExecutionResult(success=True, message="Screen locked", device=TargetDevice.LAPTOP)

    def _browser_navigate(self, params: dict) -> ExecutionResult:
        """Navigate the browser to a URL.

        Args:
            params: Must contain ``url``.

        Returns:
            ExecutionResult
        """
        import webbrowser

        url: str = params.get("url", "")
        if not url.startswith("http"):
            url = "https://" + url
        webbrowser.open(url)
        return ExecutionResult(
            success=True, message=f"Navigated to {url}", device=TargetDevice.LAPTOP
        )

    def _media_play_pause(self, _: dict) -> ExecutionResult:
        """Toggle play/pause for the active media player.

        Returns:
            ExecutionResult
        """
        import pyautogui  # type: ignore

        pyautogui.press("playpause")
        return ExecutionResult(success=True, message="Play/Pause toggled", device=TargetDevice.LAPTOP)

    def _media_next(self, _: dict) -> ExecutionResult:
        """Skip to next track.

        Returns:
            ExecutionResult
        """
        import pyautogui  # type: ignore

        pyautogui.press("nexttrack")
        return ExecutionResult(success=True, message="Next track", device=TargetDevice.LAPTOP)

    def _media_prev(self, _: dict) -> ExecutionResult:
        """Go to previous track.

        Returns:
            ExecutionResult
        """
        import pyautogui  # type: ignore

        pyautogui.press("prevtrack")
        return ExecutionResult(success=True, message="Previous track", device=TargetDevice.LAPTOP)
