"""
jarvis.phone_bridge.executor
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
High-level phone command executor.

Translates ``JarvisCommand`` actions into ADB calls (and optionally
uiautomator2 calls for element-based interactions).

Usage::

    from jarvis.phone_bridge.executor import PhoneExecutor

    phone = PhoneExecutor()
    if phone.connect():
        result = phone.run(command)
"""

from __future__ import annotations

import time
from typing import Optional

from loguru import logger

from jarvis.core.config_loader import cfg
from jarvis.core.schemas import ExecutionResult, JarvisCommand, TargetDevice
from jarvis.phone_bridge.adb_manager import ADBError, ADBManager


# Known app packages for common apps (user can extend in config)
_APP_PACKAGES: dict[str, str] = {
    "whatsapp": "com.whatsapp",
    "instagram": "com.instagram.android",
    "youtube": "com.google.android.youtube",
    "chrome": "com.android.chrome",
    "settings": "com.android.settings",
    "camera": "com.android.camera2",
    "gmail": "com.google.android.gm",
    "maps": "com.google.android.apps.maps",
    "spotify": "com.spotify.music",
    "netflix": "com.netflix.mediaclient",
    "twitter": "com.twitter.android",
    "telegram": "org.telegram.messenger",
    "facebook": "com.facebook.katana",
    "snapchat": "com.snapchat.android",
    "tiktok": "com.zhiliaoapp.musically",
    "phone": "com.google.android.dialer",
    "messages": "com.google.android.apps.messaging",
    "contacts": "com.google.android.contacts",
    "calculator": "com.google.android.calculator",
    "calendar": "com.google.android.calendar",
    "clock": "com.google.android.deskclock",
    "files": "com.google.android.documentsui",
    "play store": "com.android.vending",
}


class PhoneExecutor:
    """Routes phone-targeted JarvisCommands to ADB / uiautomator2.

    Gracefully degrades if the phone is disconnected.

    Attributes:
        adb: The underlying ADB manager instance.
        available: Whether a phone is connected and ready.
    """

    def __init__(self, adb: Optional[ADBManager] = None) -> None:
        self.adb: ADBManager = adb or ADBManager()
        self.available: bool = False
        self._u2 = None  # uiautomator2 device handle (lazy)

    # ── Public API ──────────────────────────────────────────────────────────

    def connect(self) -> bool:
        """Attempt to connect to an Android device via ADB.

        Returns:
            True if a device is ready.
        """
        if not cfg.get("phone.enabled", default=True):
            logger.info("Phone bridge disabled in config.")
            return False

        self.available = self.adb.connect()
        if self.available and cfg.get("phone.uiautomator2.enabled", default=True):
            self._init_u2()
        return self.available

    def run(self, command: JarvisCommand) -> ExecutionResult:
        """Dispatch a phone command to the correct handler.

        Args:
            command: Validated :class:`JarvisCommand` with target phone/both.

        Returns:
            :class:`ExecutionResult` describing success/failure.
        """
        if not self.available:
            return ExecutionResult(
                success=False,
                message="Phone not connected.",
                device=TargetDevice.PHONE,
            )

        action = command.action
        params = command.parameters
        pkg = command.app_package
        logger.info(f"PhoneExecutor: action='{action}'  params={params}")

        dispatch = {
            "launch_app": self._launch_app,
            "whatsapp_send": self._whatsapp_send,
            "sms_send": self._sms_send,
            "take_screenshot": self._take_screenshot,
            "tap": self._tap,
            "swipe": self._swipe,
            "type_text": self._type_text,
            "press_home": self._press_home,
            "press_back": self._press_back,
            "press_recents": self._press_recents,
            "toggle_wifi": self._toggle_wifi,
            "toggle_bluetooth": self._toggle_bluetooth,
            "toggle_airplane": self._toggle_airplane,
            "toggle_dnd": self._toggle_dnd,
            "open_notifications": self._open_notifications,
            "set_volume": self._set_volume,
            "force_stop": self._force_stop,
            "open_url": self._open_url,
            "call": self._make_call,
        }

        handler = dispatch.get(action)
        if handler is None:
            # Generic fallback: try to match on intent type
            return ExecutionResult(
                success=False,
                message=f"Unknown phone action: '{action}'",
                device=TargetDevice.PHONE,
            )

        try:
            return handler(params, pkg)
        except ADBError as exc:
            logger.error(f"ADB error in '{action}': {exc}")
            self.available = False
            return ExecutionResult(
                success=False,
                message=f"ADB error: {exc}",
                device=TargetDevice.PHONE,
            )
        except Exception as exc:
            logger.exception(f"PhoneExecutor error in '{action}': {exc}")
            return ExecutionResult(
                success=False,
                message=str(exc),
                device=TargetDevice.PHONE,
            )

    # ── Action handlers ─────────────────────────────────────────────────────

    def _launch_app(self, params: dict, pkg: Optional[str]) -> ExecutionResult:
        app_name: str = params.get("app_name", "")
        package = pkg or _APP_PACKAGES.get(app_name.lower(), "")
        if not package:
            return ExecutionResult(
                success=False,
                message=f"Unknown app: '{app_name}'. Provide app_package.",
                device=TargetDevice.PHONE,
            )
        self.adb.launch_app(package)
        return ExecutionResult(
            success=True,
            message=f"Launched {app_name or package}",
            device=TargetDevice.PHONE,
        )

    def _whatsapp_send(self, params: dict, _: Optional[str]) -> ExecutionResult:
        """Send a WhatsApp message using deep-link intent.

        Args:
            params: Must contain ``recipient`` and ``message``.

        Returns:
            ExecutionResult
        """
        recipient: str = params.get("recipient", "")
        message: str = params.get("message", "")

        # Try phone-number deep link first; otherwise search in contacts
        if recipient.lstrip("+").isdigit():
            phone_num = recipient.lstrip("+")
            self.adb.shell(
                f"am start -a android.intent.action.VIEW "
                f'-d "https://api.whatsapp.com/send?phone={phone_num}&text={message}"'
            )
        else:
            # Open WhatsApp search
            self.adb.launch_app("com.whatsapp")
            time.sleep(2)
            # Tap search button (coordinates vary by device – use u2 if available)
            if self._u2:
                try:
                    self._u2(resourceId="com.whatsapp:id/search_btn").click()
                    time.sleep(0.5)
                    self._u2(resourceId="com.whatsapp:id/search_src_text").set_text(recipient)
                    time.sleep(1)
                    self._u2(text=recipient).click()
                    time.sleep(1)
                    self._u2(resourceId="com.whatsapp:id/entry").set_text(message)
                    time.sleep(0.3)
                    self._u2(resourceId="com.whatsapp:id/send").click()
                    return ExecutionResult(
                        success=True,
                        message=f"WhatsApp message sent to {recipient}",
                        device=TargetDevice.PHONE,
                    )
                except Exception as exc:
                    logger.warning(f"u2 WhatsApp flow failed: {exc}")

        return ExecutionResult(
            success=True,
            message=f"WhatsApp opened for {recipient}. Please review and send.",
            device=TargetDevice.PHONE,
        )

    def _sms_send(self, params: dict, _: Optional[str]) -> ExecutionResult:
        recipient: str = params.get("recipient", "")
        message: str = params.get("message", "")
        self.adb.shell(
            f'am start -a android.intent.action.SENDTO '
            f'-d "sms:{recipient}" --es sms_body "{message}" --ez exit_on_sent true'
        )
        return ExecutionResult(
            success=True,
            message=f"SMS composer opened for {recipient}",
            device=TargetDevice.PHONE,
        )

    def _take_screenshot(self, params: dict, _: Optional[str]) -> ExecutionResult:
        path = self.adb.take_screenshot()
        return ExecutionResult(
            success=True,
            message=f"Phone screenshot saved: {path}",
            data={"path": path},
            device=TargetDevice.PHONE,
        )

    def _tap(self, params: dict, _: Optional[str]) -> ExecutionResult:
        x = int(params.get("x", 540))
        y = int(params.get("y", 960))
        self.adb.tap(x, y)
        return ExecutionResult(success=True, message=f"Tapped ({x},{y})", device=TargetDevice.PHONE)

    def _swipe(self, params: dict, _: Optional[str]) -> ExecutionResult:
        direction: str = params.get("direction", "up")
        w, h = self.adb.get_screen_size()
        cx, cy = w // 2, h // 2
        offsets = {
            "up": (cx, int(h * 0.7), cx, int(h * 0.3)),
            "down": (cx, int(h * 0.3), cx, int(h * 0.7)),
            "left": (int(w * 0.8), cy, int(w * 0.2), cy),
            "right": (int(w * 0.2), cy, int(w * 0.8), cy),
        }
        coords = offsets.get(direction, offsets["up"])
        self.adb.swipe(*coords)
        return ExecutionResult(
            success=True, message=f"Swiped {direction}", device=TargetDevice.PHONE
        )

    def _type_text(self, params: dict, _: Optional[str]) -> ExecutionResult:
        text: str = params.get("text", "")
        self.adb.input_text(text)
        return ExecutionResult(success=True, message=f"Typed: '{text}'", device=TargetDevice.PHONE)

    def _press_home(self, _: dict, __: Optional[str]) -> ExecutionResult:
        self.adb.press_home()
        return ExecutionResult(success=True, message="Home pressed", device=TargetDevice.PHONE)

    def _press_back(self, _: dict, __: Optional[str]) -> ExecutionResult:
        self.adb.press_back()
        return ExecutionResult(success=True, message="Back pressed", device=TargetDevice.PHONE)

    def _press_recents(self, _: dict, __: Optional[str]) -> ExecutionResult:
        self.adb.press_recents()
        return ExecutionResult(success=True, message="Recents opened", device=TargetDevice.PHONE)

    def _toggle_wifi(self, params: dict, _: Optional[str]) -> ExecutionResult:
        enabled: bool = bool(params.get("enabled", True))
        self.adb.toggle_wifi(enabled)
        state = "enabled" if enabled else "disabled"
        return ExecutionResult(success=True, message=f"WiFi {state}", device=TargetDevice.PHONE)

    def _toggle_bluetooth(self, params: dict, _: Optional[str]) -> ExecutionResult:
        enabled: bool = bool(params.get("enabled", True))
        self.adb.toggle_bluetooth(enabled)
        state = "enabled" if enabled else "disabled"
        return ExecutionResult(success=True, message=f"Bluetooth {state}", device=TargetDevice.PHONE)

    def _toggle_airplane(self, params: dict, _: Optional[str]) -> ExecutionResult:
        enabled: bool = bool(params.get("enabled", True))
        self.adb.set_airplane_mode(enabled)
        state = "on" if enabled else "off"
        return ExecutionResult(success=True, message=f"Airplane mode {state}", device=TargetDevice.PHONE)

    def _toggle_dnd(self, params: dict, _: Optional[str]) -> ExecutionResult:
        enabled: bool = bool(params.get("enabled", True))
        self.adb.set_dnd(enabled)
        state = "on" if enabled else "off"
        return ExecutionResult(success=True, message=f"DND {state}", device=TargetDevice.PHONE)

    def _open_notifications(self, _: dict, __: Optional[str]) -> ExecutionResult:
        self.adb.open_notifications()
        return ExecutionResult(success=True, message="Notifications opened", device=TargetDevice.PHONE)

    def _set_volume(self, params: dict, _: Optional[str]) -> ExecutionResult:
        level: int = int(params.get("level", 8))
        self.adb.set_volume(stream=3, level=level)
        return ExecutionResult(success=True, message=f"Phone volume set to {level}", device=TargetDevice.PHONE)

    def _force_stop(self, params: dict, pkg: Optional[str]) -> ExecutionResult:
        package = pkg or params.get("package", "")
        if package:
            self.adb.force_stop(package)
        return ExecutionResult(success=True, message=f"Force stopped {package}", device=TargetDevice.PHONE)

    def _open_url(self, params: dict, _: Optional[str]) -> ExecutionResult:
        url: str = params.get("url", "")
        self.adb.shell(f'am start -a android.intent.action.VIEW -d "{url}"')
        return ExecutionResult(success=True, message=f"Opened URL: {url}", device=TargetDevice.PHONE)

    def _make_call(self, params: dict, _: Optional[str]) -> ExecutionResult:
        number: str = params.get("number", params.get("recipient", ""))
        self.adb.shell(f'am start -a android.intent.action.CALL -d "tel:{number}"')
        return ExecutionResult(
            success=True, message=f"Calling {number}", device=TargetDevice.PHONE
        )

    # ── uiautomator2 init ───────────────────────────────────────────────────

    def _init_u2(self) -> None:
        try:
            import uiautomator2 as u2  # type: ignore

            serial = self.adb.serial or None
            self._u2 = u2.connect(serial)
            logger.info("uiautomator2 connected.")
        except ImportError:
            logger.warning("uiautomator2 not installed – element-based UI disabled.")
            self._u2 = None
        except Exception as exc:
            logger.warning(f"uiautomator2 init failed: {exc}")
            self._u2 = None
