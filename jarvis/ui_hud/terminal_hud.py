"""
jarvis.ui_hud.terminal_hud
~~~~~~~~~~~~~~~~~~~~~~~~~~~
Rich-powered terminal HUD for JARVIS.

Shows a live status panel with transcription, intent, execution result,
and system status without any external GUI dependency.

Usage::

    from jarvis.ui_hud.terminal_hud import TerminalHUD

    hud = TerminalHUD()
    hud.show_listening()
    hud.show_transcription("open Chrome and search SpaceX")
    hud.show_result(success=True, message="Opened Chrome")
"""

from __future__ import annotations

import time
from typing import Optional

from loguru import logger

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
    from rich import print as rprint

    _RICH = True
except ImportError:
    _RICH = False


class TerminalHUD:
    """Rich-based terminal display for JARVIS status updates.

    Falls back to plain print statements if ``rich`` is not installed.

    Attributes:
        show_json: Whether to print the raw intent JSON.
    """

    def __init__(self, show_json: bool = False) -> None:
        self.show_json = show_json
        if _RICH:
            self._console = Console()
        self._last_transcription = ""
        self._phone_connected = False
        self._startup_banner()

    # ── Public API ──────────────────────────────────────────────────────────

    def show_listening(self) -> None:
        """Display a listening indicator.

        Returns:
            None
        """
        self._print("[bold cyan]🎤  Listening …[/bold cyan]" if _RICH else "🎤  Listening …")

    def show_wake_word(self) -> None:
        """Indicate wake word detected.

        Returns:
            None
        """
        self._print(
            "[bold yellow]👂  Wake word detected! Speak your command …[/bold yellow]"
            if _RICH
            else "👂  Wake word detected!"
        )

    def show_transcription(self, text: str) -> None:
        """Display the transcribed speech.

        Args:
            text: Transcribed text string.

        Returns:
            None
        """
        self._last_transcription = text
        if _RICH:
            self._console.print(
                Panel(f"[white]{text}[/white]", title="[blue]You said", border_style="blue")
            )
        else:
            print(f"You said: {text}")

    def show_intent(self, intent: str, target: str, action: str, confidence: float) -> None:
        """Display parsed intent details.

        Args:
            intent: Intent type string.
            target: Target device string.
            action: Action identifier.
            confidence: Confidence score (0.0-1.0).

        Returns:
            None
        """
        if _RICH:
            table = Table(show_header=False, box=None, padding=(0, 1))
            table.add_column("Key", style="dim")
            table.add_column("Value", style="cyan")
            table.add_row("Intent", intent)
            table.add_row("Target", target)
            table.add_row("Action", action)
            table.add_row("Confidence", f"{confidence:.0%}")
            self._console.print(Panel(table, title="[green]Intent Parsed", border_style="green"))
        else:
            print(f"Intent: {intent} | Target: {target} | Action: {action} | {confidence:.0%}")

    def show_confirmation_prompt(self, message: str) -> bool:
        """Ask user to confirm a high-risk action.

        Args:
            message: Description of what will be executed.

        Returns:
            True if user confirmed, False if cancelled.
        """
        if _RICH:
            self._console.print(
                Panel(
                    f"[bold red]{message}[/bold red]\n"
                    "[yellow]Say 'yes' or 'confirm' to proceed, "
                    "or 'no' to cancel.[/yellow]",
                    title="⚠  CONFIRMATION REQUIRED",
                    border_style="red",
                )
            )
        else:
            print(f"⚠  CONFIRM: {message}")

        # For terminal mode: read input (voice confirmation handled by orchestrator)
        try:
            answer = input("Confirm? [y/N]: ").strip().lower()
            return answer in ("y", "yes", "confirm")
        except (EOFError, KeyboardInterrupt):
            return False

    def show_result(self, success: bool, message: str, data: Optional[dict] = None) -> None:
        """Display the execution result.

        Args:
            success: Whether the action succeeded.
            message: Human-readable result message.
            data: Optional extra data dict to display.

        Returns:
            None
        """
        icon = "✅" if success else "❌"
        style = "green" if success else "red"
        if _RICH:
            content = f"{icon} {message}"
            if data:
                for k, v in data.items():
                    content += f"\n  [dim]{k}:[/dim] {v}"
            self._console.print(
                Panel(content, title=f"[{style}]Result", border_style=style)
            )
        else:
            print(f"{icon} {message}")

    def show_error(self, message: str) -> None:
        """Display an error message.

        Args:
            message: Error description.

        Returns:
            None
        """
        if _RICH:
            self._console.print(f"[bold red]⚠  ERROR:[/bold red] {message}")
        else:
            print(f"ERROR: {message}")

    def set_phone_status(self, connected: bool) -> None:
        """Update the phone connection status indicator.

        Args:
            connected: True if phone is ADB-connected.

        Returns:
            None
        """
        self._phone_connected = connected
        status = "🟢 Connected" if connected else "🔴 Disconnected"
        if _RICH:
            self._console.print(f"[dim]Phone: {status}[/dim]")
        else:
            print(f"Phone: {status}")

    # ── Private ─────────────────────────────────────────────────────────────

    def _startup_banner(self) -> None:
        if _RICH:
            self._console.print(
                Panel(
                    "[bold cyan]  ██╗ █████╗ ██████╗ ██╗   ██╗██╗███████╗\n"
                    "  ██║██╔══██╗██╔══██╗██║   ██║██║██╔════╝\n"
                    "  ██║███████║██████╔╝██║   ██║██║███████╗\n"
                    "  ██║██╔══██║██╔══██╗╚██╗ ██╔╝██║╚════██║\n"
                    "  ██║██║  ██║██║  ██║ ╚████╔╝ ██║███████║\n"
                    "  ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝  ╚═══╝  ╚═╝╚══════╝[/bold cyan]\n"
                    "[dim]Just A Rather Very Intelligent System[/dim]\n"
                    "[yellow]Local AI Assistant  |  Voice + Phone Control[/yellow]",
                    title="Welcome",
                    border_style="cyan",
                )
            )
        else:
            print("=" * 50)
            print("  JARVIS – Local AI Assistant")
            print("  Voice + Phone Control")
            print("=" * 50)

    def _print(self, message: str) -> None:
        if _RICH:
            self._console.print(message)
        else:
            # Strip basic rich markup
            import re
            plain = re.sub(r"\[/?[^\]]+\]", "", message)
            print(plain)
