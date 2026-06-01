"""Command-line entry point for EchoTranslate.

Wires configuration and the terminal UI together: set the backend environment
variables, resolve and create the working directories, check (offline) which
translation packages are present, then run the menu. The network is only touched
later, at the moment a translation actually needs a package that is missing.
"""

from __future__ import annotations

import sys

from rich.console import Console

from echotranslate.config import configure_environment, default_settings, ensure_dirs
from echotranslate.errors import EchoTranslateError
from echotranslate.languages import translation_targets
from echotranslate.translation import missing_packages
from echotranslate.tui import TUI


def main() -> int:
    """Run the interactive application. Returns a process exit code."""
    console = Console()
    if not sys.stdin.isatty():
        console.print("echotranslate needs an interactive terminal.")
        return 1

    configure_environment()
    settings = default_settings()
    ensure_dirs(settings)

    try:
        if missing_packages(translation_targets()):
            console.print(
                "[dim]Some translation languages aren't installed yet; "
                "they'll download the first time you use them.[/dim]"
            )
        TUI(console, settings).run()
    except KeyboardInterrupt:
        console.print("\nInterrupted.")
        return 0
    except EchoTranslateError as exc:
        console.print(f"[red]{exc}[/red]")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
