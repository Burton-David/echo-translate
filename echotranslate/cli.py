"""Command-line entry point for EchoTranslate.

Wires configuration and the terminal UI together: set the backend environment
variables, resolve and create the working directories, check (offline) which
translation packages are present, then run the menu. The network is only touched
later, at the moment a translation actually needs a package that is missing.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from rich.console import Console

from echotranslate import __version__
from echotranslate.config import configure_environment, default_settings, ensure_dirs
from echotranslate.errors import EchoTranslateError
from echotranslate.languages import translation_targets
from echotranslate.translation import missing_packages
from echotranslate.tui import TUI

_EPILOG = """\
Everything else happens in the menu: record a voice profile, translate and
hear it in your voice, compare your pitch against a clip, live translation,
and replaying saved clips.

files:
  ./voices/          voice profiles (<name>.wav), in the current directory
  ./output/<date>/   generated clips
  ~/.cache/whisper/  Whisper weights for live mode

Voice cloning and live mode need the 'voice' extra and about 2.5 GB of model
downloads on first use; translation alone needs neither.
"""


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="echotranslate",
        description=(
            "Hear English text spoken in another language in your own cloned\n"
            "voice. Runs locally; opens an interactive menu."
        ),
        epilog=_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the interactive application. Returns a process exit code."""
    # Parse before the TTY check so --help and --version work from a pipe or CI.
    _parser().parse_args(argv)
    console = Console()
    if not sys.stdin.isatty():
        console.print(
            "echotranslate opens an interactive menu and needs a terminal. "
            "Run 'echotranslate --help' for details."
        )
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
