"""The rich terminal interface: menu, screens, and dispatch.

This module is presentation and orchestration only. Every prompt goes through the
rich ``Console``, so markup renders correctly and a test can drive the UI with a
mocked console. Each screen delegates real work to the typed modules (translation,
synthesis, transcription, audio); deliberate failures surface as
:class:`EchoTranslateError` and are rendered centrally in :meth:`TUI._dispatch`
as an actionable message.
"""

from __future__ import annotations

import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.prompt import IntPrompt, Prompt
from rich.table import Table

from echotranslate import translation
from echotranslate.audio import (
    detect_speech_segments,
    microphone_chunks,
    play,
    read_wav,
    record_until_enter,
    write_wav,
)
from echotranslate.config import (
    LIVE_SAMPLE_RATE,
    RECORD_SAMPLE_RATE,
    Settings,
    build_output_path,
)
from echotranslate.errors import EchoTranslateError
from echotranslate.languages import (
    LANGUAGES,
    Language,
    by_argos_code,
    by_menu_number,
    menu_choices,
)
from echotranslate.pitch import compare_contours, extract_f0, render_contours
from echotranslate.synthesis import VoiceSynthesizer
from echotranslate.transcription import SpeechTranscriber
from echotranslate.voices import list_voices, voice_path

_BANNER = "EchoTranslate: your voice, another language"

_MIN_PROFILE_SECONDS = 10.0


def _language_from_clip(path: Path) -> Language | None:
    """Infer a saved clip's target language from its ``..._<code>.wav`` name."""
    stem = path.stem
    code = stem.rsplit("_", 1)[-1] if "_" in stem else ""
    return by_argos_code(code)


def _colorize_contour(line: str) -> str:
    """Apply rich colour markup to the markers in a contour-chart line."""
    palette = {
        "#": "[green]#[/green]",
        "o": "[cyan]o[/cyan]",
        "*": "[magenta]*[/magenta]",
    }
    return "".join(palette.get(char, char) for char in line)


class TUI:
    """Interactive terminal interface bound to a console and settings."""

    def __init__(self, console: Console, settings: Settings) -> None:
        self.console = console
        self.settings = settings
        self._synthesizer: VoiceSynthesizer | None = None
        self._transcriber: SpeechTranscriber | None = None

    def run(self) -> None:
        """Show the menu and dispatch choices until the user exits."""
        while True:
            self._render_menu()
            choice = Prompt.ask(
                "Select an option",
                choices=["1", "2", "3", "4", "5", "6"],
                console=self.console,
            )
            if not self._dispatch(choice):
                break

    def _dispatch(self, choice: str) -> bool:
        """Run the screen for ``choice``; return ``False`` to exit the menu."""
        if choice == "6":
            self.console.print("Goodbye.")
            return False

        screens = {
            "1": self.screen_record_voice,
            "2": self.screen_translate_text,
            "3": self.screen_practice_pronunciation,
            "4": self.screen_live_translation,
            "5": self.screen_list_translations,
        }
        try:
            screens[choice]()
        except EchoTranslateError as exc:
            self.console.print(f"[red]{exc}[/red]")
            self._pause()
        return True

    def screen_record_voice(self) -> None:
        """Record a microphone sample and save it as a voice profile."""
        self._header("Record voice profile")
        name = Prompt.ask("Name for this voice profile", console=self.console).strip()
        if not name:
            self.console.print("[yellow]No name given; cancelled.[/yellow]")
            self._pause()
            return

        self.console.print(
            "\nSpeak naturally. A longer, varied sample clones better. "
            "See voice_script.txt for guidance."
        )
        self.console.input("Press Enter to start recording...")
        self.console.print("[bold red]Recording. Press Enter to stop.[/bold red]")

        samples = record_until_enter(
            RECORD_SAMPLE_RATE,
            wait_for_stop=lambda: self.console.input(),
        )
        duration = len(samples) / RECORD_SAMPLE_RATE
        if duration < _MIN_PROFILE_SECONDS:
            self.console.print(
                f"[yellow]Only {duration:.1f}s captured; longer samples sound "
                "better.[/yellow]"
            )

        destination = voice_path(self.settings, name)
        write_wav(destination, samples, RECORD_SAMPLE_RATE)
        self.console.print(
            f"[green]Saved voice profile '{name}' ({duration:.1f}s).[/green]"
        )
        self._pause()

    def screen_translate_text(self) -> None:
        """Translate typed text and speak it in a chosen voice."""
        self._header("Translate text")
        voice = self._select_voice()
        if voice is None:
            return
        language = self._select_language()

        text = Prompt.ask("Text to practise (in English)", console=self.console).strip()
        if not text:
            self.console.print("[yellow]No text entered.[/yellow]")
            self._pause()
            return

        spoken = self._to_target_text(text, language)

        synthesizer = self._synth()
        synthesizer.load(progress=self._status)
        output_path = build_output_path(self.settings, voice, language.argos_code)
        self.console.print("[cyan]Synthesising in your voice...[/cyan]")
        synthesizer.synthesize_to_file(
            spoken, voice_path(self.settings, voice), language, output_path
        )
        self.console.print(f"[green]Saved {output_path}[/green]")

        if self._confirm("Play it now?"):
            samples, sample_rate = read_wav(output_path)
            play(samples, sample_rate)
        if self._confirm("Practice your pronunciation now?"):
            self._run_pitch_comparison(output_path, language)
        self._pause()

    def screen_practice_pronunciation(self) -> None:
        """Record an attempt at a saved clip and compare your pitch to it."""
        self._header("Practice pronunciation")
        clips = self._saved_clips()
        if not clips:
            self.console.print(
                "[yellow]No saved clips yet. Use option 2 to generate one "
                "first.[/yellow]"
            )
            self._pause()
            return

        shown = clips[:20]
        for index, path in enumerate(shown, 1):
            self.console.print(f"  {index}. {path.name}")
        choice = IntPrompt.ask(
            "Which clip do you want to practise",
            choices=[str(i) for i in range(1, len(shown) + 1)],
            console=self.console,
        )
        target = shown[choice - 1]
        self._run_pitch_comparison(target, _language_from_clip(target))
        self._pause()

    def screen_live_translation(self) -> None:
        """Listen, transcribe, translate, and speak back in a chosen voice."""
        self._header("Live translation")
        voice = self._select_voice()
        if voice is None:
            return
        language = self._select_language()

        transcriber = self._transcribe()
        transcriber.load(progress=self._status)
        synthesizer = self._synth()
        synthesizer.load(progress=self._status)

        self.console.print(
            f"\n[green]Ready.[/green] Target: {language.display}. "
            "Speak, then pause. Press Ctrl+C to stop.\n"
        )
        chunk_frames = int(LIVE_SAMPLE_RATE * 0.1)
        try:
            for segment in detect_speech_segments(
                microphone_chunks(LIVE_SAMPLE_RATE, chunk_frames=chunk_frames),
                sample_rate=LIVE_SAMPLE_RATE,
            ):
                self._handle_live_segment(segment, voice, language)
        except KeyboardInterrupt:
            self.console.print("\n[yellow]Stopped live mode.[/yellow]")
        self._pause()

    def screen_list_translations(self) -> None:
        """List rendered audio files, most recent first, and optionally play one."""
        self._header("Saved audio")
        files = self._saved_clips()
        if not files:
            self.console.print("[yellow]No saved audio yet.[/yellow]")
            self._pause()
            return

        table = Table(show_header=True, header_style="bold cyan")
        table.add_column("#", width=3)
        table.add_column("File", width=44)
        table.add_column("Size", justify="right", width=10)
        table.add_column("Recorded", width=18)
        shown = files[:20]
        for index, path in enumerate(shown, 1):
            stat = path.stat()
            size = f"{stat.st_size / 1024:.0f} KB"
            when = datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M")
            table.add_row(str(index), path.name, size, when)
        self.console.print(table)

        choice = Prompt.ask(
            "Play which? (number, or Enter to skip)",
            default="",
            show_default=False,
            console=self.console,
        ).strip()
        if choice.isdigit():
            index = int(choice) - 1
            if 0 <= index < len(shown):
                samples, sample_rate = read_wav(shown[index])
                play(samples, sample_rate)
        self._pause()

    def _saved_clips(self) -> list[Path]:
        """Return saved output clips, most recent first."""
        return sorted(
            self.settings.output_dir.rglob("*.wav"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )

    def _run_pitch_comparison(
        self, target_wav: Path, language: Language | None
    ) -> None:
        """Record the user's attempt and chart its pitch against ``target_wav``."""
        self.console.print("\nNow say it yourself.")
        self.console.input("Press Enter to start recording...")
        self.console.print("[bold red]Recording. Press Enter to stop.[/bold red]")
        attempt = record_until_enter(
            RECORD_SAMPLE_RATE, wait_for_stop=lambda: self.console.input()
        )

        target_samples, target_rate = read_wav(target_wav)
        target_contour = extract_f0(target_samples, target_rate)
        attempt_contour = extract_f0(attempt, RECORD_SAMPLE_RATE)
        result = compare_contours(target_contour, attempt_contour)
        if not result.enough_data:
            self.console.print(f"[yellow]{result.message}[/yellow]")
            return

        self.console.print("\n[bold]Pitch contour[/bold] (top = higher pitch)")
        self.console.print(
            "[green]#[/green] target   [cyan]o[/cyan] you   "
            "[magenta]*[/magenta] both\n"
        )
        for line in render_contours(target_contour, attempt_contour):
            self.console.print(_colorize_contour(line))
        self.console.print(f"\nMatch: [bold]{result.match:.0f}%[/bold]")
        if language is not None and language.tonal:
            self.console.print(
                "[dim]This language is tonal: match the shape of the line, "
                "not its height.[/dim]"
            )

    def _handle_live_segment(
        self, segment: np.ndarray, voice: str, language: Language
    ) -> None:
        """Transcribe one captured segment, translate it, and speak it back."""
        result = self._transcribe().transcribe(segment)
        if not result.text:
            return
        self.console.print(f"[dim]({result.language})[/dim] {result.text}")

        spoken = self._pivot_to_target(result.text, result.language, language)
        self.console.print(f"[green]{spoken}[/green]")

        temp_path = Path(tempfile.gettempdir()) / "echotranslate_live.wav"
        self._synth().synthesize_to_file(
            spoken, voice_path(self.settings, voice), language, temp_path
        )
        samples, sample_rate = read_wav(temp_path)
        play(samples, sample_rate, gain=0.8)

    def _to_target_text(self, text: str, language: Language) -> str:
        """Translate English ``text`` into ``language`` (no-op for English)."""
        if language.is_english:
            return text
        translation.ensure_packages([language.argos_code], progress=self._status)
        self.console.print(f"[cyan]Translating to {language.display}...[/cyan]")
        translated = translation.translate(text, "en", language.argos_code)
        self.console.print(f"[green]{translated}[/green]")
        return translated

    def _pivot_to_target(self, text: str, source_code: str, language: Language) -> str:
        """Translate transcribed ``text`` from ``source_code`` into ``language``.

        Pivots through English when the source and target differ from it, matching
        how Argos routes between arbitrary pairs.
        """
        if source_code == language.argos_code:
            return text
        if source_code != "en":
            translation.ensure_pair(source_code, "en", progress=self._status)
            english = translation.translate(text, source_code, "en")
        else:
            english = text
        if language.is_english:
            return english
        translation.ensure_packages([language.argos_code], progress=self._status)
        return translation.translate(english, "en", language.argos_code)

    def _select_voice(self) -> str | None:
        """Prompt for a voice profile, or guide the user to record one."""
        voices = list_voices(self.settings)
        if not voices:
            self.console.print(
                "[yellow]No voice profiles yet. Record one first "
                "(option 1).[/yellow]"
            )
            self._pause()
            return None
        for index, name in enumerate(voices, 1):
            self.console.print(f"  {index}. {name}")
        selection = IntPrompt.ask(
            "Select a voice",
            choices=[str(i) for i in range(1, len(voices) + 1)],
            console=self.console,
        )
        return voices[selection - 1]

    def _select_language(self) -> Language:
        """Prompt for a target language and return it."""
        table = Table(show_header=True, header_style="bold cyan")
        table.add_column("#", width=3)
        table.add_column("Language", width=22)
        table.add_column("Region", width=22)
        for index, language in enumerate(LANGUAGES, 1):
            table.add_row(str(index), language.display, language.region)
        self.console.print(table)
        choice = Prompt.ask(
            "Select a language", choices=menu_choices(), console=self.console
        )
        return by_menu_number(int(choice))

    def _synth(self) -> VoiceSynthesizer:
        """Return the shared, lazily-created synthesizer."""
        if self._synthesizer is None:
            self._synthesizer = VoiceSynthesizer(self.settings)
        return self._synthesizer

    def _transcribe(self) -> SpeechTranscriber:
        """Return the shared, lazily-created transcriber."""
        if self._transcriber is None:
            self._transcriber = SpeechTranscriber(self.settings)
        return self._transcriber

    def _render_menu(self) -> None:
        """Print the banner, known voices, and the main menu."""
        self.console.print(Panel(_BANNER, style="bold cyan"))
        voices = list_voices(self.settings)
        if voices:
            self.console.print(
                f"[green]{len(voices)} voice profile(s):[/green] "
                f"{', '.join(voices)}\n"
            )
        else:
            self.console.print("[yellow]No voice profiles yet.[/yellow]\n")
        self.console.print("[bold]Main menu[/bold]")
        self.console.print("  1. Record a voice profile")
        self.console.print("  2. Translate text and speak it in your voice")
        self.console.print("  3. Practice pronunciation (compare your pitch)")
        self.console.print("  4. Live translation")
        self.console.print("  5. Saved audio")
        self.console.print("  6. Exit\n")

    def _header(self, title: str) -> None:
        """Print a screen header panel."""
        self.console.print(Panel(title, style="bold green"))

    def _status(self, message: str) -> None:
        """Print a dim progress line (used as a progress callback)."""
        self.console.print(f"[dim]{message}[/dim]")

    def _confirm(self, question: str) -> bool:
        """Ask a yes/no question, defaulting to yes."""
        answer = Prompt.ask(
            question, choices=["y", "n"], default="y", console=self.console
        )
        return answer == "y"

    def _pause(self) -> None:
        """Wait for the user to acknowledge before returning to the menu."""
        self.console.input("\nPress Enter to continue...")
