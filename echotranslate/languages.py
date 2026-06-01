"""The supported-language registry.

Each language carries separate codes for the two backends, which disagree on
spelling: Argos Translate uses ISO 639-1 (``zh`` for Chinese), while XTTS expects
its own locale-style tags (``zh-cn``).

English is the pivot language: Argos translates everything through English, and
English text is spoken directly without a translation step.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Language:
    """A target language and the codes each backend needs for it.

    Attributes:
        display: Human-readable name shown in the menu, e.g. ``"Spanish (Español)"``.
        region: Short note on the dialect/region the models target.
        argos_code: ISO 639-1 code passed to Argos Translate (English is the pivot).
        xtts_code: Locale tag passed to the XTTS synthesis model.
    """

    display: str
    region: str
    argos_code: str
    xtts_code: str

    @property
    def is_english(self) -> bool:
        """Whether this is the English reference language (no translation needed)."""
        return self.argos_code == "en"


# Ordered tuple; the 1-based position is the number shown in the menu.
LANGUAGES: tuple[Language, ...] = (
    Language("English", "Reference voice check", "en", "en"),
    Language("Spanish (Español)", "Latin America/Spain", "es", "es"),
    Language("Chinese (中文)", "Mainland", "zh", "zh-cn"),
    Language("French (Français)", "France", "fr", "fr"),
    Language("Arabic (العربية)", "Middle East", "ar", "ar"),
    Language("German (Deutsch)", "Germany", "de", "de"),
    Language("Italian (Italiano)", "Italy", "it", "it"),
    Language("Portuguese (Português)", "Brazil/Portugal", "pt", "pt"),
    Language("Russian (Русский)", "Russia", "ru", "ru"),
    Language("Japanese (日本語)", "Japan", "ja", "ja"),
    Language("Korean (한국어)", "South Korea", "ko", "ko"),
)


def by_menu_number(number: int) -> Language:
    """Return the language at a 1-based menu position.

    Args:
        number: The menu number the user selected (1 through ``len(LANGUAGES)``).

    Returns:
        The matching :class:`Language`.

    Raises:
        ValueError: If ``number`` is outside the valid range.
    """
    if not 1 <= number <= len(LANGUAGES):
        raise ValueError(f"No language for menu number {number}")
    return LANGUAGES[number - 1]


def menu_choices() -> list[str]:
    """Return the menu numbers as strings, e.g. ``["1", "2", ..., "11"]``.

    Suitable for passing to a rich ``Prompt``'s ``choices`` argument.
    """
    return [str(i) for i in range(1, len(LANGUAGES) + 1)]


def translation_targets() -> list[str]:
    """Return the Argos codes that need an installed package (English excluded)."""
    return [lang.argos_code for lang in LANGUAGES if not lang.is_english]
