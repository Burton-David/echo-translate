"""Tests for the language registry."""

from __future__ import annotations

import pytest

from echotranslate import languages
from echotranslate.languages import LANGUAGES, by_argos_code, by_menu_number


def test_registry_has_expected_size_and_english_first() -> None:
    assert len(LANGUAGES) == 11
    english = LANGUAGES[0]
    assert english.display == "English"
    assert english.argos_code == "en"
    assert english.xtts_code == "en"
    assert english.is_english


def test_chinese_uses_distinct_argos_and_xtts_codes() -> None:
    # Argos uses "zh" for Chinese; XTTS uses "zh-cn". They must stay distinct.
    chinese = by_menu_number(3)
    assert chinese.argos_code == "zh"
    assert chinese.xtts_code == "zh-cn"


def test_by_menu_number_is_one_based() -> None:
    assert by_menu_number(1).display == "English"
    assert by_menu_number(2).argos_code == "es"
    assert by_menu_number(11).argos_code == "ko"


@pytest.mark.parametrize("number", [0, -1, 12, 99])
def test_by_menu_number_rejects_out_of_range(number: int) -> None:
    with pytest.raises(ValueError):
        by_menu_number(number)


def test_menu_choices_match_registry() -> None:
    assert languages.menu_choices() == [str(i) for i in range(1, 12)]


def test_translation_targets_exclude_english() -> None:
    targets = languages.translation_targets()
    assert "en" not in targets
    assert len(targets) == 10
    assert "zh" in targets and "es" in targets


def test_only_chinese_is_marked_tonal() -> None:
    assert by_menu_number(3).tonal is True
    assert by_menu_number(2).tonal is False
    assert LANGUAGES[0].tonal is False


def test_by_argos_code_lookup() -> None:
    assert by_argos_code("zh").display.startswith("Chinese")
    assert by_argos_code("es").argos_code == "es"
    assert by_argos_code("xx") is None


def test_no_duplicate_display_names_and_codes_non_empty() -> None:
    displays = [lang.display for lang in LANGUAGES]
    assert len(displays) == len(set(displays))
    for lang in LANGUAGES:
        assert lang.argos_code
        assert lang.xtts_code
        assert lang.display
