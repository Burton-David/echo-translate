"""Tests for the offline-first Argos translation wiring.

Argos itself is monkeypatched so these tests never touch the network or load a
real model. The behaviour under test is our wiring: offline checks, when the
index is refreshed, and the English passthrough.
"""

from __future__ import annotations

import argostranslate.package as pkg_mod
import argostranslate.translate as translate_mod
import pytest

from echotranslate import translation
from echotranslate.errors import TranslationPackageError


class _FakePackage:
    def __init__(self, from_code: str, to_code: str) -> None:
        self.from_code = from_code
        self.to_code = to_code

    def download(self) -> str:
        return f"/tmp/{self.from_code}_{self.to_code}.argosmodel"


def _install_fake_index(
    monkeypatch: pytest.MonkeyPatch,
    *,
    installed: list[_FakePackage],
    available: list[_FakePackage] | None = None,
) -> dict[str, list]:
    """Wire up fake Argos package functions and return call recorders."""
    calls: dict[str, list] = {"update": [], "installed_from": []}

    monkeypatch.setattr(pkg_mod, "get_installed_packages", lambda: installed)
    monkeypatch.setattr(pkg_mod, "get_available_packages", lambda: available or [])
    monkeypatch.setattr(
        pkg_mod, "update_package_index", lambda: calls["update"].append(True)
    )
    monkeypatch.setattr(
        pkg_mod, "install_from_path", lambda path: calls["installed_from"].append(path)
    )
    return calls


def test_translate_forwards_arguments(monkeypatch: pytest.MonkeyPatch) -> None:
    recorded: list[tuple[str, str, str]] = []

    def fake_translate(text: str, from_code: str, to_code: str) -> str:
        recorded.append((text, from_code, to_code))
        return "hola"

    monkeypatch.setattr(translate_mod, "translate", fake_translate)
    assert translation.translate("hello", "en", "es") == "hola"
    assert recorded == [("hello", "en", "es")]


def test_translate_passthrough_for_same_language(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail(*_args: object) -> str:
        raise AssertionError("Argos should not be called for en->en")

    monkeypatch.setattr(translate_mod, "translate", fail)
    assert translation.translate("hello", "en", "en") == "hello"


def test_installed_target_codes_filters_to_english_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_index(
        monkeypatch,
        installed=[
            _FakePackage("en", "es"),
            _FakePackage("es", "en"),
            _FakePackage("en", "fr"),
        ],
    )
    assert translation.installed_target_codes() == {"es", "fr"}


def test_missing_packages_is_pure_diff(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_index(monkeypatch, installed=[_FakePackage("en", "es")])
    assert translation.missing_packages(["es", "fr", "de"]) == [
        ("en", "fr"),
        ("en", "de"),
    ]


def test_ensure_packages_offline_raises_without_network(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _install_fake_index(monkeypatch, installed=[])
    with pytest.raises(TranslationPackageError):
        translation.ensure_packages(["es"], allow_download=False)
    assert calls["update"] == []  # never reached the network


def test_ensure_packages_all_present_skips_network(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _install_fake_index(
        monkeypatch, installed=[_FakePackage("en", "es"), _FakePackage("en", "fr")]
    )
    assert translation.ensure_packages(["es", "fr"]) == []
    assert calls["update"] == []  # offline-first: nothing missing, no refresh


def test_ensure_packages_downloads_only_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = _install_fake_index(
        monkeypatch,
        installed=[_FakePackage("en", "es")],
        available=[_FakePackage("en", "de")],
    )
    installed = translation.ensure_packages(["es", "de"])
    assert installed == [("en", "de")]
    assert len(calls["update"]) == 1  # index refreshed exactly once
    assert calls["installed_from"] == ["/tmp/en_de.argosmodel"]


def test_ensure_packages_unknown_pair_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_index(monkeypatch, installed=[], available=[])
    with pytest.raises(TranslationPackageError):
        translation.ensure_packages(["xx"])
