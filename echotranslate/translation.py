"""Offline-first translation via Argos Translate.

Argos translates through English: with ``en->es`` and ``en->ja`` installed it can
also go ``es->ja`` by pivoting, so the app only needs the ``en->X`` packages for
its target languages plus, in live mode, the ``X->en`` package for whatever
language the speaker used.

Startup stays offline: checking which packages are installed never touches the
network, and the package index is refreshed only when a required package is
missing and the caller has allowed a download.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable

import argostranslate.package
import argostranslate.translate

from echotranslate.errors import TranslationPackageError

# A directed language pair, e.g. ``("en", "es")``.
LanguagePair = tuple[str, str]

ProgressCallback = Callable[[str], None]


def installed_pairs() -> set[LanguagePair]:
    """Return every installed ``(from_code, to_code)`` pair. Offline; no network."""
    return {
        (pkg.from_code, pkg.to_code)
        for pkg in argostranslate.package.get_installed_packages()
    }


def installed_target_codes() -> set[str]:
    """Return target codes ``X`` for which an ``en->X`` package is installed.

    Offline; no network.
    """
    return {to_code for (from_code, to_code) in installed_pairs() if from_code == "en"}


def missing_packages(required: Iterable[str]) -> list[LanguagePair]:
    """Return the ``en->X`` pairs that are required but not installed.

    Args:
        required: Target language codes the app wants to be able to reach.

    Returns:
        The missing ``("en", code)`` pairs, preserving the order of ``required``.
        Pure set difference; no network.
    """
    installed = installed_target_codes()
    return [("en", code) for code in required if code not in installed]


def _install_pairs(
    pairs: list[LanguagePair],
    progress: ProgressCallback | None,
) -> list[LanguagePair]:
    """Download and install the given pairs, refreshing the index once.

    Assumes the pairs are genuinely missing and a download is permitted.
    """
    argostranslate.package.update_package_index()
    available = argostranslate.package.get_available_packages()
    by_pair = {(pkg.from_code, pkg.to_code): pkg for pkg in available}

    installed: list[LanguagePair] = []
    for pair in pairs:
        candidate = by_pair.get(pair)
        if candidate is None:
            raise TranslationPackageError(
                f"No Argos package is available for {pair[0]}->{pair[1]}."
            )
        if progress is not None:
            progress(f"Installing {pair[0]}->{pair[1]} translation model...")
        argostranslate.package.install_from_path(candidate.download())
        installed.append(pair)
    return installed


def ensure_packages(
    required: Iterable[str],
    *,
    allow_download: bool = True,
    progress: ProgressCallback | None = None,
) -> list[LanguagePair]:
    """Ensure ``en->X`` packages exist for every required target code.

    Args:
        required: Target language codes that must be reachable.
        allow_download: If ``True``, download missing packages (refreshing the
            index once). If ``False``, raise instead of reaching the network.
        progress: Optional callback invoked with a human-readable message before
            each download.

    Returns:
        The pairs that were newly installed (empty if everything was present).

    Raises:
        TranslationPackageError: If a package is missing and ``allow_download``
            is ``False``, or no matching package exists in the index.
    """
    missing = missing_packages(required)
    if not missing:
        return []
    if not allow_download:
        pretty = ", ".join(f"{a}->{b}" for a, b in missing)
        raise TranslationPackageError(
            f"Missing translation packages ({pretty}) and downloads are disabled."
        )
    return _install_pairs(missing, progress)


def ensure_pair(
    from_code: str,
    to_code: str,
    *,
    allow_download: bool = True,
    progress: ProgressCallback | None = None,
) -> None:
    """Ensure a single ``from_code -> to_code`` package is installed.

    Used by live mode for the speaker's detected language, which cannot be known
    ahead of time. Same offline-first contract as :func:`ensure_packages`.
    """
    pair = (from_code, to_code)
    if pair in installed_pairs():
        return
    if not allow_download:
        raise TranslationPackageError(
            f"Missing translation package ({from_code}->{to_code}) "
            "and downloads are disabled."
        )
    _install_pairs([pair], progress)


def translate(text: str, from_code: str, to_code: str) -> str:
    """Translate ``text`` from one language to another.

    Returns the text unchanged when the languages match (the English-to-English
    reference case), avoiding a needless round-trip through Argos. Otherwise
    delegates to Argos, which pivots through English as needed.
    """
    if from_code == to_code:
        return text
    result: str = argostranslate.translate.translate(text, from_code, to_code)
    return result
