"""Shared fixtures for the EchoTranslate test suite."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from rich.console import Console

from echotranslate.config import Settings


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    """Settings rooted under a temporary directory (no real home paths touched)."""
    return Settings(
        voices_dir=tmp_path / "voices",
        output_dir=tmp_path / "output",
        whisper_cache_dir=tmp_path / "whisper",
    )


@pytest.fixture
def console() -> MagicMock:
    """A stand-in console that records print/input calls without rendering."""
    return MagicMock(spec=Console)
