#!/usr/bin/env python3
"""Download and verify the Whisper model used for EchoTranslate's live mode.

Fetches a faster-whisper (CTranslate2) model and confirms it loads, caching it
locally so live mode can run offline afterwards.
"""

import argparse
import sys
from pathlib import Path

from faster_whisper import WhisperModel

# Approximate on-disk sizes in MB for the int8 CTranslate2 models.
MODEL_SIZES = {"tiny": 39, "base": 74, "small": 244, "medium": 769, "large-v3": 1550}


def download_model(model_name: str = "small") -> bool:
    """Download and load a faster-whisper model.

    Args:
        model_name: Model size to fetch (tiny, base, small, medium, large-v3).

    Returns:
        True if the model downloaded and loaded, False otherwise.
    """
    cache_dir = Path.home() / ".cache" / "whisper"
    cache_dir.mkdir(parents=True, exist_ok=True)

    size = MODEL_SIZES.get(model_name, 244)
    print(f"Downloading faster-whisper '{model_name}' model (about {size} MB)...")
    print(f"Caching in: {cache_dir}")

    try:
        WhisperModel(
            model_name,
            device="auto",
            compute_type="int8",
            download_root=str(cache_dir),
        )
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"\nCould not download or load the model: {exc}")
        print("Check your internet connection and the model name, then try again.")
        return False

    print("\nModel downloaded and verified. Live mode can now run offline.")
    return True


def main() -> None:
    """Parse arguments and download the requested model."""
    parser = argparse.ArgumentParser(
        description="Download the Whisper model for EchoTranslate live mode"
    )
    parser.add_argument(
        "--model",
        default="small",
        choices=list(MODEL_SIZES),
        help="Model size to download (default: small)",
    )
    args = parser.parse_args()
    sys.exit(0 if download_model(args.model) else 1)


if __name__ == "__main__":
    main()
