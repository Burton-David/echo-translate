#!/usr/bin/env python3
"""Download and verify Whisper model for offline speech recognition.

This script downloads the specified Whisper model and verifies it works correctly.
The model is cached locally for offline use with EchoTranslate.
"""

import sys
import argparse
from pathlib import Path

import whisper


# Model sizes in MB (approximate)
MODEL_SIZES = {
    "tiny": 39,
    "base": 74,
    "small": 244,
    "medium": 769,
    "large": 1550
}


def download_model(model_name="small"):
    """Download and verify Whisper model.
    
    Args:
        model_name: Model size to download (tiny, base, small, medium, large)
        
    Returns:
        bool: True if successful, False otherwise
    """
    cache_dir = Path.home() / ".cache" / "whisper"
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    model_size = MODEL_SIZES.get(model_name, 244)
    print(f"Downloading Whisper {model_name} model...")
    print(f"This will download approximately {model_size}MB")
    print(f"Model will be cached in: {cache_dir}")
    print()
    
    try:
        # Download model
        print("Downloading model (this may take several minutes)...")
        model = whisper.load_model(model_name, download_root=str(cache_dir))
        
        # Verify model works
        print("\nVerifying model...")
        import numpy as np
        test_audio = np.zeros(16000, dtype=np.float32)  # 1 second of silence
        result = model.transcribe(test_audio, fp16=False)
        
        print("\nModel downloaded and verified successfully!")
        print("You can now use EchoTranslate offline")
        return True
        
    except ConnectionError as e:
        print(f"\nConnection error: {e}")
        print("Please check your internet connection and try again")
        return False
        
    except MemoryError:
        print(f"\nMemory error: Not enough RAM for {model_name} model")
        print("Try a smaller model (tiny or base)")
        return False
        
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        print("Please report this issue if it persists")
        return False


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Download Whisper model for offline use with EchoTranslate"
    )
    parser.add_argument(
        "--model",
        default="small",
        choices=list(MODEL_SIZES.keys()),
        help="Model size to download (default: small)"
    )
    
    args = parser.parse_args()
    
    success = download_model(args.model)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()