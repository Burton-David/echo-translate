# EchoTranslate

EchoTranslate is a local, offline command-line tool. You record a short sample of
your voice once, type (or speak) some English, and it plays the translation back
to you in another language, in your own voice.

Everything runs on your machine. After the one-time model downloads, no audio or
text leaves your computer.

## Why I built this

I learned Arabic at a school in Egypt and Mandarin at a language school in Taiwan.
In both, my teacher was a woman, and my voice is deep. That mismatch mattered more
than I expected.

With Mandarin, I genuinely couldn't get the tones. Demonstrated in a voice much
higher than mine, the pitch contours never mapped onto my own range, and I
couldn't tell what I was supposed to reproduce. Hearing the same words in my own
voice was the thing that finally made the tones click.

With Arabic, it was about confidence. I'd picked up a softer, higher register than
my own, and in some settings sounding noticeably unlike the men around me made me
hesitant to speak at all. Hearing phrases in my own voice helped me settle into a
register that actually felt like mine, and that was the difference between staying
quiet and joining the conversation.

## What it does today

- **Records a voice profile** from your microphone and saves it as a WAV file.
- **Translates English into 10 languages** offline with [Argos Translate](https://github.com/argosopentech/argos-translate)
  (Spanish, French, German, Italian, Portuguese, Russian, Chinese, Japanese,
  Korean, Arabic).
- **Speaks the translation in your cloned voice** using Coqui XTTS v2 (via the
  community-maintained [`coqui-tts`](https://github.com/idiap/coqui-ai-TTS) fork)
  and saves each clip under `output/<date>/`.
- **Compares your pitch against the target.** Record yourself saying the phrase
  and it charts your pitch contour over the target's, with a match score. The
  contours are normalised so a deep voice and a high one are judged on shape, not
  height. This is most useful for tonal languages like Mandarin, where the tones
  *are* pitch shapes.
- **Live mode:** speak into the mic, and it transcribes you with
  [faster-whisper](https://github.com/SYSTRAN/faster-whisper), translates, and
  replies in your voice.
- **Reviews saved clips:** browse and replay everything you've generated.
- A small [rich](https://github.com/Textualize/rich) terminal menu ties it
  together.

## Example

```text
$ echotranslate

Main menu
  1. Record a voice profile
  2. Translate text and speak it in your voice
  3. Live translation
  4. Saved audio
  5. Exit

Select an option [1/2/3/4/5]: 2
```

Pick your voice profile and Spanish, then type some English:

```text
Text to practise (in English): I would like a coffee and a glass of water, please.
Translating to Spanish (Español)...
Me gustaría un café y un vaso de agua, por favor.
Synthesising in your voice...
Saved output/2026-06-01/120000_myvoice_es.wav
Play it now? [y/n] (y):
```

The clip plays back in your voice, speaking Spanish.

## Installation

EchoTranslate needs **Python 3.10–3.14**. The voice-cloning and live modes
download large models the first time you use them (~2 GB for XTTS, ~240 MB for
the Whisper model); translation-only use needs neither.

There are three install tiers so you don't have to pull the full machine-learning
stack just to try translation:

| Install | Command | Adds |
| --- | --- | --- |
| Core | `pip install -e .` | Text translation, the menu, browsing saved clips |
| + audio | `pip install -e ".[audio]"` | Microphone recording and playback |
| Full | `pip install -e ".[voice]"` | Voice cloning (XTTS) and live speech-to-text (faster-whisper) |

```bash
# 1. Recording/playback need the PortAudio system library (skip for core-only):
#    macOS:         brew install portaudio
#    Debian/Ubuntu: sudo apt install libportaudio2

# 2. Create and activate a virtual environment:
python3 -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate

# 3. Install (full tier shown):
pip install -e ".[voice]"

# 4. (Full tier) Accept the XTTS licence and download the model (~2 GB):
export COQUI_TOS_AGREED=1
python -c "from TTS.api import TTS; TTS('tts_models/multilingual/multi-dataset/xtts_v2')"

# 5. (Full tier) Download the Whisper model for live mode (~240 MB):
python download_whisper.py           # defaults to the 'small' model

# 6. Run it:
echotranslate
```

PyTorch is pulled in automatically. If you need a specific build (for example a
particular CUDA version), install it from the
[PyTorch index](https://pytorch.org/get-started/locally/) before step 3.

## Usage

Run `echotranslate` and choose from the menu:

1. **Record a voice profile.** Speak naturally; a longer, varied sample clones
   better. See [`voice_script.txt`](voice_script.txt) for guidance on what to say.
2. **Translate text and speak it.** Choose a voice and target language, type
   English, and hear it back. Clips are saved to
   `output/<date>/<time>_<voice>_<lang>.wav`. You can practise your pronunciation
   straight afterwards.
3. **Practice pronunciation.** Pick a saved clip, record yourself saying it, and
   see your pitch contour drawn over the target's with a match score. The score is
   a pitch-shape similarity, not a full pronunciation grade.
4. **Live translation.** Speak; it transcribes (faster-whisper), translates, and
   replies in your voice. It is near-real-time, not instant: synthesis takes a
   moment per phrase. Use headphones to avoid feedback. Press Ctrl+C to stop.
5. **Saved audio.** List and replay your generated clips.

## How it works

```text
Record:    microphone ──▶ voices/<name>.wav            (your reference voice)

Translate: English text ──▶ Argos Translate ──▶ target-language text
                                                      │
              your voice profile ─────────────────────┘
                                                      ▼
                                  XTTS v2 ──▶ output/<date>/<clip>.wav ──▶ playback

Live:      speech ──▶ faster-whisper (detect + transcribe) ──▶ Argos ──▶ XTTS ──▶ playback
```

Argos translates through English, so it can reach a target language even without
a direct model by pivoting. Required language packages download on first use and
are then available offline.

## Limitations and what's verified

This is honest about what has and hasn't been exercised automatically.

**Verified by the test suite (runs on every change, no models or microphone
needed):**

- Translation wiring (English-pivot routing, offline-first package checks).
- Configuration, output-path construction, and voice/clip discovery.
- WAV read/write round-trips.
- Pitch-contour extraction, register-invariant comparison, and chart rendering
  (verified on synthetic signals).
- Terminal-menu state, dispatch, and error rendering.
- That the package imports and the menu runs with only the core dependencies, and
  that the `voice` extra's dependencies resolve on Python 3.12.

**Not run automatically (needs the downloaded models and a microphone):**

- Voice-cloning quality and how much a clip sounds like you (subjective, and
  dependent on the length and quality of your reference recording).
- End-to-end synthesis latency and live-mode round-trip latency.
- Real microphone capture and speaker playback.

These paths depend on roughly 2.5 GB of models and audio hardware. They are
covered by tests marked `heavy`, which are excluded from the default run; assess
quality and latency on your own machine (`pytest -m heavy`).

**Performance and maintenance notes:**

- **XTTS runs on CPU.** The current backend does not support Apple Silicon (MPS)
  acceleration, so expect a few seconds of synthesis per sentence.
- **Coqui TTS is discontinued upstream.** EchoTranslate depends on the
  community-maintained Idiap [`coqui-tts`](https://github.com/idiap/coqui-ai-TTS)
  fork, which keeps the same API and model.
- **Offline after download.** Model and language-package downloads need an
  internet connection; once they're cached, translation and synthesis run
  offline.

## Privacy

All translation, transcription, and synthesis run locally. Your recordings and
generated clips stay in the `voices/` and `output/` directories on your machine
and are never uploaded. The only network access is the one-time download of the
models and language packages.

## System requirements

- **OS:** macOS, Linux, or Windows.
- **Python:** 3.10–3.14.
- **Disk:** ~2.5–3 GB for the full install (XTTS ~2 GB, Whisper ~240 MB, plus
  language packages).
- **Audio:** a microphone for recording and live mode; PortAudio installed.
- **Compute:** CPU-only inference; 8 GB RAM recommended.

## Development

```bash
python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"
.venv/bin/black --check . && .venv/bin/ruff check . && .venv/bin/mypy echotranslate
.venv/bin/python -m pytest          # default suite; add -m heavy for model tests
```

The code is organised as a typed package: `config`, `languages`, `translation`,
`synthesis`, `transcription`, `audio`, `voices`, `tui`, and `cli`. The heavy ML
backends are imported lazily, so the core installs and the suite runs without
them.

## License

MIT. See [LICENSE](LICENSE).
