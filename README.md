# EchoTranslate

Learn pronunciation in any language using your own voice.

## Overview

EchoTranslate is a language learning tool that helps you master pronunciation by hearing translated text spoken in your own voice. This revolutionary approach eliminates the confusion between instructor accents and actual pronunciation requirements, allowing you to focus on the essential sounds of your target language.

## Key Features

- **Voice Profile Creation**: Record your voice once, use it across all languages
- **Multi-Language Support**: Practice pronunciation in 10+ languages
- **Live Translation**: Real-time practice with instant feedback
- **Simple Terminal GUI**: Easy-to-use menu system
- **Privacy-Focused**: Everything runs locally on your machine

## Installation

### Prerequisites

1. **Activate Conda Environment**:
   ```bash
   conda activate echotranslate
   ```

2. **Install the program**:
   ```bash
   chmod +x install_simple.sh
   sudo ./install_simple.sh
   ```

3. **Download the voice synthesis model** (1.87 GB - one time only):
   ```bash
   python -c "import os; os.environ['COQUI_TOS_AGREED']='1'; from TTS.api import TTS; TTS('tts_models/multilingual/multi-dataset/xtts_v2')"
   ```
   This will take 5-10 minutes depending on your internet speed.

4. **Download Whisper model for offline speech recognition** (460 MB):
   ```bash
   python download_whisper.py
   ```

## Usage

Simply type in your terminal:
```bash
echotranslate
```

You'll see this menu:
```
╔═══════════════════════════════════════════════════════════╗
║                     EchoTranslate                         ║
║            Your Voice Speaking Any Language               ║
╚═══════════════════════════════════════════════════════════╝

✓ 2 voice(s) available: User1, demo_voice

Main Menu:

  1. Record Your Voice
  2. Translate Text (with your voice)
  3. Live Translation Mode
  4. View Saved Translations
  5. Exit

Select option:
```

### Menu Options Explained

#### 1. Create Your Voice Profile
- Record a 60-second sample (customizable duration)
- Saves to `voices/` directory with your chosen name
- **First time? Read this sample text for best results:**

> "Hello, my name is [your name]. Today is a beautiful day, and I'm excited to explore new languages. The weather is nice, with temperatures around seventy-five degrees. I enjoy reading books, watching movies, and traveling to new places. Learning languages opens new doors to understanding different cultures. One, two, three, four, five, six, seven, eight, nine, ten. Proper pronunciation is essential for effective communication. Thank you for listening to my voice sample."

**Recording Tips:**
- Find a quiet room
- Speak naturally with clear pronunciation
- Include statements, questions, and exclamations
- Vary your pace and tone

#### 2. Practice with Text Translation
1. Select your voice profile
2. Choose target language:
   - English - Test your voice profile
   - Spanish (Español) - Latin America/Spain
   - Chinese (中文) - Taiwan
   - French (Français) - France
   - Arabic (العربية) - Middle East
   - German (Deutsch) - Germany
   - Italian (Italiano) - Italy
   - Portuguese (Português) - Brazil/Portugal
   - Russian (Русский) - Russia
   - Japanese (日本語) - Japan
   - Korean (한국어) - South Korea
3. Type your practice text
4. Hear the translation in your voice
5. Audio saved to `output/YYYY-MM-DD/HHMMSS_ProfileName_Language.wav`

#### 3. Live Translation Mode
- Speak naturally in any language
- Hear immediate translation in your voice
- Real-time processing
- Perfect for pronunciation practice

#### 4. View Saved Translations
- Browse all your translated audio files
- Organized by date
- Play any file directly from the menu

## How It Works

```
1. Voice Profile Creation:
   Your Voice → Microphone → 22050Hz WAV → voices/yourname.wav
                                             ↓
                                   Voice characteristics saved

2. Text Translation:
   Your Text → Argos Translate → Target Language Text
      ↓                              ↓
   English                     e.g., "Hello" → "Hola"

3. Voice Synthesis:
   Your Voice Profile + Translated Text → XTTS v2 Model → Audio Output
                                                             ↓
                                                  output/date/time_profile_lang.wav

4. Live Practice Mode:
   Your Speech → Whisper (Offline) → Detected Language Text
                                             ↓
                                      Argos Translate
                                             ↓
                                     Target Language Text
                                             ↓
   Your Voice Profile → XTTS v2 Synthesis → Immediate Playback
```

## Technical Architecture

### Core Components
- **Voice Synthesis Engine**: Coqui TTS XTTS v2 
  - Multilingual neural model (1.87GB)
  - Maintains voice characteristics across languages
  - Supports 15+ languages

- **Translation Engine**: Argos Translate
  - Offline translation
  - 50-100MB per language pair
  - Neural machine translation

- **Speech Recognition**: OpenAI Whisper (small model)
  - Offline speech-to-text
  - Automatic language detection
  - 460MB model

- **Audio Processing**:
  - Recording: sounddevice (22050Hz, mono, float32)
  - File I/O: soundfile (WAV format)
  - Playback: sounddevice (80% volume)

- **Interface**: Rich terminal UI
  - Interactive menus
  - Progress indicators
  - Clear visual feedback

### Configuration
- Environment variables:
  - `COQUI_TOS_AGREED=1`: Auto-accepts XTTS license
  - `KMP_DUPLICATE_LIB_OK=TRUE`: Prevents OpenMP conflicts

## File Structure

```
EchoTranslate/
├── echotranslate              # Main program (all functionality)
├── install_simple.sh          # Installer script
├── README.md                  # This file
├── requirements.txt           # Python dependencies
│
├── voices/                   # Your voice profiles
│   ├── [YourName].wav       # Example: 60-second recording
│   └── demo_voice.wav       # Example: demo profile
│
├── output/                   # Translated audio files
│   └── 2024-06-28/          # Organized by date
│       ├── 150325_User1_es.wav
│       ├── 151247_User1_zh-tw.wav
│       └── 152108_User1_fr.wav
│
└── voice_script.txt          # Sample text for recording
```

## Tips for Best Results

### Voice Profile Recording
1. **Optimal Duration**: 60-120 seconds
2. **Content Variety**:
   - Clear statements: "The weather is beautiful today."
   - Questions: "How are you feeling?"
   - Exclamations: "That's wonderful!"
   - Numbers: "One through ten"
   - Various tones: Happy, neutral, questioning

3. **Technical Quality**:
   - Quiet environment
   - Consistent microphone distance
   - Natural speaking volume
   - Minimal background noise

### Better Language Practice
- Start with simple, clear sentences
- Focus on common phrases first
- Break complex sentences into parts
- Listen multiple times to perfect pronunciation

### Live Mode Tips
- **Use headphones** to prevent feedback loops
- Speak clearly with pauses between phrases
- Keep phrases under 10 seconds
- Allow 2-3 seconds for processing

## Troubleshooting

| Issue | Solution |
|-------|----------|
| "XTTS model not found!" | Run the model download command (Prerequisites step 3) |
| "No voices recorded yet" | Use option 1 to record your voice first |
| Live mode audio feedback | Use headphones or reduce speaker volume to 50% |
| "Could not understand audio" | Speak more clearly, reduce background noise |
| Audio sounds unnatural | Record a longer, more varied voice sample |
| Long processing time | First run loads models (~30 seconds), subsequent runs are faster |
| "Error loading Whisper" | Run: `python download_whisper.py` while connected to internet |

## Privacy & Security

- **100% Local Processing**: All translation and synthesis happens on your machine
- **No Cloud Storage**: Your recordings never leave your computer
- **Offline Translation**: Works without internet connection
- **Offline Speech Recognition**: Whisper runs locally
- **Your Data**: All files remain in local `voices/` and `output/` folders

## System Requirements

- **OS**: macOS, Linux, Windows (with WSL)
- **Python**: 3.11 (via conda environment)
- **Disk Space**: ~3GB (1.87GB for XTTS, 1GB for translations)
- **RAM**: 4GB minimum, 8GB recommended
- **Audio**: Microphone for recording/live mode
- **Internet**: Only for initial model downloads

## Examples

### Practice Business Phrases
```bash
# After setup:
echotranslate

# Select option 2, choose Spanish, type:
# "Dear colleagues, I hope this email finds you well."
# Hear in your voice: "Estimados colegas, espero que este correo les encuentre bien."
```

### Practice Session Workflow
1. Prepare phrases to practice
2. Use option 2 for each phrase
3. All audio saved for review

### Effective Language Learning
1. Create your voice profile (option 1)
2. Practice common phrases (option 2)
3. Listen to proper pronunciation in your voice
4. Use live mode for real-time practice

---

Ready to start learning? Create your voice profile and begin practicing!