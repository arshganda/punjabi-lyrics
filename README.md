# Punjabi Lyrics Extractor

Extract and translate Punjabi song lyrics from MP3 files using Gemini AI.

## Features

- Audio preprocessing for better transcription quality
- Gurmukhi text extraction using Gemini 2.0 Flash
- Automatic correction of Gurmukhi transcription errors
- ISO 15919 standard romanization
- Context-aware English translation
- Structured JSON output for lyrics websites
- Line-by-line mapping across all three formats

## Setup

### Prerequisites

1. Install `uv` if you haven't already:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. Install optional audio tools for better preprocessing:
```bash
# macOS
brew install sox ffmpeg

# Linux
sudo apt-get install sox ffmpeg
```

### Installation

1. Install dependencies:
```bash
uv sync
```

2. Set up your Gemini API key:
```bash
export GEMINI_API_KEY="your-api-key-here"
```

Or create a `.env` file:
```bash
cp .env.example .env
# Edit .env with your API key
```

## Usage

### Basic usage:
```bash
uv run python punjabi_lyrics_extractor.py <mp3_file>
```

### With custom output:
```bash
uv run python punjabi_lyrics_extractor.py <mp3_file> <output.json>
```

### Example:
```bash
uv run python punjabi_lyrics_extractor.py supreme-sidhu-bad-habits.mp3
```

## Output Format

The tool generates structured JSON output:

```json
{
  "metadata": {
    "title": "Song Title",
    "artist": "Artist Name",
    "processed_date": "2025-06-03T...",
    "confidence_score": 0.85
  },
  "full_text": {
    "gurmukhi": "*p>,@ H8",
    "romanized": "panjabi text",
    "english": "English translation"
  },
  "lines": [
    {
      "line_number": 1,
      "gurmukhi": "*9?2@ 2>(",
      "romanized": "pahili lain",
      "english": "First line"
    }
  ],
  "sections": [...],
  "processing_notes": {
    "unclear_sections": [],
    "autocorrections_made": true
  }
}
```

## Development

### Running tests:
```bash
uv run pytest
```

### Adding new dependencies:
```bash
uv add <package_name>
```

## Troubleshooting

1. **Audio processing errors**: Make sure `ffmpeg` is installed
2. **Low confidence scores**: Try preprocessing the audio with vocal isolation tools
3. **API errors**: Check your Gemini API key and quotas

## Future Improvements

- Vocal isolation using Demucs/Spleeter
- Support for timestamp-synced lyrics (SRT format)
- Batch processing for multiple songs
- Web UI for easier access
- Custom fine-tuning for better Punjabi recognition