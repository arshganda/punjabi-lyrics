#!/usr/bin/env python3
"""
Example usage of the Punjabi Lyrics Extractor
"""

import os
from pathlib import Path
from punjabi_lyrics_extractor import PunjabiLyricsExtractor

# Set your Gemini API key
os.environ['GEMINI_API_KEY'] = 'your-api-key-here'

# Initialize extractor
extractor = PunjabiLyricsExtractor(os.getenv('GEMINI_API_KEY'))

# Process the example MP3
mp3_file = "supreme-sidhu-bad-habits.mp3"
output_file = "bad_habits_lyrics.json"

# Run extraction
result = extractor.process_song(mp3_file, output_file)

# Access specific parts of the result
print("\n" + "="*50)
print("ACCESSING STRUCTURED DATA:")
print("="*50)

# Get full texts
print(f"Full Gurmukhi text length: {len(result['full_text']['gurmukhi'])} chars")
print(f"Full English translation length: {len(result['full_text']['english'])} chars")

# Get line-by-line data
print(f"\nTotal lines: {len(result['lines'])}")
print("\nLine 5 (if exists):")
if len(result['lines']) >= 5:
    line = result['lines'][4]  # 0-indexed
    print(f"  Gurmukhi: {line['gurmukhi']}")
    print(f"  Romanized: {line['romanized']}")
    print(f"  English: {line['english']}")

# Get metadata
print(f"\nSong metadata:")
print(f"  Title: {result['metadata']['title']}")
print(f"  Artist: {result['metadata']['artist']}")
print(f"  Confidence: {result['metadata']['confidence_score']}")