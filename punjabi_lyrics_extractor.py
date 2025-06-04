#!/usr/bin/env python3
"""
Punjabi Lyrics Extractor
Extracts and translates Punjabi song lyrics from MP3 files using Gemini AI
"""

import os
import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import tempfile
import subprocess

# Audio processing
from pydub import AudioSegment
from pydub.silence import detect_nonsilent
import librosa
import soundfile as sf

# AI and language processing
import google.generativeai as genai
from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate

# For translation fallback
from googletrans import Translator


class PunjabiLyricsExtractor:
    def __init__(self, gemini_api_key: str):
        """Initialize the extractor with Gemini API key"""
        self.api_key = gemini_api_key
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel('gemini-2.0-flash')
        self.translator = Translator()
        
    def preprocess_audio(self, mp3_path: str) -> str:
        """Preprocess audio for better transcription"""
        print(f"Preprocessing audio: {mp3_path}")
        
        # Load audio
        audio = AudioSegment.from_mp3(mp3_path)
        
        # Convert to mono
        audio = audio.set_channels(1)
        
        # Normalize volume
        target_dBFS = -20.0
        change_in_dBFS = target_dBFS - audio.dBFS
        audio = audio.apply_gain(change_in_dBFS)
        
        # Export as WAV for better compatibility
        temp_wav = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        audio.export(temp_wav.name, format='wav')
        
        # Optional: Apply noise reduction using sox (if installed)
        try:
            processed_wav = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
            subprocess.run([
                'sox', temp_wav.name, processed_wav.name,
                'noisered', '-', '0.21',
                'compand', '0.3,1', '6:-70,-60,-20', '-5', '-90', '0.2'
            ], check=True, capture_output=True)
            os.unlink(temp_wav.name)
            return processed_wav.name
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("Sox not found, skipping advanced audio processing")
            return temp_wav.name
    
    def extract_gurmukhi_lyrics(self, audio_path: str) -> Dict:
        """Extract Gurmukhi lyrics using Gemini"""
        print("Extracting Gurmukhi lyrics with Gemini...")
        
        # Upload audio file
        audio_file = genai.upload_file(path=audio_path)
        
        prompt = """You are an expert in Punjabi music and Gurmukhi script. 
        
        Listen to this Punjabi song and:
        1. Transcribe ONLY the sung lyrics in Gurmukhi script (ਪੰਜਾਬੀ)
        2. Ignore any instrumental sections
        3. Format with proper line breaks for verses/chorus
        4. Include repeated sections only once with notation like [Chorus x2]
        5. Note any unclear words with [?]
        
        Output format:
        {
            "title": "Song title if recognizable",
            "artist": "Artist if recognizable", 
            "lyrics_gurmukhi": "Full Gurmukhi text with line breaks",
            "sections": [
                {
                    "type": "verse/chorus/hook",
                    "text": "Gurmukhi text for this section",
                    "repeat_count": 1
                }
            ],
            "confidence": 0.0-1.0,
            "unclear_sections": ["List of unclear parts"]
        }
        
        Output valid JSON only."""
        
        response = self.model.generate_content([prompt, audio_file])
        
        try:
            # Parse JSON from response
            result = json.loads(response.text)
            return result
        except json.JSONDecodeError:
            # Fallback parsing
            return {
                "title": "Unknown",
                "artist": "Unknown",
                "lyrics_gurmukhi": response.text,
                "sections": [],
                "confidence": 0.5,
                "unclear_sections": []
            }
    
    def autocorrect_gurmukhi(self, lyrics_data: Dict) -> Dict:
        """Autocorrect Gurmukhi text based on context"""
        print("Autocorrecting Gurmukhi text...")
        
        prompt = f"""You are an expert in Punjabi language and music.
        
        Review and correct this Gurmukhi transcription:
        
        Title: {lyrics_data.get('title', 'Unknown')}
        Artist: {lyrics_data.get('artist', 'Unknown')}
        
        Lyrics:
        {lyrics_data['lyrics_gurmukhi']}
        
        Tasks:
        1. Fix any spelling mistakes in Gurmukhi
        2. Correct common transcription errors (ਸ਼/ਸ, ਜ਼/ਜ, etc.)
        3. Ensure proper use of lagaan maatra (ੱ), bindi (ਂ), tippi (ੰ)
        4. Fix word boundaries and spacing
        5. Maintain poetic structure
        
        Return the corrected text only."""
        
        response = self.model.generate_content(prompt)
        lyrics_data['lyrics_gurmukhi_corrected'] = response.text.strip()
        
        return lyrics_data
    
    def romanize_text(self, gurmukhi_text: str) -> str:
        """Convert Gurmukhi to romanized text"""
        print("Romanizing text...")
        
        # Use indic-transliteration library
        romanized = transliterate(
            gurmukhi_text,
            sanscript.GURMUKHI,
            sanscript.ISO
        )
        
        # Post-process for better readability
        romanized = romanized.replace('ñ', 'n')
        romanized = romanized.replace('ṅ', 'ng')
        
        return romanized
    
    def translate_to_english(self, gurmukhi_text: str, context: Dict) -> str:
        """Translate Gurmukhi to English with context awareness"""
        print("Translating to English...")
        
        prompt = f"""Translate this Punjabi song from Gurmukhi to English.
        
        Context:
        - Title: {context.get('title', 'Unknown')}
        - Artist: {context.get('artist', 'Unknown')}
        
        Gurmukhi text:
        {gurmukhi_text}
        
        Guidelines:
        1. Preserve the poetic meaning and emotion
        2. Keep cultural references with explanations in brackets
        3. Maintain verse structure
        4. Translate idioms appropriately
        5. Keep the flow natural in English
        
        Provide only the English translation."""
        
        response = self.model.generate_content(prompt)
        return response.text.strip()
    
    def create_structured_output(self, lyrics_data: Dict, romanized: str, english: str) -> Dict:
        """Create structured output for lyrics website"""
        
        # Split into lines for line-by-line mapping
        gurmukhi_lines = lyrics_data['lyrics_gurmukhi_corrected'].split('\n')
        romanized_lines = romanized.split('\n')
        english_lines = english.split('\n')
        
        # Create line-by-line mapping
        lines_data = []
        for i in range(max(len(gurmukhi_lines), len(romanized_lines), len(english_lines))):
            line_data = {
                "line_number": i + 1,
                "gurmukhi": gurmukhi_lines[i] if i < len(gurmukhi_lines) else "",
                "romanized": romanized_lines[i] if i < len(romanized_lines) else "",
                "english": english_lines[i] if i < len(english_lines) else ""
            }
            lines_data.append(line_data)
        
        structured_output = {
            "metadata": {
                "title": lyrics_data.get('title', 'Unknown'),
                "artist": lyrics_data.get('artist', 'Unknown'),
                "processed_date": datetime.now().isoformat(),
                "confidence_score": lyrics_data.get('confidence', 0.0)
            },
            "full_text": {
                "gurmukhi": lyrics_data['lyrics_gurmukhi_corrected'],
                "romanized": romanized,
                "english": english
            },
            "lines": lines_data,
            "sections": lyrics_data.get('sections', []),
            "processing_notes": {
                "unclear_sections": lyrics_data.get('unclear_sections', []),
                "autocorrections_made": len(lyrics_data.get('lyrics_gurmukhi', '')) != len(lyrics_data.get('lyrics_gurmukhi_corrected', ''))
            }
        }
        
        return structured_output
    
    def process_song(self, mp3_path: str, output_path: Optional[str] = None) -> Dict:
        """Main processing pipeline"""
        print(f"\n{'='*50}")
        print(f"Processing: {Path(mp3_path).name}")
        print(f"{'='*50}\n")
        
        # Step 1: Preprocess audio
        processed_audio = self.preprocess_audio(mp3_path)
        
        try:
            # Step 2: Extract Gurmukhi lyrics
            lyrics_data = self.extract_gurmukhi_lyrics(processed_audio)
            
            # Step 3: Autocorrect Gurmukhi
            lyrics_data = self.autocorrect_gurmukhi(lyrics_data)
            
            # Step 4: Romanize
            romanized = self.romanize_text(lyrics_data['lyrics_gurmukhi_corrected'])
            
            # Step 5: Translate to English
            english = self.translate_to_english(
                lyrics_data['lyrics_gurmukhi_corrected'],
                lyrics_data
            )
            
            # Step 6: Create structured output
            result = self.create_structured_output(lyrics_data, romanized, english)
            
            # Save to file if output path provided
            if output_path:
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(result, f, ensure_ascii=False, indent=2)
                print(f"\nOutput saved to: {output_path}")
            
            # Print summary to terminal
            self.print_summary(result)
            
            return result
            
        finally:
            # Cleanup temporary files
            if os.path.exists(processed_audio):
                os.unlink(processed_audio)
    
    def print_summary(self, result: Dict):
        """Print a summary of the extracted lyrics"""
        print("\n" + "="*50)
        print("EXTRACTION COMPLETE")
        print("="*50)
        print(f"Title: {result['metadata']['title']}")
        print(f"Artist: {result['metadata']['artist']}")
        print(f"Confidence: {result['metadata']['confidence_score']:.2f}")
        print(f"Lines extracted: {len(result['lines'])}")
        
        print("\nFirst few lines:")
        print("-"*30)
        for line in result['lines'][:3]:
            print(f"ਪੰਜਾਬੀ: {line['gurmukhi']}")
            print(f"Roman: {line['romanized']}")
            print(f"English: {line['english']}")
            print()


def main():
    """CLI interface"""
    if len(sys.argv) < 2:
        print("Usage: python punjabi_lyrics_extractor.py <mp3_file> [output_json]")
        sys.exit(1)
    
    mp3_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    # Load environment variables from .env file
    from dotenv import load_dotenv
    load_dotenv()
    
    # Get API key from environment
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        print("Error: Please set GEMINI_API_KEY in .env file or as environment variable")
        sys.exit(1)
    
    # Process the song
    extractor = PunjabiLyricsExtractor(api_key)
    result = extractor.process_song(mp3_file, output_file)
    
    # Also save to a default location for persistence
    default_output = Path(mp3_file).stem + "_lyrics.json"
    with open(default_output, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\nAlso saved to: {default_output}")


if __name__ == "__main__":
    main()