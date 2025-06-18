#!/usr/bin/env python3
"""
Punjabi Lyrics Extractor
Extracts and translates Punjabi song lyrics from MP3 files using Gemini AI
"""

import os
import json
import sys
import re
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
        self.model = genai.GenerativeModel('gemini-2.5-flash')
        self.translator = Translator()
        
    def clean_text_response(self, text: str) -> str:
        """Clean LLM response text from formatting artifacts"""
        if not text:
            return ""
            
        # Remove markdown code blocks
        text = re.sub(r'```[a-zA-Z]*\n?', '', text)
        text = re.sub(r'```\n?', '', text)
        
        # Remove JSON structure artifacts
        text = re.sub(r'^\s*\{.*?"lyrics_gurmukhi":\s*"', '', text, flags=re.DOTALL)
        text = re.sub(r'",?\s*\}?\s*$', '', text)
        text = re.sub(r'^\s*"([^"]*)":\s*"', '', text)
        
        # Clean up escaped characters
        text = text.replace('\\n', '\n')
        text = text.replace('\\"', '"')
        
        # Remove leading/trailing quotes
        text = text.strip().strip('"\'')
        
        # Clean up multiple newlines
        text = re.sub(r'\n\s*\n', '\n', text)
        
        return text.strip()
    
    def validate_and_retry_response(self, prompt: str, audio_file, max_retries: int = 2) -> str:
        """Validate LLM response and retry if needed"""
        for attempt in range(max_retries + 1):
            try:
                response = self.model.generate_content([prompt, audio_file])
                cleaned_text = self.clean_text_response(response.text)
                
                # Basic validation
                if len(cleaned_text) > 20 and not cleaned_text.startswith('{'):
                    return cleaned_text
                    
                print(f"Response validation failed (attempt {attempt + 1}), retrying...")
                
            except Exception as e:
                print(f"API call failed (attempt {attempt + 1}): {e}")
                
        # Return last attempt even if not perfect
        return cleaned_text if 'cleaned_text' in locals() else ""
        
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
            print(f"Running sox noise reduction...")
            print(f"Input file: {temp_wav.name}")
            print(f"Output file: {processed_wav.name}")
            
            result = subprocess.run([
                'sox', temp_wav.name, processed_wav.name,
                'compand', '0.3,1', '6:-70,-60,-20', '-5', '-90', '0.2'
            ], check=True, capture_output=True, timeout=30)
            
            print(f"Sox completed successfully")
            os.unlink(temp_wav.name)
            return processed_wav.name
        except subprocess.TimeoutExpired:
            print("Sox command timed out after 30 seconds")
            return temp_wav.name
        except subprocess.CalledProcessError as e:
            print(f"Sox command failed: {e}")
            print(f"Stderr: {e.stderr.decode() if e.stderr else 'None'}")
            return temp_wav.name
        except FileNotFoundError:
            print("Sox not found, skipping advanced audio processing")
            return temp_wav.name
    
    def extract_gurmukhi_lyrics(self, audio_path: str) -> Dict:
        """Extract Gurmukhi lyrics using Gemini"""
        print("Extracting Gurmukhi lyrics with Gemini...")
        
        # Upload audio file
        audio_file = genai.upload_file(path=audio_path)
        
        prompt = """You are an expert in Punjabi music and Gurmukhi script.

        CRITICAL INSTRUCTIONS:
        - Output ONLY clean Gurmukhi text, NO markdown formatting
        - NO code blocks (```), NO JSON structure, NO extra formatting
        - Just the raw Gurmukhi lyrics with line breaks
        
        Listen to this Punjabi song and transcribe ONLY the sung lyrics in Gurmukhi script (ਪੰਜਾਬੀ):
        1. Write each line of lyrics on a separate line
        2. Ignore instrumental sections
        3. Include repeated sections each time they appear
        4. Use [?] for unclear words only
        5. No extra text, explanations, or formatting
        
        Example output format:
        ਪਹਿਲੀ ਲਾਈਨ ਇੱਥੇ
        ਦੂਜੀ ਲਾਈਨ ਇੱਥੇ
        ਤੀਜੀ ਲਾਈਨ ਇੱਥੇ
        
        Output only clean Gurmukhi text."""
        
        lyrics_text = self.validate_and_retry_response(prompt, audio_file)
        
        # Try to extract title/artist with a separate prompt
        meta_prompt = """Listen to this Punjabi song and identify:
        - Song title (if recognizable)
        - Artist name (if recognizable)
        
        Respond in this exact format:
        Title: [song title or "Unknown"]
        Artist: [artist name or "Unknown"]"""
        
        try:
            meta_response = self.model.generate_content([meta_prompt, audio_file])
            meta_text = meta_response.text.strip()
            
            title = "Unknown"
            artist = "Unknown"
            
            title_match = re.search(r'Title:\s*(.+)', meta_text)
            artist_match = re.search(r'Artist:\s*(.+)', meta_text)
            
            if title_match:
                title = title_match.group(1).strip()
            if artist_match:
                artist = artist_match.group(1).strip()
                
        except:
            title = "Unknown"
            artist = "Unknown"
        
        return {
            "title": title,
            "artist": artist,
            "lyrics_gurmukhi": lyrics_text,
            "sections": [],
            "confidence": 0.8 if len(lyrics_text) > 50 else 0.3,
            "unclear_sections": []
        }
    
    def autocorrect_gurmukhi(self, lyrics_data: Dict) -> Dict:
        """Autocorrect Gurmukhi text based on context"""
        print("Autocorrecting Gurmukhi text...")
        
        prompt = f"""You are an expert in Punjabi language and music.

        CRITICAL INSTRUCTIONS:
        - Output ONLY clean corrected Gurmukhi text
        - NO markdown formatting, NO explanations, NO extra text
        - Keep the same line structure as input
        
        Review and correct this Gurmukhi transcription:
        
        Title: {lyrics_data.get('title', 'Unknown')}
        Artist: {lyrics_data.get('artist', 'Unknown')}
        
        Lyrics to correct:
        {lyrics_data['lyrics_gurmukhi']}
        
        Fix:
        1. Spelling mistakes in Gurmukhi
        2. Common transcription errors (ਸ਼/ਸ, ਜ਼/ਜ, etc.)
        3. Proper use of lagaan maatra (ੱ), bindi (ਂ), tippi (ੰ)
        4. Word boundaries and spacing
        5. Maintain original line structure
        
        Output only the corrected Gurmukhi text with same line breaks."""
        
        corrected_text = self.clean_text_response(self.model.generate_content(prompt).text)
        lyrics_data['lyrics_gurmukhi_corrected'] = corrected_text
        
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

        CRITICAL INSTRUCTIONS:
        - Output ONLY the English translation
        - NO markdown formatting, NO explanations, NO extra text
        - Keep the same line structure as the Gurmukhi input
        - One English line for each Gurmukhi line
        
        Context:
        - Title: {context.get('title', 'Unknown')}
        - Artist: {context.get('artist', 'Unknown')}
        
        Gurmukhi text to translate:
        {gurmukhi_text}
        
        Guidelines:
        1. Preserve poetic meaning and emotion
        2. Keep cultural references with brief explanations in brackets
        3. Maintain exact line structure (line-by-line translation)
        4. Translate idioms appropriately
        5. Natural English flow
        
        Output only the English translation with same line breaks."""
        
        english_text = self.clean_text_response(self.model.generate_content(prompt).text)
        return english_text
    
    def smart_line_alignment(self, gurmukhi_text: str, romanized_text: str, english_text: str) -> List[Dict]:
        """Smart line alignment with filtering and validation"""
        
        # Clean and split lines
        gurmukhi_lines = [line.strip() for line in gurmukhi_text.split('\n') if line.strip()]
        romanized_lines = [line.strip() for line in romanized_text.split('\n') if line.strip()]
        english_lines = [line.strip() for line in english_text.split('\n') if line.strip()]
        
        # Get the minimum non-zero count to avoid empty padding
        line_counts = [len(lines) for lines in [gurmukhi_lines, romanized_lines, english_lines] if lines]
        if not line_counts:
            return []
            
        target_count = max(line_counts)
        
        # Pad shorter lists with empty strings
        while len(gurmukhi_lines) < target_count:
            gurmukhi_lines.append("")
        while len(romanized_lines) < target_count:
            romanized_lines.append("")
        while len(english_lines) < target_count:
            english_lines.append("")
            
        # Create aligned line data
        lines_data = []
        for i in range(target_count):
            # Skip lines that are artifacts or clearly broken
            gurmukhi = gurmukhi_lines[i] if i < len(gurmukhi_lines) else ""
            romanized = romanized_lines[i] if i < len(romanized_lines) else ""
            english = english_lines[i] if i < len(english_lines) else ""
            
            # Filter out obvious artifacts
            if any(artifact in gurmukhi.lower() for artifact in ['```', 'json', '{']):
                continue
            if any(artifact in romanized.lower() for artifact in ['```', 'json', '{']):
                continue
                
            line_data = {
                "line_number": len(lines_data) + 1,
                "gurmukhi": gurmukhi,
                "romanized": romanized,
                "english": english
            }
            lines_data.append(line_data)
            
        return lines_data

    def create_structured_output(self, lyrics_data: Dict, romanized: str, english: str) -> Dict:
        """Create structured output for lyrics website"""
        
        # Use smart line alignment
        lines_data = self.smart_line_alignment(
            lyrics_data['lyrics_gurmukhi_corrected'],
            romanized,
            english
        )
        
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
    
    # Also save to a default location with v2 naming for comparison
    default_output = Path(mp3_file).stem + "_lyrics_v2.json"
    with open(default_output, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\nAlso saved to: {default_output}")
    
    # Compare with original if it exists
    original_file = Path(mp3_file).stem + "_lyrics.json"
    if Path(original_file).exists():
        try:
            from quality_validator import LyricsQualityValidator
            validator = LyricsQualityValidator()
            
            with open(original_file, 'r', encoding='utf-8') as f:
                old_result = json.load(f)
            
            comparison = validator.compare_results(old_result, result)
            
            print(f"\n=== QUALITY COMPARISON ===")
            print(f"Original score: {comparison['old_scores']['overall']:.2f}")
            print(f"New (v2) score: {comparison['new_scores']['overall']:.2f}")
            print(f"Improvement: {comparison['improvements']['overall']['change']:.2f}")
            
        except Exception as e:
            print(f"Could not compare with original: {e}")


if __name__ == "__main__":
    main()