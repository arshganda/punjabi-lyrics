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
    
    def extract_gurmukhi_lyrics_and_metadata(self, audio_path: str) -> Dict:
        """Extract Gurmukhi lyrics and metadata"""
        print("Extracting Gurmukhi lyrics and metadata with Gemini...")
        
        # Upload audio file
        audio_file = genai.upload_file(path=audio_path)
        
        prompt = """<task>
<role>You are an expert in Punjabi music and Gurmukhi script with perfect transcription abilities.</role>

<objective>Listen to this Punjabi song and extract both lyrics and metadata accurately.</objective>

<examples>
<example_1>
Audio: Folk song with clear vocals
<output>
<title>Mera Laung Gawacha</title>
<artist>Satinder Sartaaj</artist>
<lyrics_gurmukhi>
ਮੇਰਾ ਲੌਂਗ ਗਵਾਚਾ ਮਾਹੀ
ਤੇਰੇ ਵਿਚ ਦੇਸ਼ ਦੇ ਸਾਹੀ
ਮੇਰਾ ਲੌਂਗ ਗਵਾਚਾ ਮਾਹੀ
ਤੇਰੇ ਵਿਚ ਦੇਸ਼ ਦੇ ਸਾਹੀ
</lyrics_gurmukhi>
</output>
</example_1>

<example_2>
Audio: Modern song with unclear artist
<output>
<title>Jatt Da Muqabala</title>
<artist>Unknown</artist>
<lyrics_gurmukhi>
ਜੱਟ ਦਾ ਮੁਕਾਬਲਾ ਕੌਣ ਕਰੇਗਾ
ਸ਼ੇਰ ਦਾ ਸਾਹਮਣਾ ਕੌਣ ਕਰੇਗਾ
[?] ਸ਼ਬਦ ਸਪਸ਼ਟ ਨਹੀਂ
ਜੱਟ ਦਾ ਮੁਕਾਬਲਾ ਕੌਣ ਕਰੇਗਾ
</lyrics_gurmukhi>
</output>
</example_2>
</examples>

<constraints>
<must_do>
- Transcribe ONLY sung lyrics in clean Gurmukhi script
- Each lyric line on separate line
- Include repeated sections each occurrence
- Use [?] only for genuinely unclear words
- Preserve exact vocal phrasing and repetition
</must_do>

<must_not_do>
- Include instrumental sections or music descriptions
- Add markdown formatting, code blocks, or JSON
- Guess unclear lyrics instead of using [?]
- Add explanations or commentary
- Mix languages within the Gurmukhi text
</must_not_do>
</constraints>

<metadata_rules>
<title>Use recognizable song title or "Unknown" if unclear</title>
<artist>Use recognizable artist name or "Unknown" if unclear</artist>
<when_uncertain>Always prefer "Unknown" over guessing</when_uncertain>
</metadata_rules>

<output_format>
GOOD:
<title>Mera Laung Gawacha</title>
<artist>Satinder Sartaaj</artist>
<lyrics_gurmukhi>
ਮੇਰਾ ਲੌਂਗ ਗਵਾਚਾ ਮਾਹੀ
ਤੇਰੇ ਵਿਚ ਦੇਸ਼ ਦੇ ਸਾਹੀ
</lyrics_gurmukhi>

BAD:
```
{
  "title": "Mera Laung Gawacha",
  "lyrics": "ਮੇਰਾ ਲੌਂਗ ਗਵਾਚਾ ਮਾਹੀ\nਤੇਰੇ ਵਿਚ ਦੇਸ਼ ਦੇ ਸਾਹੀ"
}
```

BAD:
**Title:** Mera Laung Gawacha
**Artist:** Satinder Sartaaj
ਮੇਰਾ ਲੌਂਗ ਗਵਾਚਾ ਮਾਹੀ
</output_format>
</task>"""
        
        response_text = self.validate_and_retry_response(prompt, audio_file)
        
        # Parse XML-structured response
        title = "Unknown"
        artist = "Unknown"
        lyrics_text = ""
        
        # Extract title
        title_match = re.search(r'<title>(.*?)</title>', response_text, re.DOTALL)
        if title_match:
            title = title_match.group(1).strip()
        
        # Extract artist
        artist_match = re.search(r'<artist>(.*?)</artist>', response_text, re.DOTALL)
        if artist_match:
            artist = artist_match.group(1).strip()
        
        # Extract lyrics
        lyrics_match = re.search(r'<lyrics_gurmukhi>(.*?)</lyrics_gurmukhi>', response_text, re.DOTALL)
        if lyrics_match:
            lyrics_text = lyrics_match.group(1).strip()
        else:
            # Fallback: use entire response if XML parsing fails
            lyrics_text = self.clean_text_response(response_text)
        
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
        
        prompt = f"""<task>
<role>You are a Punjabi language expert specializing in Gurmukhi script correction and standardization.</role>

<objective>Review and correct this Gurmukhi transcription while preserving its original structure and meaning.</objective>

<context>
<song_title>{lyrics_data.get('title', 'Unknown')}</song_title>
<artist_name>{lyrics_data.get('artist', 'Unknown')}</artist_name>
</context>

<examples>
<example_1>
<input_with_errors>
ਮੇਰਾ ਲੌਗ ਗਵਾਚਾ ਮਾਹੀ
ਤੇਰੇ ਵਿੱਚ ਦੇਸ ਦੇ ਸਾਹੀ
</input_with_errors>
<corrected_output>
ਮੇਰਾ ਲੌਂਗ ਗਵਾਚਾ ਮਾਹੀ
ਤੇਰੇ ਵਿਚ ਦੇਸ਼ ਦੇ ਸਾਹੀ
</corrected_output>
<corrections_made>Missing ਂ in ਲੌਂਗ, wrong ਵਿੱਚ should be ਵਿਚ, missing ਼ in ਦੇਸ਼</corrections_made>
</example_1>

<example_2>
<input_with_errors>
ਜਟ ਦਾ ਮੁਕਾਬਲਾ ਕੋਣ ਕਰੇਗਾ
ਸੇਰ ਦਾ ਸਾਮਨਾ ਕੋਣ ਕਰੇਗਾ
</input_with_errors>
<corrected_output>
ਜੱਟ ਦਾ ਮੁਕਾਬਲਾ ਕੌਣ ਕਰੇਗਾ
ਸ਼ੇਰ ਦਾ ਸਾਹਮਣਾ ਕੌਣ ਕਰੇਗਾ
</corrected_output>
<corrections_made>Missing ੱ in ਜੱਟ, ਕੋਣ to ਕੌਣ, missing ਼ in ਸ਼ੇਰ, ਸਾਮਨਾ to ਸਾਹਮਣਾ</corrections_made>
</example_2>
</examples>

<input_lyrics>
{lyrics_data['lyrics_gurmukhi']}
</input_lyrics>

<correction_categories>
<spelling_errors>
- Common misspellings in Punjabi words
- Wrong vowel combinations
- Incorrect consonant clusters
</spelling_errors>

<diacritic_corrections>
<lagaan_maatra>ੱ - double consonant marker (ਜੱਟ, ਪੱਤਰ)</lagaan_maatra>
<bindi>ਂ - nasal sound (ਲੌਂਗ, ਮਾਂ)</bindi>
<tippi>ੰ - nasal sound (ਰੰਗ, ਗੰਗਾ)</tippi>
<nukta>਼ - dot below for Persian/Arabic sounds (ਸ਼, ਜ਼, ਖ਼, ਫ਼)</nukta>
</diacritic_corrections>

<transcription_errors>
<common_mistakes>ਸ਼/ਸ, ਜ਼/ਜ, ਖ਼/ਖ, ਫ਼/ਫ, ਵ/ਬ, ਰ/ੜ</common_mistakes>
<vowel_errors>ਿ/ੀ, ੁ/ੂ, ੇ/ੈ, ੋ/ੌ confusion</vowel_errors>
</transcription_errors>

<structural_preservation>
<line_breaks>Maintain exact same number of lines</line_breaks>
<word_spacing>Proper spacing between words</word_spacing>
<punctuation>Keep existing punctuation marks</punctuation>
<unclear_sections>Preserve [?] markers for unclear words</unclear_sections>
</structural_preservation>
</correction_categories>

<constraints>
<must_preserve>
- Original line structure and count
- Overall meaning and flow
- [?] markers for unclear sections
- Poetic rhythm and meter
</must_preserve>

<must_fix>
- Spelling errors in standard Punjabi words
- Missing or incorrect diacritical marks
- Wrong nukta usage for Persian/Arabic sounds
- Improper word spacing
</must_fix>

<must_not_do>
- Change the meaning of words
- Add or remove lines
- Add explanatory text
- Use markdown or formatting
- Modernize archaic spellings unnecessarily
</must_not_do>
</constraints>

<output_format>
GOOD:
ਮੇਰਾ ਲੌਂਗ ਗਵਾਚਾ ਮਾਹੀ
ਤੇਰੇ ਵਿਚ ਦੇਸ਼ ਦੇ ਸਾਹੀ

BAD:
```gurmukhi
ਮੇਰਾ ਲੌਂਗ ਗਵਾਚਾ ਮਾਹੀ
ਤੇਰੇ ਵਿਚ ਦੇਸ਼ ਦੇ ਸਾਹੀ
```

BAD:
**Corrected lyrics:**
ਮੇਰਾ ਲੌਂਗ ਗਵਾਚਾ ਮਾਹੀ

BAD:
Here are the corrected lyrics: ਮੇਰਾ ਲੌਂਗ ਗਵਾਚਾ ਮਾਹੀ
</output_format>
</task>"""
        
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
        
        prompt = f"""<task>
<role>You are a expert literary translator specializing in Punjabi-English translation with deep cultural knowledge.</role>

<objective>Translate this Punjabi song from Gurmukhi to English while preserving its poetic essence, cultural nuances, and emotional depth.</objective>

<context>
<song_title>{context.get('title', 'Unknown')}</song_title>
<artist_name>{context.get('artist', 'Unknown')}</artist_name>
<genre>Traditional/Folk/Modern Punjabi music</genre>
</context>

<examples>
<example_1>
<gurmukhi_input>
ਮੇਰਾ ਲੌਂਗ ਗਵਾਚਾ ਮਾਹੀ
ਤੇਰੇ ਵਿਚ ਦੇਸ਼ ਦੇ ਸਾਹੀ
</gurmukhi_input>
<english_output>
My earring got lost, beloved
In your country, oh king
</english_output>
<translation_notes>ਮਾਹੀ = beloved/dear one, ਸਾਹੀ = king/ruler - preserved emotional intimacy</translation_notes>
</example_1>

<example_2>
<gurmukhi_input>
ਜੱਟ ਦਾ ਮੁਕਾਬਲਾ ਕੌਣ ਕਰੇਗਾ
ਸ਼ੇਰ ਦਾ ਸਾਹਮਣਾ ਕੌਣ ਕਰੇਗਾ
</gurmukhi_input>
<english_output>
Who will compete with this Jatt [Punjabi farmer-warrior]
Who will face this lion
</english_output>
<translation_notes>ਜੱਟ kept with cultural explanation, ਸ਼ੇਰ = lion (metaphor for brave person)</translation_notes>
</example_2>

<example_3>
<gurmukhi_input>
ਸਤਿਗੁਰੂ ਨਾਨਕ ਪ੍ਰਗਟਿਆ ਮਿਟੀ ਧੁੰਧ ਜਗ ਚਾਨਣ ਹੋਆ
[?] ਸ਼ਬਦ ਸਪਸ਼ਟ ਨਹੀਂ
</gurmukhi_input>
<english_output>
True Guru Nanak appeared, the mist cleared and the world became illuminated
[?] word unclear
</english_output>
<translation_notes>Religious context preserved, [?] maintained for unclear words</translation_notes>
</example_3>
</examples>

<input_gurmukhi>
{gurmukhi_text}
</input_gurmukhi>

<translation_principles>
<poetic_preservation>
<rhythm>Maintain natural rhythm and flow in English</rhythm>
<imagery>Preserve metaphors and poetic imagery</imagery>
<emotion>Capture the emotional tone and intensity</emotion>
<repetition>Maintain repetitive elements and refrains</repetition>
</poetic_preservation>

<cultural_handling>
<religious_terms>Preserve Sikh/Hindu/Islamic religious terms with context</religious_terms>
<punjabi_concepts>Keep uniquely Punjabi concepts with brief explanations in brackets</punjabi_concepts>
<honorifics>Translate titles and honorifics appropriately (ਜੀ, ਸਾਹਿਬ, etc.)</honorifics>
<regional_refs>Maintain references to Punjab, villages, etc.</regional_refs>
</cultural_handling>

<linguistic_accuracy>
<idioms>Convert Punjabi idioms to equivalent English expressions where possible</idioms>
<wordplay>Attempt to preserve puns and wordplay when feasible</wordplay>
<archaic_terms>Handle old Punjabi terms with appropriate English equivalents</archaic_terms>
<unclear_words>Preserve [?] markers exactly as they appear</unclear_words>
</linguistic_accuracy>

<structural_requirements>
<line_correspondence>One English line for each Gurmukhi line</line_correspondence>
<line_count>Same number of lines as input</line_count>
<spacing>Maintain blank lines and spacing from original</spacing>
<punctuation>Add appropriate English punctuation for clarity</punctuation>
</structural_requirements>
</translation_principles>

<constraints>
<must_preserve>
- Exact line structure and count
- Cultural and religious significance
- Emotional depth and poetic beauty
- [?] markers for unclear sections
- Overall meaning and message
</must_preserve>

<must_achieve>
- Natural English flow and readability
- Cultural accessibility for English speakers
- Appropriate register (formal/informal/poetic)
- Consistent terminology throughout
</must_achieve>

<must_avoid>
- Literal word-for-word translation
- Loss of cultural context
- Awkward or unnatural English phrasing
- Adding explanatory text outside brackets
- Changing the fundamental meaning
</must_avoid>
</constraints>

<output_format>
GOOD:
My earring got lost, beloved
In your country, oh king

BAD:
```
My earring got lost, beloved
In your country, oh king
```

BAD:
**English Translation:**
My earring got lost, beloved
In your country, oh king

BAD:
Here is the English translation:
My earring got lost, beloved
In your country, oh king
</output_format>
</task>"""
        
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
        
        # Preprocess audio
        processed_audio = self.preprocess_audio(mp3_path)
        
        try:
            # Extract Gurmukhi lyrics and metadata
            lyrics_data = self.extract_gurmukhi_lyrics_and_metadata(processed_audio)
            
            # Autocorrect Gurmukhi
            lyrics_data = self.autocorrect_gurmukhi(lyrics_data)
            
            # Romanize text
            romanized = self.romanize_text(lyrics_data['lyrics_gurmukhi_corrected'])
            
            # Translate to English
            english = self.translate_to_english(
                lyrics_data['lyrics_gurmukhi_corrected'],
                lyrics_data
            )
            
            # Create structured output
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
        print("Usage: uv run python punjabi_lyrics_extractor.py <mp3_file>")
        print("Output will be saved to: outputs/<filename>_lyrics.json")
        sys.exit(1)
    
    mp3_file = sys.argv[1]
    
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
    result = extractor.process_song(mp3_file)
    
    # Save to outputs folder
    outputs_dir = Path("outputs")
    outputs_dir.mkdir(exist_ok=True)
    
    # Save result with clean naming
    output_file = outputs_dir / f"{Path(mp3_file).stem}_lyrics.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\nOutput saved to: {output_file}")


if __name__ == "__main__":
    main()