#!/usr/bin/env python3
"""
Quality Validation and Scoring for Punjabi Lyrics JSON Output
"""

import json
import re
from typing import Dict, List, Tuple
from pathlib import Path


class LyricsQualityValidator:
    def __init__(self):
        self.formatting_artifacts = [
            r'```', r'```json', r'```\n', 
            r'\{[\s\n]*"title":', r'\}[\s\n]*$',
            r'^\s*\{', r'\}\s*$'
        ]
        
        # Gurmukhi script quality patterns
        self.gurmukhi_quality_patterns = {
            'proper_diacritics': [r'ੱ', r'ਂ', r'ੰ', r'਼'],  # lagaan, bindi, tippi, nukta
            'common_words': [r'ਜੱਟ', r'ਪੰਜਾਬੀ', r'ਸਿੰਘ', r'ਕੌਰ'],  # proper spellings
            'conjuncts': [r'ਤ੍ਰ', r'ਪ੍ਰ', r'ਦ੍ਰ', r'ਸ੍ਰ'],  # conjunct consonants
        }
        
        # Translation quality indicators
        self.translation_quality_indicators = {
            'cultural_preservation': [r'\[.*?\]', r'Jatt', r'Punjab', r'Gurmukhi'],
            'poetic_language': [r'beloved', r'oh king', r'my dear', r'intoxication'],
            'natural_flow': [r"don't", r"they're", r"we're", r"can't"],  # contractions
        }
        
    def clean_text_score(self, text: str) -> float:
        """Score text cleanliness (0-1, higher is better)"""
        if not text or not text.strip():
            return 0.0
            
        artifact_count = 0
        for pattern in self.formatting_artifacts:
            artifact_count += len(re.findall(pattern, text, re.MULTILINE))
        
        # Penalty for artifacts
        penalty = min(artifact_count * 0.2, 0.8)
        return max(0.0, 1.0 - penalty)
    
    def line_alignment_score(self, lines_data: List[Dict]) -> float:
        """Score line-by-line alignment quality (0-1)"""
        if not lines_data:
            return 0.0
            
        aligned_lines = 0
        total_lines = len(lines_data)
        
        for line in lines_data:
            gurmukhi = line.get('gurmukhi', '').strip()
            romanized = line.get('romanized', '').strip()
            english = line.get('english', '').strip()
            
            # Check if all three fields have content or are all empty
            has_content = [bool(gurmukhi), bool(romanized), bool(english)]
            if all(has_content) or not any(has_content):
                aligned_lines += 1
        
        return aligned_lines / total_lines if total_lines > 0 else 0.0
    
    def gurmukhi_script_quality(self, gurmukhi_text: str) -> Dict[str, float]:
        """Analyze Gurmukhi script quality (0-1 for each aspect)"""
        if not gurmukhi_text:
            return {'diacritics': 0.0, 'spelling': 0.0, 'authenticity': 0.0}
        
        scores = {}
        
        # Count diacritic usage
        total_chars = len(gurmukhi_text)
        diacritic_chars = sum(len(re.findall(pattern, gurmukhi_text)) 
                             for pattern in self.gurmukhi_quality_patterns['proper_diacritics'])
        scores['diacritics'] = min(diacritic_chars / (total_chars * 0.1), 1.0)  # Expect ~10% diacritics
        
        # Common word spelling accuracy
        word_matches = sum(len(re.findall(pattern, gurmukhi_text)) 
                          for pattern in self.gurmukhi_quality_patterns['common_words'])
        scores['spelling'] = min(word_matches / 5.0, 1.0)  # Normalize by expected count
        
        # Script authenticity (no Latin chars mixed in)
        latin_chars = len(re.findall(r'[a-zA-Z]', gurmukhi_text))
        scores['authenticity'] = max(0.0, 1.0 - (latin_chars / max(total_chars, 1)) * 5)
        
        return scores
    
    def translation_quality_analysis(self, english_text: str) -> Dict[str, float]:
        """Analyze English translation quality (0-1 for each aspect)"""
        if not english_text:
            return {'cultural_preservation': 0.0, 'naturalness': 0.0, 'poetic_quality': 0.0}
        
        scores = {}
        
        # Cultural preservation (bracketed explanations, cultural terms)
        cultural_matches = sum(len(re.findall(pattern, english_text, re.IGNORECASE)) 
                              for pattern in self.translation_quality_indicators['cultural_preservation'])
        scores['cultural_preservation'] = min(cultural_matches / 3.0, 1.0)
        
        # Poetic language quality
        poetic_matches = sum(len(re.findall(pattern, english_text, re.IGNORECASE)) 
                            for pattern in self.translation_quality_indicators['poetic_language'])
        scores['poetic_quality'] = min(poetic_matches / 5.0, 1.0)
        
        # Natural English flow (contractions, natural phrasing)
        natural_matches = sum(len(re.findall(pattern, english_text)) 
                             for pattern in self.translation_quality_indicators['natural_flow'])
        scores['naturalness'] = min(natural_matches / 3.0, 1.0)
        
        return scores
    
    def metadata_accuracy_score(self, result: Dict) -> Dict[str, float]:
        """Analyze metadata accuracy beyond just Unknown/not Unknown"""
        metadata = result.get('metadata', {})
        scores = {}
        
        # Title specificity
        title = metadata.get('title', 'Unknown')
        if title == 'Unknown':
            scores['title_specificity'] = 0.0
        elif len(title) < 5:
            scores['title_specificity'] = 0.3  # Very short title
        elif any(word in title.lower() for word in ['song', 'track', 'untitled']):
            scores['title_specificity'] = 0.5  # Generic title
        else:
            scores['title_specificity'] = 1.0  # Specific title
        
        # Artist identification
        artist = metadata.get('artist', 'Unknown')
        if artist == 'Unknown':
            scores['artist_specificity'] = 0.0
        elif len(artist) < 3:
            scores['artist_specificity'] = 0.3
        else:
            scores['artist_specificity'] = 1.0
        
        # Confidence calibration
        confidence = metadata.get('confidence_score', 0)
        if confidence > 0.8:
            scores['confidence_calibration'] = 1.0
        elif confidence > 0.6:
            scores['confidence_calibration'] = 0.8
        elif confidence > 0.4:
            scores['confidence_calibration'] = 0.6
        else:
            scores['confidence_calibration'] = 0.2
        
        return scores
    
    def content_richness_score(self, result: Dict) -> Dict[str, float]:
        """Measure content richness and detail"""
        scores = {}
        
        lines = result.get('lines', [])
        
        # Line count adequacy
        line_count = len(lines)
        if line_count > 40:
            scores['line_count'] = 1.0
        elif line_count > 20:
            scores['line_count'] = 0.8
        elif line_count > 10:
            scores['line_count'] = 0.6
        else:
            scores['line_count'] = 0.3
        
        # Average line length (content richness)
        if lines:
            avg_gurmukhi_length = sum(len(line.get('gurmukhi', '')) for line in lines) / len(lines)
            avg_english_length = sum(len(line.get('english', '')) for line in lines) / len(lines)
            
            scores['gurmukhi_richness'] = min(avg_gurmukhi_length / 30.0, 1.0)
            scores['english_richness'] = min(avg_english_length / 50.0, 1.0)
        else:
            scores['gurmukhi_richness'] = 0.0
            scores['english_richness'] = 0.0
        
        # Vocabulary diversity (unique words)
        all_english = ' '.join(line.get('english', '') for line in lines)
        words = re.findall(r'\w+', all_english.lower())
        unique_words = len(set(words))
        total_words = len(words)
        
        if total_words > 0:
            scores['vocabulary_diversity'] = min(unique_words / total_words, 1.0)
        else:
            scores['vocabulary_diversity'] = 0.0
        
        return scores
    
    def content_completeness_score(self, result: Dict) -> float:
        """Score overall content completeness (0-1)"""
        score = 0.0
        
        # Check metadata
        metadata = result.get('metadata', {})
        if metadata.get('title') != 'Unknown':
            score += 0.1
        if metadata.get('artist') != 'Unknown':
            score += 0.1
        if metadata.get('confidence_score', 0) > 0.5:
            score += 0.1
            
        # Check full text
        full_text = result.get('full_text', {})
        if len(full_text.get('gurmukhi', '').strip()) > 50:
            score += 0.2
        if len(full_text.get('romanized', '').strip()) > 50:
            score += 0.2
        if len(full_text.get('english', '').strip()) > 50:
            score += 0.3
            
        return min(score, 1.0)
    
    def validate_json_structure(self, result: Dict) -> float:
        """Validate JSON has expected structure (0-1)"""
        required_fields = ['metadata', 'full_text', 'lines']
        score = 0.0
        
        for field in required_fields:
            if field in result:
                score += 0.33
                
        # Check lines structure
        lines = result.get('lines', [])
        if lines and isinstance(lines, list):
            sample_line = lines[0] if lines else {}
            required_line_fields = ['line_number', 'gurmukhi', 'romanized', 'english']
            if all(field in sample_line for field in required_line_fields):
                score += 0.01
                
        return min(score, 1.0)
    
    def calculate_overall_score(self, result: Dict) -> Dict[str, float]:
        """Calculate comprehensive quality score with enhanced discrimination"""
        scores = {}
        
        # Basic structural scores
        scores['structure'] = self.validate_json_structure(result)
        scores['completeness'] = self.content_completeness_score(result)
        scores['line_alignment'] = self.line_alignment_score(result.get('lines', []))
        
        # Enhanced quality analysis
        full_text = result.get('full_text', {})
        
        # Text cleanliness (formatting artifacts)
        scores['gurmukhi_clean'] = self.clean_text_score(full_text.get('gurmukhi', ''))
        scores['romanized_clean'] = self.clean_text_score(full_text.get('romanized', ''))
        scores['english_clean'] = self.clean_text_score(full_text.get('english', ''))
        
        # Gurmukhi script quality analysis
        gurmukhi_scores = self.gurmukhi_script_quality(full_text.get('gurmukhi', ''))
        scores['gurmukhi_diacritics'] = gurmukhi_scores['diacritics']
        scores['gurmukhi_spelling'] = gurmukhi_scores['spelling']
        scores['gurmukhi_authenticity'] = gurmukhi_scores['authenticity']
        
        # Translation quality analysis
        translation_scores = self.translation_quality_analysis(full_text.get('english', ''))
        scores['cultural_preservation'] = translation_scores['cultural_preservation']
        scores['poetic_quality'] = translation_scores['poetic_quality']
        scores['translation_naturalness'] = translation_scores['naturalness']
        
        # Metadata accuracy
        metadata_scores = self.metadata_accuracy_score(result)
        scores['title_specificity'] = metadata_scores['title_specificity']
        scores['artist_specificity'] = metadata_scores['artist_specificity']
        scores['confidence_calibration'] = metadata_scores['confidence_calibration']
        
        # Content richness
        richness_scores = self.content_richness_score(result)
        scores['line_count'] = richness_scores['line_count']
        scores['content_richness'] = (richness_scores['gurmukhi_richness'] + richness_scores['english_richness']) / 2
        scores['vocabulary_diversity'] = richness_scores['vocabulary_diversity']
        
        # Calculate weighted overall score with enhanced discrimination
        weights = {
            'structure': 0.08,
            'completeness': 0.08,
            'line_alignment': 0.08,
            'gurmukhi_clean': 0.06,
            'romanized_clean': 0.04,
            'english_clean': 0.06,
            'gurmukhi_diacritics': 0.08,
            'gurmukhi_spelling': 0.06,
            'gurmukhi_authenticity': 0.04,
            'cultural_preservation': 0.10,
            'poetic_quality': 0.08,
            'translation_naturalness': 0.06,
            'title_specificity': 0.06,
            'artist_specificity': 0.06,
            'confidence_calibration': 0.04,
            'line_count': 0.04,
            'content_richness': 0.04,
            'vocabulary_diversity': 0.04
        }
        
        scores['overall'] = sum(scores[key] * weights[key] for key in weights.keys())
        
        return scores
    
    def compare_results(self, old_result: Dict, new_result: Dict) -> Dict:
        """Compare two results and show improvement"""
        old_scores = self.calculate_overall_score(old_result)
        new_scores = self.calculate_overall_score(new_result)
        
        comparison = {
            'old_scores': old_scores,
            'new_scores': new_scores,
            'improvements': {}
        }
        
        for key in old_scores.keys():
            improvement = new_scores[key] - old_scores[key]
            comparison['improvements'][key] = {
                'change': improvement,
                'percentage': (improvement / old_scores[key] * 100) if old_scores[key] > 0 else 0
            }
            
        return comparison
    
    def generate_report(self, file_path: str, scores: Dict[str, float]) -> str:
        """Generate a human-readable quality report"""
        report = f"\n=== ENHANCED QUALITY REPORT: {Path(file_path).stem} ===\n"
        report += f"Overall Score: {scores['overall']:.3f}/1.000\n\n"
        
        report += "📋 STRUCTURAL QUALITY:\n"
        report += f"  Structure:      {scores['structure']:.3f}\n"
        report += f"  Completeness:   {scores['completeness']:.3f}\n"
        report += f"  Line Alignment: {scores['line_alignment']:.3f}\n\n"
        
        report += "🔤 SCRIPT & TEXT QUALITY:\n"
        report += f"  Gurmukhi Clean:     {scores['gurmukhi_clean']:.3f}\n"
        report += f"  Gurmukhi Diacritics:{scores['gurmukhi_diacritics']:.3f}\n"
        report += f"  Gurmukhi Spelling:  {scores['gurmukhi_spelling']:.3f}\n"
        report += f"  Gurmukhi Authentic: {scores['gurmukhi_authenticity']:.3f}\n"
        report += f"  Romanized Clean:    {scores['romanized_clean']:.3f}\n"
        report += f"  English Clean:      {scores['english_clean']:.3f}\n\n"
        
        report += "🌍 TRANSLATION QUALITY:\n"
        report += f"  Cultural Preservation: {scores['cultural_preservation']:.3f}\n"
        report += f"  Poetic Quality:        {scores['poetic_quality']:.3f}\n"
        report += f"  Translation Natural:   {scores['translation_naturalness']:.3f}\n\n"
        
        report += "🎵 METADATA & CONTENT:\n"
        report += f"  Title Specificity:     {scores['title_specificity']:.3f}\n"
        report += f"  Artist Specificity:    {scores['artist_specificity']:.3f}\n"
        report += f"  Confidence Calibration:{scores['confidence_calibration']:.3f}\n"
        report += f"  Line Count:            {scores['line_count']:.3f}\n"
        report += f"  Content Richness:      {scores['content_richness']:.3f}\n"
        report += f"  Vocabulary Diversity:  {scores['vocabulary_diversity']:.3f}\n\n"
        
        # Quality assessment with finer granularity
        overall = scores['overall']
        if overall >= 0.9:
            assessment = "EXCEPTIONAL"
        elif overall >= 0.8:
            assessment = "EXCELLENT"
        elif overall >= 0.7:
            assessment = "VERY GOOD"
        elif overall >= 0.6:
            assessment = "GOOD"
        elif overall >= 0.5:
            assessment = "FAIR"
        elif overall >= 0.4:
            assessment = "BELOW AVERAGE"
        else:
            assessment = "POOR"
            
        report += f"🏆 Assessment: {assessment}\n"
        
        return report


def validate_json_file(file_path: str) -> Dict:
    """Validate a JSON file and return quality scores"""
    validator = LyricsQualityValidator()
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            result = json.load(f)
        
        scores = validator.calculate_overall_score(result)
        report = validator.generate_report(file_path, scores)
        
        return {
            'file_path': file_path,
            'scores': scores,
            'report': report,
            'valid': True
        }
        
    except Exception as e:
        return {
            'file_path': file_path,
            'error': str(e),
            'valid': False
        }


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python quality_validator.py <json_file>")
        sys.exit(1)
        
    result = validate_json_file(sys.argv[1])
    if result['valid']:
        print(result['report'])
    else:
        print(f"Error validating {result['file_path']}: {result['error']}")