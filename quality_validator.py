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
        """Calculate comprehensive quality score"""
        scores = {}
        
        # Individual component scores
        scores['structure'] = self.validate_json_structure(result)
        scores['completeness'] = self.content_completeness_score(result)
        scores['line_alignment'] = self.line_alignment_score(result.get('lines', []))
        
        # Text cleanliness scores
        full_text = result.get('full_text', {})
        scores['gurmukhi_clean'] = self.clean_text_score(full_text.get('gurmukhi', ''))
        scores['romanized_clean'] = self.clean_text_score(full_text.get('romanized', ''))
        scores['english_clean'] = self.clean_text_score(full_text.get('english', ''))
        
        # Overall weighted score
        weights = {
            'structure': 0.15,
            'completeness': 0.25,
            'line_alignment': 0.25,
            'gurmukhi_clean': 0.15,
            'romanized_clean': 0.10,
            'english_clean': 0.10
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
        report = f"\n=== QUALITY REPORT: {Path(file_path).stem} ===\n"
        report += f"Overall Score: {scores['overall']:.2f}/1.00\n\n"
        
        report += "Component Scores:\n"
        report += f"  Structure:      {scores['structure']:.2f}\n"
        report += f"  Completeness:   {scores['completeness']:.2f}\n"
        report += f"  Line Alignment: {scores['line_alignment']:.2f}\n"
        report += f"  Gurmukhi Clean: {scores['gurmukhi_clean']:.2f}\n"
        report += f"  Romanized Clean:{scores['romanized_clean']:.2f}\n"
        report += f"  English Clean:  {scores['english_clean']:.2f}\n"
        
        # Quality assessment
        overall = scores['overall']
        if overall >= 0.8:
            assessment = "EXCELLENT"
        elif overall >= 0.6:
            assessment = "GOOD"
        elif overall >= 0.4:
            assessment = "FAIR"
        else:
            assessment = "POOR"
            
        report += f"\nAssessment: {assessment}\n"
        
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