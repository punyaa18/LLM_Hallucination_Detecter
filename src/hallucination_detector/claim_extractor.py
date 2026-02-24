"""Module for extracting factual claims from text."""

import re
import nltk
from typing import List
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.tag import pos_tag
from nltk.chunk import ne_chunk

from .data_models import Claim
from config.config import ClaimExtractionConfig

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger')

try:
    nltk.data.find('chunkers/ne_chunk')
except LookupError:
    nltk.download('maxent_ne_chunker')
    nltk.download('words')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')


class ClaimExtractor:
    """Extracts factual claims from text."""
    
    def __init__(self, config: ClaimExtractionConfig = None):
        """Initialize claim extractor.
        
        Args:
            config: Configuration for claim extraction.
        """
        self.config = config or ClaimExtractionConfig()
        self.stop_words = set(stopwords.words('english'))
    
    def extract_claims(self, text: str, max_claims: int = 10) -> List[Claim]:
        """Extract factual claims from text.
        
        Args:
            text: Input text to extract claims from.
            max_claims: Maximum number of claims to extract.
            
        Returns:
            List of extracted claims.
        """
        if not text or not text.strip():
            return []
        
        claims = []
        
        # Method 1: Extract claims from sentences with entities
        entity_claims = self._extract_entity_based_claims(text)
        claims.extend(entity_claims)
        
        # Method 2: Extract claims based on patterns
        pattern_claims = self._extract_pattern_based_claims(text)
        claims.extend(pattern_claims)
        
        # Remove duplicates and filter
        unique_claims = self._deduplicate_claims(claims)
        unique_claims = self._filter_claims(unique_claims)
        
        # Sort by confidence and return top N
        unique_claims.sort(key=lambda c: c.confidence, reverse=True)
        return unique_claims[:max_claims]
    
    def _extract_entity_based_claims(self, text: str) -> List[Claim]:
        """Extract claims that mention named entities.
        
        Args:
            text: Input text.
            
        Returns:
            List of entity-based claims.
        """
        claims = []
        sentences = sent_tokenize(text)
        current_pos = 0
        
        for sentence in sentences:
            # Check if sentence contains named entities or domain words
            tokens = word_tokenize(sentence)
            pos_tags = pos_tag(tokens)
            
            # Check for proper nouns (NNP, NNPS) or numbers (CD)
            has_entity = any(tag in ['NNP', 'NNPS', 'CD'] for _, tag in pos_tags)
            
            # Check for domain words (verbs, numbers, dates, etc.)
            has_domain_words = any(
                tag.startswith('VB') or tag.startswith('JJ') or tag == 'CD'
                for _, tag in pos_tags
            )
            
            if (has_entity or has_domain_words) and len(tokens) >= self.config.min_tokens:
                start_idx = text.find(sentence, current_pos)
                end_idx = start_idx + len(sentence)
                
                claim = Claim(
                    text=sentence.strip(),
                    start_idx=start_idx,
                    end_idx=end_idx,
                    confidence=0.8
                )
                claims.append(claim)
                current_pos = end_idx
        
        return claims
    
    def _extract_pattern_based_claims(self, text: str) -> List[Claim]:
        """Extract claims based on linguistic patterns.
        
        Args:
            text: Input text.
            
        Returns:
            List of pattern-based claims.
        """
        claims = []
        
        # Pattern 1: [Subject] + [Verb] + [Object]
        # Pattern for: "[Noun/Entity] is/was/are [adjective/noun]"
        pattern1 = r'(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:is|was|are|were)\s+(?:[a-z\s]+)'
        
        # Pattern 2: "[Noun] [verb] [number/percentage]"
        pattern2 = r'(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:has|have|contain|produced|created|had)\s+[\d\s,\.]+'
        
        # Pattern 3: Temporal claims
        pattern3 = r'(?:In|On|During|Since)\s+[\d]{4}|(?:In\s+[\d]{4})\s+.*'
        
        patterns = [pattern1, pattern2, pattern3]
        
        for pattern in patterns:
            for match in re.finditer(pattern, text):
                claim_text = match.group(0).strip()
                if len(claim_text.split()) >= self.config.min_tokens:
                    claim = Claim(
                        text=claim_text,
                        start_idx=match.start(),
                        end_idx=match.end(),
                        confidence=0.7
                    )
                    claims.append(claim)
        
        return claims
    
    def _deduplicate_claims(self, claims: List[Claim]) -> List[Claim]:
        """Remove duplicate or very similar claims.
        
        Args:
            claims: List of claims to deduplicate.
            
        Returns:
            List of unique claims.
        """
        unique_claims = []
        seen_texts = set()
        
        for claim in claims:
            # Normalize text
            normalized = claim.text.lower().strip()
            
            # Check if we've seen this or a very similar claim
            is_duplicate = False
            for seen_text in seen_texts:
                # Simple similarity check
                if self._text_similarity(normalized, seen_text) > 0.8:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_claims.append(claim)
                seen_texts.add(normalized)
        
        return unique_claims
    
    def _filter_claims(self, claims: List[Claim]) -> List[Claim]:
        """Filter out low-quality claims.
        
        Args:
            claims: List of claims to filter.
            
        Returns:
            List of high-quality claims.
        """
        filtered = []
        
        for claim in claims:
            tokens = word_tokenize(claim.text)
            
            # Filter by length
            if not (self.config.min_tokens <= len(tokens) <= 50):
                continue
            
            # Filter out all-stop-words claims
            if self.config.filter_stop_words:
                non_stop_words = [
                    t for t in tokens
                    if t.lower() not in self.stop_words and t.isalpha()
                ]
                if len(non_stop_words) < 2:
                    continue
            
            # Filter out questions or incomplete sentences
            if claim.text.strip().endswith('?'):
                continue
            
            filtered.append(claim)
        
        return filtered
    
    def _text_similarity(self, text1: str, text2: str) -> float:
        """Calculate simple text similarity.
        
        Args:
            text1: First text.
            text2: Second text.
            
        Returns:
            Similarity score between 0 and 1.
        """
        words1 = set(text1.split())
        words2 = set(text2.split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        return intersection / union if union > 0 else 0.0
