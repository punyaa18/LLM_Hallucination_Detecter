"""Module for retrieving evidence from trusted sources."""

import wikipedia
import requests
from typing import List, Optional
from sentence_transformers import SentenceTransformer

from .data_models import Evidence, Claim
from config.config import EvidenceRetrievalConfig


class EvidenceRetriever:
    """Retrieves evidence from trusted sources for claims."""
    
    def __init__(
        self,
        config: EvidenceRetrievalConfig = None,
        embedding_model: str = "all-MiniLM-L6-v2"
    ):
        """Initialize evidence retriever.
        
        Args:
            config: Configuration for evidence retrieval.
            embedding_model: Name of embedding model for relevance scoring.
        """
        self.config = config or EvidenceRetrievalConfig()
        self.embedding_model = SentenceTransformer(embedding_model)
        self.wikipedia = WikipediaRetriever(config)
    
    def retrieve_evidence(self, claim: Claim, num_results: int = 3) -> List[Evidence]:
        """Retrieve evidence for a claim.
        
        Args:
            claim: Claim to retrieve evidence for.
            num_results: Number of evidence pieces to retrieve.
            
        Returns:
            List of evidence items.
        """
        evidence_list = []
        
        # Try Wikipedia first
        wiki_evidence = self.wikipedia.retrieve(claim.text, num_results)
        evidence_list.extend(wiki_evidence)
        
        # Score evidence by relevance
        if evidence_list:
            evidence_list = self._score_evidence_relevance(claim.text, evidence_list)
            evidence_list.sort(key=lambda e: e.relevance_score, reverse=True)
        
        return evidence_list[:num_results]
    
    def retrieve_evidence_batch(
        self,
        claims: List[Claim],
        num_results: int = 2
    ) -> dict:
        """Retrieve evidence for multiple claims.
        
        Args:
            claims: List of claims to retrieve evidence for.
            num_results: Number of evidence pieces per claim.
            
        Returns:
            Dictionary mapping claim texts to evidence lists.
        """
        evidence_dict = {}
        
        for claim in claims:
            evidence_dict[claim.text] = self.retrieve_evidence(claim, num_results)
        
        return evidence_dict
    
    def _score_evidence_relevance(
        self,
        claim_text: str,
        evidence_list: List[Evidence]
    ) -> List[Evidence]:
        """Score evidence by relevance to claim.
        
        Args:
            claim_text: The claim text.
            evidence_list: List of evidence to score.
            
        Returns:
            Evidence list with relevance scores.
        """
        try:
            # Get embeddings
            claim_embedding = self.embedding_model.encode(claim_text)
            
            for evidence in evidence_list:
                evidence_embedding = self.embedding_model.encode(evidence.text)
                
                # Calculate cosine similarity
                from scipy.spatial.distance import cosine
                similarity = 1 - cosine(claim_embedding, evidence_embedding)
                evidence.relevance_score = float(similarity)
        except Exception as e:
            print(f"Error scoring evidence relevance: {e}")
            # Fallback: assign equal scores
            for evidence in evidence_list:
                evidence.relevance_score = 0.5
        
        return evidence_list


class WikipediaRetriever:
    """Retrieves evidence from Wikipedia."""
    
    def __init__(self, config: EvidenceRetrievalConfig = None):
        """Initialize Wikipedia retriever.
        
        Args:
            config: Configuration for evidence retrieval.
        """
        self.config = config or EvidenceRetrievalConfig()
        wikipedia.set_lang(self.config.wikipedia_lang)
    
    def retrieve(self, query: str, num_results: int = 3) -> List[Evidence]:
        """Retrieve evidence from Wikipedia.
        
        Args:
            query: Query string to search.
            num_results: Number of results to retrieve.
            
        Returns:
            List of evidence from Wikipedia.
        """
        evidence_list = []
        
        try:
            # Search Wikipedia
            search_results = wikipedia.search(query, results=num_results)
            
            if not search_results:
                return evidence_list
            
            for result in search_results[:num_results]:
                try:
                    # Get the page
                    page = wikipedia.page(result, auto_suggest=False)
                    
                    # Extract relevant sections
                    content = self._extract_relevant_content(page, query)
                    
                    if content:
                        evidence = Evidence(
                            source="Wikipedia",
                            text=content,
                            url=page.url,
                            relevance_score=0.0
                        )
                        evidence_list.append(evidence)
                
                except (wikipedia.exceptions.DisambiguationError,
                        wikipedia.exceptions.PageError):
                    # Skip problematic pages
                    continue
        
        except Exception as e:
            print(f"Error retrieving Wikipedia evidence: {e}")
        
        return evidence_list
    
    def _extract_relevant_content(self, page, query: str) -> Optional[str]:
        """Extract relevant content from Wikipedia page.
        
        Args:
            page: Wikipedia page object.
            query: Original query string.
            
        Returns:
            Relevant content excerpt or None.
        """
        content = page.content
        
        # If content is too long, try to extract relevant sections
        if len(content) > self.config.max_evidence_length:
            # Try to find sections mentioning the query
            sections = content.split('\n')
            relevant_sections = []
            
            for i, section in enumerate(sections):
                if any(
                    word in section.lower()
                    for word in query.lower().split()
                    if len(word) > 3
                ):
                    # Include this section and nearby ones
                    start = max(0, i - 1)
                    end = min(len(sections), i + 2)
                    relevant_sections.extend(sections[start:end])
            
            if relevant_sections:
                content = ' '.join(relevant_sections)
        
        # Ensure content meets minimum length requirements
        if len(content) < self.config.min_evidence_length:
            return None
        
        # Truncate to max length
        if len(content) > self.config.max_evidence_length:
            content = content[:self.config.max_evidence_length] + "..."
        
        return content
