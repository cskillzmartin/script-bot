"""Supporting documentation agent for fact-checking and content validation."""

import re
from typing import Dict, List, Optional
from urllib.parse import urlparse

import structlog
from sentence_transformers import SentenceTransformer

from config import settings
from models import SearchResult, ValidatedContent

logger = structlog.get_logger(__name__)


class SupportingDocumentationAgent:
    """
    Agent responsible for validating claims and enriching content with authoritative sources.

    This agent performs:
    1. Fact-checking claims against reliable sources
    2. Identifying authoritative corroboration
    3. Enriching content with verified material
    4. Assessing confidence in assertions
    """

    def __init__(self):
        """Initialize the validation agent."""
        self.logger = logger
        self.embedding_model = SentenceTransformer(settings.embedding_model)

        # Define authoritative domains for validation
        self.authoritative_domains = {
            "gov", "edu", "org", "ac", "int",
            "wikipedia.org", "bbc.com", "reuters.com", "apnews.com",
            "nytimes.com", "wsj.com", "ft.com", "economist.com",
            "nature.com", "sciencemag.org", "pnas.org",
            "harvard.edu", "mit.edu", "stanford.edu", "ox.ac.uk",
            "who.int", "cdc.gov", "nih.gov", "fda.gov",
            "whitehouse.gov", "congress.gov"
        }

    async def validate_content(
        self,
        search_results: List[SearchResult],
        subject: str,
        validation_threshold: float = 0.7
    ) -> List[ValidatedContent]:
        """
        Validate content claims and enrich with authoritative sources.

        Args:
            search_results: List of search results to validate
            subject: Main subject for context
            validation_threshold: Minimum similarity for claim matching

        Returns:
            List of validated content with confidence scores
        """
        self.logger.info(
            "Starting content validation",
            results_count=len(search_results),
            subject=subject
        )

        # Extract claims from search results
        claims = self._extract_claims_from_results(search_results)

        # Validate each claim
        validated_content = []
        for claim in claims:
            validation = await self._validate_single_claim(
                claim, search_results, subject, validation_threshold
            )
            if validation:
                validated_content.append(validation)

        self.logger.info(
            "Validation completed",
            validated_claims=len(validated_content),
            total_claims=len(claims)
        )

        return validated_content

    def _extract_claims_from_results(self, search_results: List[SearchResult]) -> List[str]:
        """Extract factual claims from search results."""
        claims = []

        for result in search_results:
            content = result.content

            # Look for sentences that contain factual assertions
            sentences = re.split(r'[.!?]+', content)

            for sentence in sentences:
                sentence = sentence.strip()
                if self._is_likely_claim(sentence):
                    claims.append(sentence)

        # Remove duplicates while preserving order
        seen = set()
        unique_claims = []
        for claim in claims:
            if claim not in seen:
                seen.add(claim)
                unique_claims.append(claim)

        return unique_claims[:50]  # Limit to prevent overload

    def _is_likely_claim(self, sentence: str) -> bool:
        """Determine if a sentence is likely a factual claim."""
        if len(sentence) < 20 or len(sentence) > 200:
            return False

        # Look for claim indicators
        claim_indicators = [
            "according to", "research shows", "studies indicate",
            "experts say", "data shows", "evidence suggests",
            "it is", "they are", "this means", "researchers found",
            "statistics show", "surveys indicate"
        ]

        sentence_lower = sentence.lower()
        return any(indicator in sentence_lower for indicator in claim_indicators)

    async def _validate_single_claim(
        self,
        claim: str,
        search_results: List[SearchResult],
        subject: str,
        threshold: float
    ) -> Optional[ValidatedContent]:
        """Validate a single claim against authoritative sources."""
        # Find supporting evidence in search results
        supporting_sources = []
        confidence_scores = []

        for result in search_results:
            similarity = self._calculate_claim_similarity(claim, result)

            if similarity >= threshold:
                # Check if this is an authoritative source
                if self._is_authoritative_source(result.url):
                    supporting_sources.append(result.url)
                    confidence_scores.append(similarity)

        if not supporting_sources:
            return None

        # Calculate overall confidence
        avg_confidence = sum(confidence_scores) / len(confidence_scores)
        max_confidence = max(confidence_scores)

        # Determine validation status
        if max_confidence > 0.85 and len(supporting_sources) >= 2:
            status = "strongly_validated"
        elif max_confidence > 0.75 and len(supporting_sources) >= 1:
            status = "validated"
        elif max_confidence > 0.6:
            status = "partially_validated"
        else:
            status = "insufficient_evidence"

        return ValidatedContent(
            original_claim=claim,
            validation_status=status,
            supporting_sources=supporting_sources,
            confidence_score=avg_confidence,
            notes=self._generate_validation_notes(
                status, len(supporting_sources), avg_confidence
            )
        )

    def _calculate_claim_similarity(self, claim: str, search_result: SearchResult) -> float:
        """Calculate semantic similarity between claim and search result."""
        try:
            # Generate embeddings for comparison
            claim_embedding = self.embedding_model.encode(claim)
            content_embedding = search_result.embedding

            # Calculate cosine similarity
            import numpy as np
            dot_product = np.dot(claim_embedding, content_embedding)
            norm_claim = np.linalg.norm(claim_embedding)
            norm_content = np.linalg.norm(content_embedding)

            if norm_claim == 0 or norm_content == 0:
                return 0.0

            similarity = dot_product / (norm_claim * norm_content)
            return float(similarity)

        except Exception as e:
            self.logger.warning("Similarity calculation failed", error=str(e))
            return 0.0

    def _is_authoritative_source(self, url: str) -> bool:
        """Check if a URL represents an authoritative source."""
        domain = urlparse(url).netloc.lower()

        # Check against authoritative domains
        return any(auth_domain in domain for auth_domain in self.authoritative_domains)

    def _generate_validation_notes(
        self,
        status: str,
        source_count: int,
        confidence: float
    ) -> str:
        """Generate human-readable validation notes."""
        if status == "strongly_validated":
            return f"Strong validation from {source_count} authoritative sources with {confidence:.2f} average confidence."
        elif status == "validated":
            return f"Validated by {source_count} authoritative source(s) with {confidence:.2f} confidence."
        elif status == "partially_validated":
            return f"Partial validation with moderate confidence ({confidence:.2f}) from available sources."
        else:
            return f"Insufficient authoritative evidence found (confidence: {confidence:.2f})."

    async def enrich_content_with_validation(
        self,
        search_results: List[SearchResult],
        validated_content: List[ValidatedContent]
    ) -> List[SearchResult]:
        """Enrich search results with validation metadata."""
        # Create a mapping of claims to validation results
        claim_validation_map = {
            vc.original_claim: vc for vc in validated_content
        }

        # Add validation metadata to search results
        enriched_results = []
        for result in search_results:
            # Find claims in this result that have been validated
            result_claims = []
            for claim, validation in claim_validation_map.items():
                if claim in result.content:
                    result_claims.append(validation)

            # Add validation metadata to the result
            enriched_result = result.copy()
            enriched_result.metadata["validated_claims"] = [
                {
                    "claim": vc.original_claim,
                    "status": vc.validation_status,
                    "confidence": vc.confidence_score,
                    "sources": vc.supporting_sources
                }
                for vc in result_claims
            ]

            enriched_results.append(enriched_result)

        return enriched_results
