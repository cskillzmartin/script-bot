"""Search agent for web research and content harvesting."""

import asyncio
import hashlib
from typing import List, Optional
from urllib.parse import urlparse

import structlog
import trafilatura
from ddgs import DDGS
from sentence_transformers import SentenceTransformer

from config import settings
from models import SearchResult

logger = structlog.get_logger(__name__)


class SearchAgent:
    """
    Agent responsible for web search, content extraction, and vector embedding.

    This agent performs comprehensive web research by:
    1. Executing web searches using Tavily API
    2. Following URLs to extract readable text content
    3. Generating summaries and embeddings for semantic search
    4. Storing results for use by other agents
    """

    def __init__(self):
        """Initialize the search agent."""
        self.logger = logger
        self.ddg_client = DDGS()
        self.embedding_model = SentenceTransformer(settings.embedding_model)
        self.semaphore = asyncio.Semaphore(settings.max_concurrent_searches)

    async def execute_search(
        self,
        subject: str,
        scope: str,
        max_results: int = 10,
        include_domains: Optional[List[str]] = None,
        exclude_domains: Optional[List[str]] = None,
    ) -> List[SearchResult]:
        """
        Execute comprehensive web search and content extraction.

        Args:
            subject: Main topic to search for
            scope: Specific scope or angle for the search
            max_results: Maximum number of search results to process
            include_domains: Optional list of domains to prioritize
            exclude_domains: Optional list of domains to exclude

        Returns:
            List of processed search results with embeddings
        """
        self.logger.info(
            "Starting web search",
            subject=subject,
            scope=scope,
            max_results=max_results
        )

        # Construct search query
        search_query = self._build_search_query(subject, scope)

        try:
            # Execute search using DuckDuckGo
            search_results = list(self.ddg_client.text(
                query=search_query,
                max_results=max_results * 2,  # Get more results for filtering
                region="wt-wt",  # World-wide results
                safesearch="moderate",
                timelimit="y"  # Results from past year
            ))

            self.logger.info(
                "Search completed",
                raw_results=len(search_results)
            )

            # Process and filter results
            processed_results = await self._process_search_results(
                search_results,
                max_results,
                include_domains,
                exclude_domains
            )

            self.logger.info(
                "Content processing completed",
                processed_results=len(processed_results)
            )

            return processed_results

        except Exception as e:
            self.logger.error("Search execution failed", error=str(e))
            raise

    def _build_search_query(self, subject: str, scope: str) -> str:
        """Build an optimized search query from subject and scope."""
        # Create a focused query that combines subject and scope
        if scope and scope.lower() not in subject.lower():
            return f"{subject} {scope}"
        return subject

    async def _process_search_results(
        self,
        raw_results: List[dict],
        max_results: int,
        include_domains: Optional[List[str]],
        exclude_domains: Optional[List[str]]
    ) -> List[SearchResult]:
        """Process raw search results into structured SearchResult objects."""
        filtered_results = self._filter_results(
            raw_results, include_domains, exclude_domains
        )

        # Limit to max_results
        limited_results = filtered_results[:max_results]

        # Process results concurrently
        tasks = [
            self._process_single_result(result_data)
            for result_data in limited_results
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter out any exceptions and return successful results
        successful_results = [
            result for result in results
            if isinstance(result, SearchResult)
        ]

        return successful_results

    def _filter_results(
        self,
        raw_results: List[dict],
        include_domains: Optional[List[str]],
        exclude_domains: Optional[List[str]]
    ) -> List[dict]:
        """Filter search results based on domain criteria."""
        filtered = []

        for result in raw_results:
            url = result.get("href", result.get("url", ""))
            domain = urlparse(url).netloc.lower()

            # Skip if in exclude list
            if exclude_domains and any(
                excluded_domain in domain for excluded_domain in exclude_domains
            ):
                continue

            # Prioritize if in include list
            if include_domains and any(
                included_domain in domain for included_domain in include_domains
            ):
                filtered.insert(0, result)  # Add to front
            else:
                filtered.append(result)

        return filtered

    async def _process_single_result(self, result_data: dict) -> SearchResult:
        """Process a single search result into a SearchResult object."""
        url = result_data.get("href", result_data.get("url", ""))
        title = result_data.get("title", result_data.get("body", ""))
        content_snippet = result_data.get("body", result_data.get("content", ""))

        async with self.semaphore:
            try:
                # Extract full content from the URL
                full_content = await self._extract_page_content(url)

                if not full_content or len(full_content.strip()) < 100:
                    self.logger.warning("Insufficient content extracted", url=url)
                    # Use the snippet as fallback
                    full_content = content_snippet

                # Generate summary
                summary = await self._generate_summary(full_content, title)

                # Generate embedding
                embedding = self._generate_embedding(full_content)

                # Calculate credibility score
                credibility_score = self._calculate_credibility_score(url, title)

                return SearchResult(
                    url=url,
                    title=title,
                    content=full_content,
                    summary=summary,
                    embedding=embedding,
                    credibility_score=credibility_score,
                    metadata={
                        "search_snippet": content_snippet,
                        "word_count": len(full_content.split()),
                        "extraction_method": "trafilatura",
                        "search_engine": "duckduckgo"
                    }
                )

            except Exception as e:
                self.logger.error("Failed to process result", url=url, error=str(e))
                raise

    async def _extract_page_content(self, url: str) -> Optional[str]:
        """Extract readable text content from a webpage."""
        try:
            # Use trafilatura for robust content extraction
            extracted = trafilatura.extract(
                url,
                timeout=settings.content_extraction_timeout_seconds,
                include_comments=False,
                include_tables=True,
                include_images=False,
                include_links=False,
                deduplicate=True,
            )

            if extracted:
                # Clean and truncate content
                cleaned_content = self._clean_extracted_content(extracted)
                return cleaned_content[:settings.max_content_length]

            return None

        except Exception as e:
            self.logger.warning("Content extraction failed", url=url, error=str(e))
            return None

    def _clean_extracted_content(self, content: str) -> str:
        """Clean and normalize extracted content."""
        # Remove excessive whitespace
        cleaned = " ".join(content.split())

        # Remove common navigation/footer text patterns
        noise_patterns = [
            r"© \d{4}",
            r"All rights reserved",
            r"Terms of Service",
            r"Privacy Policy",
            r"Contact Us",
            r"Follow us on",
        ]

        for pattern in noise_patterns:
            import re
            cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE)

        return cleaned.strip()

    async def _generate_summary(self, content: str, title: str) -> str:
        """Generate a concise summary of the content."""
        # For now, use a simple extractive approach
        # In a full implementation, this could use the LLM for better summaries
        sentences = content.split(".")[:3]  # First 3 sentences
        summary = ". ".join(sentences).strip()

        if not summary.endswith("."):
            summary += "."

        return summary

    def _generate_embedding(self, content: str) -> List[float]:
        """Generate vector embedding for the content."""
        try:
            # Use sentence transformers for consistent embeddings
            embedding = self.embedding_model.encode(content)
            return embedding.tolist()
        except Exception as e:
            self.logger.error("Embedding generation failed", error=str(e))
            # Return zero vector as fallback
            return [0.0] * 384  # Common embedding dimension

    def _calculate_credibility_score(self, url: str, title: str) -> float:
        """Calculate a basic credibility score for the source."""
        domain = urlparse(url).netloc.lower()
        score = 0.5  # Base score

        # Boost score for authoritative domains
        authoritative_domains = [
            ".gov", ".edu", ".org",
            "wikipedia.org", "bbc.com", "reuters.com",
            "nytimes.com", "wsj.com", "nature.com",
            "sciencemag.org", "harvard.edu", "mit.edu"
        ]

        for authoritative_domain in authoritative_domains:
            if authoritative_domain in domain:
                score += 0.3
                break

        # Penalize score for known low-quality domains
        low_quality_domains = [
            "reddit.com", "quora.com", "yahoo.com",
            "aol.com", "blogspot.com"
        ]

        for low_quality_domain in low_quality_domains:
            if low_quality_domain in domain:
                score -= 0.2
                break

        return min(1.0, max(0.0, score))
