"""Qdrant vector database client for semantic content storage and retrieval."""

from typing import Dict, List, Optional, Tuple
import uuid

import structlog
from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    Distance,
    PointStruct,
    SearchParams,
    VectorParams
)

from config import settings, get_qdrant_collection_name
from models import SearchResult

logger = structlog.get_logger(__name__)


class QdrantVectorStore:
    """
    Vector store client for Qdrant database operations.

    Handles:
    1. Vector storage and indexing
    2. Semantic similarity search
    3. Collection management
    4. Batch operations for efficiency
    """

    def __init__(self):
        """Initialize Qdrant client."""
        self.client = QdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key,
            timeout=30
        )
        self.collection_name = get_qdrant_collection_name()
        self.logger = logger

        # Ensure collection exists
        self._ensure_collection()

    def _ensure_collection(self):
        """Ensure the content collection exists in Qdrant."""
        try:
            # Check if collection exists
            collections = self.client.get_collections()
            collection_names = [col.name for col in collections.collections]

            if self.collection_name not in collection_names:
                self.logger.info("Creating Qdrant collection", collection=self.collection_name)

                # Create collection with vector configuration
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=384,  # all-MiniLM-L6-v2 dimension
                        distance=Distance.COSINE
                    )
                )

                self.logger.info("Collection created successfully")
            else:
                self.logger.debug("Collection already exists", collection=self.collection_name)

        except Exception as e:
            self.logger.error("Failed to ensure collection", error=str(e))
            raise

    async def store_search_results(self, search_results: List[SearchResult]) -> List[str]:
        """
        Store search results as vectors in Qdrant.

        Args:
            search_results: List of search results to store

        Returns:
            List of point IDs for stored vectors
        """
        if not search_results:
            return []

        self.logger.info("Storing search results in vector store", count=len(search_results))

        points = []
        for result in search_results:
            point_id = str(uuid.uuid4())

            # Create point structure
            point = PointStruct(
                id=point_id,
                vector=result.embedding,
                payload={
                    "url": result.url,
                    "title": result.title,
                    "content": result.content[:settings.max_content_length],
                    "summary": result.summary,
                    "credibility_score": result.credibility_score,
                    "extracted_at": result.extracted_at.isoformat(),
                    "word_count": result.metadata.get("word_count", 0),
                    "search_snippet": result.metadata.get("search_snippet", ""),
                }
            )

            points.append(point)

        try:
            # Batch upload points
            self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )

            point_ids = [point.id for point in points]
            self.logger.info("Search results stored successfully", point_ids=point_ids)

            return point_ids

        except Exception as e:
            self.logger.error("Failed to store search results", error=str(e))
            raise

    async def search_similar_content(
        self,
        query_embedding: List[float],
        limit: int = 5,
        threshold: float = 0.7,
        filters: Optional[Dict] = None
    ) -> List[Dict]:
        """
        Search for similar content using vector similarity.

        Args:
            query_embedding: Query vector for similarity search
            limit: Maximum number of results to return
            threshold: Minimum similarity score
            filters: Optional metadata filters

        Returns:
            List of similar content with scores
        """
        self.logger.debug(
            "Searching similar content",
            limit=limit,
            threshold=threshold
        )

        try:
            # Build search parameters
            search_params = SearchParams(
                hnsw_ef=128,
                exact=False
            )

            # Execute search
            search_results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                query_filter=filters,
                search_params=search_params,
                limit=limit,
                score_threshold=threshold
            )

            # Format results
            formatted_results = []
            for result in search_results:
                formatted_results.append({
                    "id": result.id,
                    "score": result.score,
                    "url": result.payload.get("url"),
                    "title": result.payload.get("title"),
                    "content": result.payload.get("content"),
                    "summary": result.payload.get("summary"),
                    "credibility_score": result.payload.get("credibility_score", 0.0),
                    "metadata": {
                        "word_count": result.payload.get("word_count", 0),
                        "extracted_at": result.payload.get("extracted_at"),
                    }
                })

            self.logger.debug(
                "Similar content search completed",
                results=len(formatted_results)
            )

            return formatted_results

        except Exception as e:
            self.logger.error("Similar content search failed", error=str(e))
            return []

    async def search_by_text(
        self,
        query_text: str,
        embedding_model,
        limit: int = 5,
        threshold: float = 0.7
    ) -> List[Dict]:
        """
        Search for content using text query (converts to embedding first).

        Args:
            query_text: Text query to search for
            embedding_model: Model to generate embeddings
            limit: Maximum results to return
            threshold: Minimum similarity score

        Returns:
            List of similar content results
        """
        # Generate embedding for the query text
        query_embedding = embedding_model.encode(query_text).tolist()

        # Search using the embedding
        return await self.search_similar_content(
            query_embedding=query_embedding,
            limit=limit,
            threshold=threshold
        )

    async def get_content_by_id(self, point_id: str) -> Optional[Dict]:
        """Retrieve specific content by point ID."""
        try:
            result = self.client.retrieve(
                collection_name=self.collection_name,
                ids=[point_id]
            )

            if result:
                point = result[0]
                return {
                    "id": point.id,
                    "url": point.payload.get("url"),
                    "title": point.payload.get("title"),
                    "content": point.payload.get("content"),
                    "summary": point.payload.get("summary"),
                    "credibility_score": point.payload.get("credibility_score", 0.0),
                }

            return None

        except Exception as e:
            self.logger.error("Failed to retrieve content by ID", point_id=point_id, error=str(e))
            return None

    async def delete_collection(self) -> bool:
        """Delete the entire collection (for testing/cleanup)."""
        try:
            self.client.delete_collection(collection_name=self.collection_name)
            self.logger.info("Collection deleted", collection=self.collection_name)
            return True

        except Exception as e:
            self.logger.error("Failed to delete collection", error=str(e))
            return False

    def get_collection_info(self) -> Dict:
        """Get information about the collection."""
        try:
            collection_info = self.client.get_collection(collection_name=self.collection_name)

            return {
                "name": collection_info.name,
                "vectors_count": collection_info.points_count,
                "status": "active" if collection_info.status else "inactive"
            }

        except Exception as e:
            self.logger.error("Failed to get collection info", error=str(e))
            return {"error": str(e)}

    async def update_content_metadata(self, point_id: str, metadata: Dict) -> bool:
        """Update metadata for a specific point."""
        try:
            # Create update payload
            self.client.set_payload(
                collection_name=self.collection_name,
                payload=metadata,
                points=[point_id]
            )

            self.logger.debug("Content metadata updated", point_id=point_id)
            return True

        except Exception as e:
            self.logger.error("Failed to update metadata", point_id=point_id, error=str(e))
            return False
