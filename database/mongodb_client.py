"""MongoDB client for metadata, logs, and structured data storage."""

import json
from datetime import datetime
from typing import Any, Dict, List, Optional

import structlog
from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.database import Database

from config import settings, get_database_name

logger = structlog.get_logger(__name__)


class MongoDBStore:
    """
    MongoDB client for persistent storage of workflow metadata and artifacts.

    Handles:
    1. Workflow state and metadata storage
    2. Search results and validation logs
    3. Script generation history
    4. Audit trails and analytics
    """

    def __init__(self):
        """Initialize MongoDB client."""
        self.client = MongoClient(settings.mongodb_url)
        self.database = self.client[get_database_name()]
        self.logger = logger

        # Test connection
        self._test_connection()

        # Ensure indexes for performance
        self._ensure_indexes()

    def _test_connection(self):
        """Test MongoDB connection."""
        try:
            # Ping the server
            self.client.admin.command('ping')
            self.logger.info("MongoDB connection established")
        except Exception as e:
            self.logger.error("MongoDB connection failed", error=str(e))
            raise

    def _ensure_indexes(self):
        """Ensure necessary indexes exist for query performance."""
        collections_indexes = {
            "workflow_runs": [
                ("session_id", 1),
                ("created_at", -1),
                ("status", 1)
            ],
            "search_results": [
                ("session_id", 1),
                ("url", 1),
                ("extracted_at", -1)
            ],
            "content_plans": [
                ("session_id", 1),
                ("timestamp", -1)
            ],
            "validated_content": [
                ("session_id", 1),
                ("validation_status", 1),
                ("confidence_score", -1)
            ],
            "scripts": [
                ("session_id", 1),
                ("created_at", -1),
                ("subject", 1)
            ]
        }

        for collection_name, indexes in collections_indexes.items():
            collection = self.database[collection_name]

            for index in indexes:
                field, direction = index
                index_name = f"{field}_{direction}"

                # Check if index exists
                existing_indexes = collection.list_indexes()
                index_exists = any(idx.get("name") == index_name for idx in existing_indexes)

                if not index_exists:
                    collection.create_index([(field, direction)], name=index_name)
                    self.logger.debug("Created index", collection=collection_name, index=index_name)

    # Workflow Management
    def save_workflow_run(
        self,
        session_id: str,
        workflow_state: Dict[str, Any],
        status: str = "in_progress"
    ) -> bool:
        """Save workflow run metadata."""
        try:
            collection = self.database["workflow_runs"]

            document = {
                "session_id": session_id,
                "status": status,
                "workflow_state": workflow_state,
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow()
            }

            # Upsert - insert or update if exists
            result = collection.replace_one(
                {"session_id": session_id},
                document,
                upsert=True
            )

            self.logger.debug(
                "Workflow run saved",
                session_id=session_id,
                status=status,
                upserted=result.upserted_id is not None
            )

            return True

        except Exception as e:
            self.logger.error("Failed to save workflow run", error=str(e))
            return False

    def get_workflow_run(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve workflow run by session ID."""
        try:
            collection = self.database["workflow_runs"]
            document = collection.find_one({"session_id": session_id})

            if document:
                # Convert ObjectId to string for JSON serialization
                document["_id"] = str(document["_id"])
                return document

            return None

        except Exception as e:
            self.logger.error("Failed to retrieve workflow run", session_id=session_id, error=str(e))
            return None

    def update_workflow_status(
        self,
        session_id: str,
        status: str,
        error_message: Optional[str] = None
    ) -> bool:
        """Update workflow run status."""
        try:
            collection = self.database["workflow_runs"]

            update_data = {
                "status": status,
                "updated_at": datetime.utcnow()
            }

            if error_message:
                update_data["error_message"] = error_message

            result = collection.update_one(
                {"session_id": session_id},
                {"$set": update_data}
            )

            self.logger.debug(
                "Workflow status updated",
                session_id=session_id,
                status=status
            )

            return result.modified_count > 0

        except Exception as e:
            self.logger.error("Failed to update workflow status", error=str(e))
            return False

    # Search Results Storage
    def save_search_results(
        self,
        session_id: str,
        search_results: List[Dict[str, Any]]
    ) -> bool:
        """Save search results for a session."""
        try:
            collection = self.database["search_results"]

            documents = []
            for result in search_results:
                document = {
                    "session_id": session_id,
                    "url": result.get("url"),
                    "title": result.get("title"),
                    "content": result.get("content"),
                    "summary": result.get("summary"),
                    "credibility_score": result.get("credibility_score", 0.0),
                    "embedding": result.get("embedding"),
                    "extracted_at": result.get("extracted_at", datetime.utcnow()),
                    "metadata": result.get("metadata", {})
                }
                documents.append(document)

            if documents:
                collection.insert_many(documents)

            self.logger.debug(
                "Search results saved",
                session_id=session_id,
                count=len(documents)
            )

            return True

        except Exception as e:
            self.logger.error("Failed to save search results", error=str(e))
            return False

    # Content Plan Storage
    def save_content_plan(
        self,
        session_id: str,
        content_plan: Dict[str, Any]
    ) -> bool:
        """
        Save content plan to MongoDB.

        Args:
            session_id: Session identifier
            content_plan: Content plan dictionary

        Returns:
            bool: Success status
        """
        try:
            content_plan_doc = {
                "session_id": session_id,
                "content_plan": content_plan,
                "timestamp": datetime.utcnow()
            }

            collection = self.database["content_plans"]
            result = collection.insert_one(content_plan_doc)

            self.logger.info(
                "Content plan saved successfully",
                session_id=session_id,
                content_plan_id=str(result.inserted_id)
            )
            return True

        except Exception as e:
            self.logger.error("Failed to save content plan", error=str(e))
            return False

    # Validated Content Storage
    def save_validated_content(
        self,
        session_id: str,
        validated_content: List[Dict[str, Any]]
    ) -> bool:
        """Save validated content for a session."""
        try:
            collection = self.database["validated_content"]

            documents = []
            for content in validated_content:
                document = {
                    "session_id": session_id,
                    "original_claim": content.get("original_claim"),
                    "validation_status": content.get("validation_status"),
                    "supporting_sources": content.get("supporting_sources", []),
                    "confidence_score": content.get("confidence_score", 0.0),
                    "notes": content.get("notes"),
                    "validated_at": datetime.utcnow()
                }
                documents.append(document)

            if documents:
                collection.insert_many(documents)

            self.logger.debug(
                "Validated content saved",
                session_id=session_id,
                count=len(documents)
            )

            return True

        except Exception as e:
            self.logger.error("Failed to save validated content", error=str(e))
            return False

    # Script Storage
    def save_script(
        self,
        session_id: str,
        script_data: Dict[str, Any]
    ) -> bool:
        """Save generated script."""
        try:
            collection = self.database["scripts"]

            document = {
                "session_id": session_id,
                "title": script_data.get("title"),
                "subject": script_data.get("subject"),
                "target_audience": script_data.get("target_audience"),
                "target_length_minutes": script_data.get("target_length_minutes"),
                "full_content": script_data.get("full_content"),
                "total_word_count": script_data.get("total_word_count"),
                "estimated_read_time_seconds": script_data.get("estimated_read_time_seconds"),
                "sections": script_data.get("sections", []),
                "created_at": script_data.get("created_at", datetime.utcnow()),
                "file_path": script_data.get("file_path")
            }

            collection.insert_one(document)

            self.logger.debug(
                "Script saved",
                session_id=session_id,
                title=script_data.get("title")
            )

            return True

        except Exception as e:
            self.logger.error("Failed to save script", error=str(e))
            return False

    # Analytics and Query Methods
    async def get_workflow_analytics(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get workflow analytics for monitoring."""
        try:
            collection = self.database["workflow_runs"]

            pipeline = [
                {
                    "$group": {
                        "_id": "$status",
                        "count": {"$sum": 1},
                        "avg_duration": {
                            "$avg": {
                                "$subtract": ["$updated_at", "$created_at"]
                            }
                        }
                    }
                },
                {"$sort": {"count": -1}},
                {"$limit": limit}
            ]

            results = list(collection.aggregate(pipeline))

            # Convert timedelta to seconds for JSON serialization
            for result in results:
                if "avg_duration" in result and result["avg_duration"]:
                    result["avg_duration_seconds"] = result["avg_duration"].total_seconds()

            return results

        except Exception as e:
            self.logger.error("Failed to get workflow analytics", error=str(e))
            return []

    async def get_recent_scripts(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recently generated scripts."""
        try:
            collection = self.database["scripts"]

            documents = collection.find(
                {},
                {"_id": 1, "title": 1, "subject": 1, "created_at": 1, "total_word_count": 1}
            ).sort("created_at", -1).limit(limit)

            scripts = []
            for doc in documents:
                doc["_id"] = str(doc["_id"])
                scripts.append(doc)

            return scripts

        except Exception as e:
            self.logger.error("Failed to get recent scripts", error=str(e))
            return []

    async def cleanup_old_runs(self, days_old: int = 30) -> int:
        """Clean up workflow runs older than specified days."""
        try:
            cutoff_date = datetime.utcnow() - datetime.timedelta(days=days_old)

            # Delete old workflow runs
            workflow_result = self.database["workflow_runs"].delete_many(
                {"created_at": {"$lt": cutoff_date}}
            )

            # Delete associated search results
            search_result = self.database["search_results"].delete_many(
                {"extracted_at": {"$lt": cutoff_date}}
            )

            # Delete old scripts (keep for longer)
            script_result = self.database["scripts"].delete_many(
                {"created_at": {"$lt": cutoff_date}}
            )

            total_deleted = (
                workflow_result.deleted_count +
                search_result.deleted_count +
                script_result.deleted_count
            )

            self.logger.info(
                "Cleanup completed",
                days_old=days_old,
                total_deleted=total_deleted
            )

            return total_deleted

        except Exception as e:
            self.logger.error("Cleanup failed", error=str(e))
            return 0
