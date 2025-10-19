"""Orchestrator agent for coordinating the multi-agent workflow."""

import asyncio
import uuid
from datetime import datetime
from typing import Any, Dict, Optional

import structlog

from config import settings
from database import MongoDBStore, QdrantVectorStore
from models import (
    ContentPlan,
    GeneratedScript,
    SearchResult,
    UserInput,
    ValidatedContent,
    WorkflowMetadata,
    WorkflowStage,
    WorkflowState,
)
from agents.audience import AudienceModelingAgent
from agents.script_writer import ScriptWritingAgent
from agents.search import SearchAgent
from agents.validation import SupportingDocumentationAgent

logger = structlog.get_logger(__name__)


class OrchestratorAgent:
    """
    Orchestrates the entire content generation workflow.

    This agent manages the state machine that coordinates all other agents,
    handles transitions between workflow stages, and manages timing and data flow.
    """

    def __init__(self):
        """Initialize the orchestrator agent."""
        self.session_id = str(uuid.uuid4())
        self.logger = logger.bind(session_id=self.session_id)

        # Initialize sub-agents
        self.search_agent = SearchAgent()
        self.validation_agent = SupportingDocumentationAgent()
        self.audience_agent = AudienceModelingAgent()
        self.script_agent = ScriptWritingAgent()

        # Initialize database clients
        self.mongo_store = MongoDBStore()
        self.vector_store = QdrantVectorStore()

        # Workflow state
        self.state: Optional[WorkflowState] = None

    async def execute_workflow(self, user_input: UserInput) -> WorkflowState:
        """
        Execute the complete content generation workflow.

        Args:
            user_input: User input containing subject, scope, audience, and length

        Returns:
            Complete workflow state with final results
        """
        self.logger.info(
            "Starting workflow execution",
            subject=user_input.subject,
            audience=user_input.target_audience,
            length=user_input.target_length_minutes
        )

        # Initialize workflow state
        self.state = WorkflowState(
            session_id=self.session_id,
            subject=user_input.subject,
            scope=user_input.scope,
            target_audience=user_input.target_audience,
            target_length=user_input.target_length_minutes,
            current_stage=WorkflowStage.PLANNING,
            search_results=[],
            validated_content=[],
            content_plan=None,
            final_script=None,
            metadata=WorkflowMetadata(
                session_id=self.session_id,
                workflow_stage=WorkflowStage.PLANNING
            ),
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )

        try:
            # Execute workflow stages directly (simplified approach)
            await self._execute_workflow_stages()

            self.logger.info("Workflow execution completed successfully")

            return self.state

        except Exception as e:
            self.logger.error("Workflow execution failed", error=str(e))

            # Update state with error
            if self.state:
                self.state["current_stage"] = WorkflowStage.ERROR
                self.state["metadata"].workflow_stage = WorkflowStage.ERROR
                self.state["metadata"].error_message = str(e)
                self.state["metadata"].completed_at = datetime.utcnow()

            raise

    async def _execute_workflow_stages(self):
        """Execute workflow stages directly without LangGraph."""
        # Save initial workflow state to MongoDB
        self.mongo_store.save_workflow_run(
            session_id=self.session_id,
            workflow_state={"status": "starting"},
            status="in_progress"
        )

        # Stage 1: Research
        await self._research_stage_direct()

        # Stage 2: Validation
        await self._validation_stage_direct()

        # Stage 3: Structuring
        await self._structuring_stage_direct()

        # Stage 4: Script Generation
        await self._script_generation_stage_direct()

        # Save final workflow state
        self.mongo_store.update_workflow_status(
            session_id=self.session_id,
            status="completed"
        )

    async def _research_stage_direct(self):
        """Execute research stage directly."""
        self.state["current_stage"] = WorkflowStage.RESEARCH
        self.state["metadata"].workflow_stage = WorkflowStage.RESEARCH

        try:
            search_results = await self.search_agent.execute_search(
                subject=self.state["subject"],
                scope=self.state["scope"],
                max_results=settings.max_search_results
            )

            self.state["search_results"] = search_results
            self.state["metadata"].search_results_count = len(search_results)

            # Store search results in Qdrant vector store
            await self.vector_store.store_search_results(search_results)

            # Save search results to MongoDB
            search_results_data = [
                {
                    "url": r.url,
                    "title": r.title,
                    "content": r.content,
                    "summary": r.summary,
                    "credibility_score": r.credibility_score,
                    "embedding": r.embedding,
                    "extracted_at": r.extracted_at.isoformat() if isinstance(r.extracted_at, datetime) else r.extracted_at,
                    "metadata": r.metadata
                }
                for r in search_results
            ]
            self.mongo_store.save_search_results(
                session_id=self.session_id,
                search_results=search_results_data
            )

        except Exception as e:
            raise Exception(f"Research stage failed: {str(e)}")

    async def _validation_stage_direct(self):
        """Execute validation stage directly."""
        self.state["current_stage"] = WorkflowStage.VALIDATION
        self.state["metadata"].workflow_stage = WorkflowStage.VALIDATION

        try:
            validated_content = await self.validation_agent.validate_content(
                search_results=self.state["search_results"],
                subject=self.state["subject"]
            )

            self.state["validated_content"] = validated_content
            self.state["metadata"].validated_content_count = len(validated_content)

            # Save validated content to MongoDB
            validated_content_data = [
                {
                    "original_claim": vc.original_claim,
                    "validation_status": vc.validation_status,
                    "supporting_sources": vc.supporting_sources,
                    "confidence_score": vc.confidence_score,
                    "notes": vc.notes
                }
                for vc in validated_content
            ]
            self.mongo_store.save_validated_content(
                session_id=self.session_id,
                validated_content=validated_content_data
            )

        except Exception as e:
            raise Exception(f"Validation stage failed: {str(e)}")

    async def _structuring_stage_direct(self):
        """Execute structuring stage directly."""
        self.state["current_stage"] = WorkflowStage.STRUCTURING
        self.state["metadata"].workflow_stage = WorkflowStage.STRUCTURING

        try:
            content_plan = await self.audience_agent.create_content_plan(
                subject=self.state["subject"],
                scope=self.state["scope"],
                target_audience=self.state["target_audience"],
                target_length=self.state["target_length"],
                search_results=self.state["search_results"],
                validated_content=self.state["validated_content"]
            )

            self.state["content_plan"] = content_plan

            # Save content plan to MongoDB
            content_plan_data = {
                "subject": content_plan.subject,
                "scope": content_plan.scope,
                "target_audience": content_plan.target_audience,
                "target_length_minutes": content_plan.target_length_minutes,
                "overall_tone": content_plan.overall_tone.value,
                "sections": [
                    {
                        "title": s.title,
                        "description": s.description,
                        "estimated_duration_seconds": s.estimated_duration_seconds,
                        "key_points": s.key_points,
                        "transition_notes": s.transition_notes
                    }
                    for s in content_plan.sections
                ],
                "key_messages": content_plan.key_messages,
                "estimated_total_duration": content_plan.estimated_total_duration,
                "created_at": content_plan.created_at.isoformat() if isinstance(content_plan.created_at, datetime) else content_plan.created_at
            }
            self.mongo_store.save_content_plan(
                session_id=self.session_id,
                content_plan=content_plan_data
            )

        except Exception as e:
            raise Exception(f"Structuring stage failed: {str(e)}")

    async def _script_generation_stage_direct(self):
        """Execute script generation stage directly."""
        self.state["current_stage"] = WorkflowStage.SCRIPT_GENERATION
        self.state["metadata"].workflow_stage = WorkflowStage.SCRIPT_GENERATION

        try:
            script = await self.script_agent.generate_script(
                subject=self.state["subject"],
                content_plan=self.state["content_plan"],
                search_results=self.state["search_results"],
                validated_content=self.state["validated_content"]
            )

            self.state["final_script"] = script
            self.state["metadata"].script_generated = True

            # Save generated script to MongoDB
            script_data = {
                "title": script.title,
                "subject": script.subject,
                "target_audience": script.target_audience,
                "target_length_minutes": script.target_length_minutes,
                "full_content": script.full_content,
                "total_word_count": script.total_word_count,
                "estimated_read_time_seconds": script.estimated_read_time_seconds,
                "sections": [
                    {
                        "section_title": s.section_title,
                        "content": s.content,
                        "word_count": s.word_count,
                        "estimated_duration_seconds": s.estimated_duration_seconds
                    }
                    for s in script.sections
                ],
                "citations": script.citations,
                "created_at": script.created_at.isoformat() if isinstance(script.created_at, datetime) else script.created_at
            }
            self.mongo_store.save_script(
                session_id=self.session_id,
                script_data=script_data
            )

            # Mark as completed
            self.state["current_stage"] = WorkflowStage.COMPLETED
            self.state["metadata"].workflow_stage = WorkflowStage.COMPLETED
            self.state["metadata"].completed_at = datetime.utcnow()

        except Exception as e:
            raise Exception(f"Script generation stage failed: {str(e)}")

    def get_workflow_status(self) -> Dict[str, Any]:
        """Get current workflow status and metadata."""
        if not self.state:
            return {"status": "not_started"}

        return {
            "session_id": self.state["session_id"],
            "current_stage": self.state["current_stage"],
            "progress": self._calculate_progress(),
            "metadata": self.state["metadata"].dict(),
            "has_results": len(self.state["search_results"]) > 0,
            "has_script": self.state["final_script"] is not None,
        }

    def _calculate_progress(self) -> float:
        """Calculate workflow completion progress as a percentage."""
        stage_order = [
            WorkflowStage.PLANNING,
            WorkflowStage.RESEARCH,
            WorkflowStage.VALIDATION,
            WorkflowStage.STRUCTURING,
            WorkflowStage.SCRIPT_GENERATION,
            WorkflowStage.COMPLETED,
            WorkflowStage.ERROR,
        ]

        current_index = stage_order.index(self.state["current_stage"])
        return (current_index / (len(stage_order) - 1)) * 100 if self.state else 0
