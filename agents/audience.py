"""Audience modeling agent for content structuring and narrative design."""

import math
from typing import Dict, List, Optional

import structlog

from config import settings
from models import (
    ContentPlan,
    ContentSection,
    ContentTone,
    SearchResult,
    ValidatedContent
)

logger = structlog.get_logger(__name__)


class AudienceModelingAgent:
    """
    Agent responsible for designing narrative structure tailored to target audience.

    This agent:
    1. Analyzes target audience characteristics and preferences
    2. Designs content structure and pacing appropriate for the audience
    3. Adapts tone and complexity based on audience expertise
    4. Creates section outlines and talking points
    5. Estimates content timing and flow
    """

    def __init__(self):
        """Initialize the audience modeling agent."""
        self.logger = logger

        # Default audience profiles as examples (used as fallback)
        # These are now used for reference only - actual analysis is dynamic
        self.default_audience_hints = {
            "business executives": {
                "tone": ContentTone.PROFESSIONAL,
                "complexity": "intermediate",
                "focus": ["roi", "strategy", "implementation", "case_studies"],
                "structure": ["executive_summary", "key_insights", "action_items", "conclusion"]
            },
            "entrepreneurs": {
                "tone": ContentTone.CONVERSATIONAL,
                "complexity": "intermediate",
                "focus": ["practical_applications", "getting_started", "challenges", "solutions"],
                "structure": ["problem_statement", "solution_overview", "step_by_step", "next_steps"]
            },
            "developers": {
                "tone": ContentTone.ENGAGING,
                "complexity": "advanced",
                "focus": ["technical_details", "implementation", "code_examples", "best_practices"],
                "structure": ["overview", "technical_breakdown", "examples", "advanced_topics"]
            },
            "general public": {
                "tone": ContentTone.FRIENDLY,
                "complexity": "basic",
                "focus": ["everyday_impact", "simple_explanations", "real_world_examples"],
                "structure": ["what_it_is", "why_it_matters", "how_it_works", "what_to_expect"]
            },
            "students": {
                "tone": ContentTone.ENGAGING,
                "complexity": "basic_to_intermediate",
                "focus": ["learning_objectives", "key_concepts", "examples", "further_reading"],
                "structure": ["introduction", "core_concepts", "examples", "summary"]
            },
            "experts": {
                "tone": ContentTone.AUTHORITATIVE,
                "complexity": "advanced",
                "focus": ["research_findings", "methodology", "implications", "future_directions"],
                "structure": ["research_overview", "methodology", "results", "discussion"]
            }
        }

    async def create_content_plan(
        self,
        subject: str,
        scope: str,
        target_audience: str,
        target_length: int,
        search_results: List[SearchResult],
        validated_content: List[ValidatedContent]
    ) -> ContentPlan:
        """
        Create a comprehensive content plan tailored to the target audience.

        Args:
            subject: Main topic
            scope: Content scope or specific angle
            target_audience: Target audience description (free-form)
            target_length: Target length in minutes
            search_results: Research results for content planning
            validated_content: Validated claims for inclusion

        Returns:
            Structured content plan with sections and timing
        """
        self.logger.info(
            "Creating content plan",
            subject=subject,
            audience=target_audience,
            target_length=target_length
        )

        # Get audience profile (try to match with defaults, or use general fallback)
        audience_profile = self._get_audience_profile(target_audience)

        # Extract key themes from research
        key_themes = self._extract_key_themes(search_results, validated_content)

        # Generate content sections
        sections = await self._generate_content_sections(
            subject, scope, key_themes, audience_profile, target_length
        )

        # Calculate timing and pacing
        total_duration = self._calculate_total_duration(sections, target_length)

        # Determine overall tone
        overall_tone = ContentTone(audience_profile["tone"])

        # Extract key messages
        key_messages = self._extract_key_messages(key_themes, audience_profile)

        content_plan = ContentPlan(
            subject=subject,
            scope=scope,
            target_audience=target_audience,
            target_length_minutes=target_length,
            sections=sections,
            overall_tone=overall_tone,
            key_messages=key_messages,
            estimated_total_duration=total_duration
        )

        self.logger.info(
            "Content plan created",
            sections=len(sections),
            estimated_duration=total_duration,
            tone=overall_tone.value
        )

        return content_plan

    def _get_audience_profile(self, target_audience: str) -> Dict:
        """
        Get audience profile based on audience description.
        Tries to match with default hints or creates a generic profile.
        
        Args:
            target_audience: Free-form audience description
            
        Returns:
            Dictionary with audience characteristics
        """
        # Normalize audience string for matching
        audience_lower = target_audience.lower().strip()
        
        # Try to find a match in default hints
        for key, profile in self.default_audience_hints.items():
            if key in audience_lower or audience_lower in key:
                self.logger.info(f"Matched audience profile: {key}")
                return profile
        
        # Create a generic profile based on common patterns
        self.logger.info(f"Using generic profile for custom audience: {target_audience}")
        
        # Determine tone based on keywords
        if any(word in audience_lower for word in ["executive", "professional", "manager", "c-level"]):
            tone = ContentTone.PROFESSIONAL
            complexity = "intermediate"
        elif any(word in audience_lower for word in ["developer", "engineer", "technical", "programmer"]):
            tone = ContentTone.ENGAGING
            complexity = "advanced"
        elif any(word in audience_lower for word in ["student", "beginner", "learner"]):
            tone = ContentTone.ENGAGING
            complexity = "basic_to_intermediate"
        elif any(word in audience_lower for word in ["expert", "researcher", "academic", "scientist"]):
            tone = ContentTone.AUTHORITATIVE
            complexity = "advanced"
        else:
            # Default to conversational for general/unknown audiences
            tone = ContentTone.CONVERSATIONAL
            complexity = "intermediate"
        
        return {
            "tone": tone,
            "complexity": complexity,
            "focus": ["overview", "key_insights", "practical_applications", "takeaways"],
            "structure": ["introduction", "main_content", "examples", "conclusion"]
        }

    def _extract_key_themes(
        self,
        search_results: List[SearchResult],
        validated_content: List[ValidatedContent]
    ) -> List[str]:
        """Extract key themes and topics from research results."""
        themes = []

        # Extract from validated content (higher priority)
        for validated in validated_content:
            if validated.validation_status in ["validated", "strongly_validated"]:
                # Extract key phrases from claims
                claim_themes = self._extract_themes_from_claim(validated.original_claim)
                themes.extend(claim_themes)

        # Extract from search results
        for result in search_results:
            if result.credibility_score > 0.7:  # Only high-credibility sources
                # Extract themes from title and summary
                title_themes = self._extract_themes_from_text(result.title)
                summary_themes = self._extract_themes_from_text(result.summary)
                themes.extend(title_themes + summary_themes)

        # Remove duplicates and prioritize
        theme_counts = {}
        for theme in themes:
            theme_counts[theme] = theme_counts.get(theme, 0) + 1

        # Sort by frequency and credibility
        sorted_themes = sorted(
            theme_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )

        return [theme for theme, count in sorted_themes[:10]]  # Top 10 themes

    def _extract_themes_from_claim(self, claim: str) -> List[str]:
        """Extract themes from a validated claim."""
        # Simple keyword extraction - in practice, could use NLP
        important_words = [
            "artificial intelligence", "machine learning", "automation", "digital transformation",
            "innovation", "technology", "data", "analytics", "cloud computing", "cybersecurity",
            "blockchain", "internet of things", "big data", "predictive analytics",
            "natural language processing", "computer vision", "robotics", "augmented reality"
        ]

        found_themes = []
        claim_lower = claim.lower()

        for word in important_words:
            if word in claim_lower:
                found_themes.append(word)

        return found_themes

    def _extract_themes_from_text(self, text: str) -> List[str]:
        """Extract themes from any text content."""
        # Simplified theme extraction
        themes = []

        # Look for technology-related terms
        tech_terms = [
            "AI", "artificial intelligence", "machine learning", "automation",
            "digital", "technology", "innovation", "data", "cloud", "cybersecurity"
        ]

        text_lower = text.lower()
        for term in tech_terms:
            if term in text_lower:
                themes.append(term)

        return themes

    async def _generate_content_sections(
        self,
        subject: str,
        scope: str,
        key_themes: List[str],
        audience_profile: Dict,
        target_length: int
    ) -> List[ContentSection]:
        """Generate content sections based on audience and themes."""
        sections = []

        # Get structure template for audience
        structure_template = audience_profile["structure"]
        focus_areas = audience_profile["focus"]

        # Calculate section timing based on target length
        total_sections = len(structure_template)
        base_section_time = target_length * 60 // total_sections  # in seconds

        for i, section_type in enumerate(structure_template):
            section_title = self._generate_section_title(section_type, subject, scope)
            section_description = self._generate_section_description(
                section_type, subject, scope, key_themes, audience_profile
            )

            # Allocate key points based on focus areas
            key_points = self._generate_key_points(
                section_type, key_themes, focus_areas, audience_profile
            )

            # Calculate section duration
            section_duration = self._calculate_section_duration(
                section_type, base_section_time, total_sections, i
            )

            # Generate transition notes
            transition_notes = self._generate_transition_notes(section_type, i, total_sections)

            sections.append(ContentSection(
                title=section_title,
                description=section_description,
                estimated_duration_seconds=section_duration,
                key_points=key_points,
                transition_notes=transition_notes
            ))

        return sections

    def _generate_section_title(self, section_type: str, subject: str, scope: str) -> str:
        """Generate an appropriate title for a section."""
        titles = {
            "executive_summary": f"Executive Overview: {subject}",
            "key_insights": "Key Insights and Implications",
            "action_items": "Strategic Recommendations",
            "conclusion": "Summary and Next Steps",
            "problem_statement": f"The Challenge: {scope}",
            "solution_overview": "Understanding the Solution",
            "step_by_step": "Implementation Guide",
            "next_steps": "Moving Forward",
            "overview": f"Overview: {subject}",
            "technical_breakdown": "Technical Deep Dive",
            "examples": "Real-World Applications",
            "advanced_topics": "Advanced Considerations",
            "what_it_is": f"What is {subject}?",
            "why_it_matters": "Why This Matters",
            "how_it_works": "How It Works",
            "what_to_expect": "What to Expect",
            "introduction": f"Introduction to {subject}",
            "core_concepts": "Core Concepts Explained",
            "summary": "Key Takeaways",
            "research_overview": "Research Background",
            "methodology": "How We Know This",
            "results": "Key Findings",
            "discussion": "Implications and Applications"
        }

        return titles.get(section_type, f"{section_type.title()}: {subject}")

    def _generate_section_description(
        self,
        section_type: str,
        subject: str,
        scope: str,
        key_themes: List[str],
        audience_profile: Dict
    ) -> str:
        """Generate a detailed description for a section."""
        base_descriptions = {
            "executive_summary": "High-level overview of key findings and strategic implications",
            "key_insights": "Critical insights derived from research and analysis",
            "action_items": "Specific, actionable recommendations for implementation",
            "conclusion": "Summary of key points and strategic next steps",
            "problem_statement": "Clear articulation of the challenge or opportunity",
            "solution_overview": "Overview of available solutions and approaches",
            "step_by_step": "Detailed implementation guidance and best practices",
            "next_steps": "Immediate actions and long-term strategic considerations"
        }

        base_desc = base_descriptions.get(section_type, f"Exploration of {section_type}")

        # Customize for audience
        if audience_profile["complexity"] == "basic":
            base_desc += " explained in clear, accessible terms"
        elif audience_profile["complexity"] == "advanced":
            base_desc += " with technical depth and detailed analysis"

        return base_desc

    def _generate_key_points(
        self,
        section_type: str,
        key_themes: List[str],
        focus_areas: List[str],
        audience_profile: Dict
    ) -> List[str]:
        """Generate key points for a section."""
        # Select relevant themes for this section
        relevant_themes = [theme for theme in key_themes if any(
            focus in theme for focus in focus_areas
        )]

        # Limit to 3-5 key points
        num_points = min(5, max(3, len(relevant_themes)))

        key_points = []
        for i in range(num_points):
            if i < len(relevant_themes):
                theme = relevant_themes[i]
                point = f"Explore {theme} and its relevance to {audience_profile['focus'][0]}"
                key_points.append(point)
            else:
                # Generate generic points
                generic_points = [
                    "Key considerations for implementation",
                    "Potential challenges and solutions",
                    "Real-world applications and examples",
                    "Strategic implications for decision-making"
                ]
                key_points.append(generic_points[i % len(generic_points)])

        return key_points

    def _calculate_section_duration(
        self,
        section_type: str,
        base_time: int,
        total_sections: int,
        section_index: int
    ) -> int:
        """Calculate appropriate duration for a section."""
        # Weight sections differently based on type
        section_weights = {
            "introduction": 1.2,
            "overview": 1.2,
            "executive_summary": 1.5,
            "conclusion": 1.0,
            "summary": 1.0,
            "technical_breakdown": 1.8,
            "examples": 1.3,
            "advanced_topics": 1.5
        }

        weight = section_weights.get(section_type, 1.0)

        # Adjust for position (first and last sections often longer)
        if section_index == 0:
            weight *= 1.2  # Introduction
        elif section_index == total_sections - 1:
            weight *= 1.1  # Conclusion

        return int(base_time * weight)

    def _generate_transition_notes(
        self,
        section_type: str,
        section_index: int,
        total_sections: int
    ) -> str:
        """Generate transition guidance between sections."""
        if section_index == 0:
            return "Start with a strong hook to engage the audience immediately"
        elif section_index == total_sections - 1:
            return "Summarize key points and end with a clear call-to-action"
        else:
            return f"Transition smoothly from {section_type}, building on previous content"

    def _calculate_total_duration(self, sections: List[ContentSection], target_minutes: int) -> int:
        """Calculate total estimated duration from sections."""
        total_seconds = sum(section.estimated_duration_seconds for section in sections)

        # Adjust to be close to target
        target_seconds = target_minutes * 60

        if total_seconds < target_seconds * 0.9:
            # Too short, add buffer time
            buffer_per_section = (target_seconds - total_seconds) // len(sections)
            for section in sections:
                section.estimated_duration_seconds += buffer_per_section
            total_seconds = target_seconds

        return total_seconds

    def _extract_key_messages(self, key_themes: List[str], audience_profile: Dict) -> List[str]:
        """Extract key messages tailored to audience."""
        messages = []

        # Generate audience-specific messages
        if audience_profile["focus"]:
            primary_focus = audience_profile["focus"][0]

            for theme in key_themes[:3]:  # Top 3 themes
                if "roi" in primary_focus:
                    messages.append(f"{theme.title()} delivers measurable business value")
                elif "practical" in primary_focus:
                    messages.append(f"Practical applications of {theme} in real-world scenarios")
                elif "technical" in primary_focus:
                    messages.append(f"Technical implementation of {theme} requires careful consideration")
                else:
                    messages.append(f"{theme.title()} represents a key opportunity for advancement")

        return messages
