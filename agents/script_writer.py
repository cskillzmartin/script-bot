"""Script writing agent for generating polished YouTube scripts using Llama 3."""

import asyncio
import os
import re
from datetime import datetime
from typing import List, Optional
from urllib.parse import urlparse

import structlog
from ollama import Client

from config import settings
from models import ContentPlan, ContentSection, GeneratedScript, ScriptSection, SearchResult, ValidatedContent

logger = structlog.get_logger(__name__)


class ScriptWritingAgent:
    """
    Agent responsible for generating polished, ready-to-record YouTube scripts.

    This agent:
    1. Uses Llama 3 via Ollama for script generation
    2. Applies RAG (Retrieval-Augmented Generation) for content accuracy
    3. Optimizes for specified length and natural speaking flow
    4. Formats scripts for easy reading and recording
    5. Ensures voice-optimized pacing and structure
    """

    def __init__(self):
        """Initialize the script writing agent."""
        self.logger = logger
        self.ollama_client = Client(host=settings.ollama_base_url)

        # Verify Ollama connection and model availability
        self._verify_ollama_setup()

    def _verify_ollama_setup(self):
        """Verify Ollama is running and model is available."""
        try:
            # List available models
            models = self.ollama_client.list()
            model_names = [model.get("name", "") for model in models.get("models", [])]

            if settings.ollama_model not in model_names:
                self.logger.warning(
                    "Ollama model not found",
                    requested_model=settings.ollama_model,
                    available_models=model_names
                )
                # Could attempt to pull the model here if needed
            else:
                self.logger.info("Ollama setup verified", model=settings.ollama_model)

        except Exception as e:
            self.logger.error("Failed to connect to Ollama", error=str(e))
            raise

    async def generate_script(
        self,
        subject: str,
        content_plan: ContentPlan,
        search_results: List[SearchResult],
        validated_content: List[ValidatedContent]
    ) -> GeneratedScript:
        """
        Generate a complete YouTube script using Llama 3.

        Args:
            subject: Main topic of the script
            content_plan: Structured content plan
            search_results: Research results for content
            validated_content: Validated claims for accuracy

        Returns:
            Complete generated script with sections and metadata
        """
        self.logger.info(
            "Starting script generation",
            subject=subject,
            target_length=content_plan.target_length_minutes,
            sections=len(content_plan.sections)
        )

        # Retrieve relevant content for RAG
        relevant_content = await self._retrieve_relevant_content(
            content_plan, search_results, validated_content
        )

        # Generate script sections
        script_sections = await self._generate_script_sections(content_plan, relevant_content)

        # Combine sections into full script
        full_script_content = self._combine_script_sections(script_sections)

        # Optimize for length and pacing
        optimized_script = await self._optimize_script_length(
            full_script_content,
            content_plan.target_length_minutes,
            script_sections
        )

        # Calculate final metadata
        total_word_count = len(optimized_script.split())
        estimated_read_time = self._estimate_read_time(total_word_count)

        # Generate APA citations from search results
        citations = self._generate_apa_citations(search_results)

        # Create script object
        script = GeneratedScript(
            title=self._generate_script_title(subject, content_plan),
            subject=subject,
            target_audience=content_plan.target_audience,
            target_length_minutes=content_plan.target_length_minutes,
            full_content=optimized_script,
            sections=script_sections,
            total_word_count=total_word_count,
            estimated_read_time_seconds=estimated_read_time,
            citations=citations
        )

        # Save script to file
        await self._save_script_to_file(script)

        self.logger.info(
            "Script generation completed",
            word_count=total_word_count,
            estimated_time=estimated_read_time,
            title=script.title
        )

        return script

    async def _retrieve_relevant_content(
        self,
        content_plan: ContentPlan,
        search_results: List[SearchResult],
        validated_content: List[ValidatedContent]
    ) -> str:
        """Retrieve and combine relevant content for script generation."""
        # Combine validated content (high priority)
        validated_text = "\n".join([
            f"CLAIM: {vc.original_claim}\nSUPPORTING SOURCES: {', '.join(vc.supporting_sources)}"
            for vc in validated_content
            if vc.validation_status in ["validated", "strongly_validated"]
        ])

        # Add top search results based on credibility
        top_results = sorted(
            search_results,
            key=lambda x: x.credibility_score,
            reverse=True
        )[:5]

        research_text = "\n---\n".join([
            f"TITLE: {result.title}\nCONTENT: {result.content[:1000]}...\nSOURCE: {result.url}"
            for result in top_results
        ])

        return f"VALIDATED CONTENT:\n{validated_text}\n\nRESEARCH CONTENT:\n{research_text}"

    async def _generate_script_sections(
        self,
        content_plan: ContentPlan,
        relevant_content: str
    ) -> List[ScriptSection]:
        """Generate individual script sections using Llama 3."""
        script_sections = []

        for section in content_plan.sections:
            self.logger.debug(
                "Generating section",
                section_title=section.title,
                estimated_duration=section.estimated_duration_seconds
            )

            # Generate section content
            section_content = await self._generate_single_section(
                section, content_plan, relevant_content
            )

            # Calculate word count and timing for this section
            word_count = len(section_content.split())
            estimated_duration = self._estimate_section_duration(word_count, section)

            script_sections.append(ScriptSection(
                section_title=section.title,
                content=section_content,
                word_count=word_count,
                estimated_duration_seconds=estimated_duration
            ))

        return script_sections

    async def _generate_single_section(
        self,
        section: ContentSection,  # Type hint for section
        content_plan: ContentPlan,
        relevant_content: str
    ) -> str:
        """Generate content for a single script section."""
        # Create focused prompt for this section
        prompt = self._build_section_prompt(section, content_plan, relevant_content)

        try:
            # Generate content using Ollama
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.ollama_client.generate(
                    model=settings.ollama_model,
                    prompt=prompt,
                    options={
                        "temperature": 0.7,
                        "top_p": 0.9,
                        "repeat_penalty": 1.2,  # Increase to reduce repetition
                        "num_predict": 1500  # Allow longer generation per section
                    }
                )
            )

            content = response.get("response", "").strip()

            # Clean up the generated content
            content = self._clean_generated_content(content)

            return content

        except Exception as e:
            self.logger.error(
                "Section generation failed",
                section_title=section.title,
                error=str(e)
            )
            # Return a fallback section
            return self._generate_fallback_section(section)

    def _build_section_prompt(
        self,
        section: ContentSection,
        content_plan: ContentPlan,
        relevant_content: str
    ) -> str:
        """Build a detailed prompt for section generation."""
        # Calculate target words for this section based on its duration
        target_words_for_section = (section.estimated_duration_seconds / 60) * settings.words_per_minute
        target_words_for_section = int(target_words_for_section * 1.2)  # Add 20% buffer
        
        return f"""
You are writing a YouTube video script section. Write ONLY the spoken words - the exact text that will be read aloud by the presenter.

SUBJECT: {content_plan.subject}
SECTION: {section.title}
DESCRIPTION: {section.description}
TARGET AUDIENCE: {content_plan.target_audience}
OVERALL TONE: {content_plan.overall_tone.value}
TARGET WORD COUNT FOR THIS SECTION: approximately {target_words_for_section} words

KEY POINTS TO COVER:
{chr(10).join(f"- {point}" for point in section.key_points)}

RELEVANT CONTENT:
{relevant_content[:2000]}  # Limit context to avoid token limits

CRITICAL INSTRUCTIONS:
- Write ONLY spoken words - no production notes, no b-roll suggestions, no [pause], no [music], no stage directions
- Write in a natural, conversational tone as if speaking directly to the viewer
- DIVE DEEP into the topic - provide comprehensive coverage with detailed explanations
- Include MULTIPLE specific examples, case studies, or real-world scenarios
- Elaborate on each key point with supporting details and context
- Use storytelling and analogies to make complex ideas accessible
- Add data, statistics, or research findings when relevant to the topic
- Anticipate and address common questions or concerns
- Use natural transitions and speaking rhythm throughout
- End with a smooth verbal transition to the next section
- Do NOT include any brackets [], parentheses for actions, or camera directions
- This should be a script someone can read aloud without any edits
- AIM FOR DEPTH AND SUBSTANCE - this is professional content for executives
- IMPORTANT: Write approximately {target_words_for_section} words for this section - be thorough and comprehensive

Write a COMPREHENSIVE spoken script for this section ({target_words_for_section} words target, no production notes):
"""

    def _clean_generated_content(self, content: str) -> str:
        """Clean and format generated script content, removing production notes."""
        # Remove common LLM artifacts
        content = content.replace("Here's the script:", "").strip()
        content = content.replace("Script:", "").strip()
        content = content.replace("**", "").strip()  # Remove markdown bold
        
        # Remove production notes and stage directions
        # Remove [bracketed] content like [pause], [b-roll], [music], etc.
        content = re.sub(r'\[.*?\]', '', content)
        
        # Remove (parenthetical) production notes
        content = re.sub(r'\([A-Z][^)]*\)', '', content)  # Remove (UPPERCASE) directions
        
        # Remove common production phrases
        production_phrases = [
            '[Intro Music Fades Out]',
            '[Pause for a brief moment]',
            '[Pause]',
            '[Music]',
            '[B-Roll]',
            '[Cut to]',
            '[Show]',
            '[Display]',
            '[Conclude with a smooth transition]',
            '[Insert link to next section]',
            'Next: [Insert link to next section]'
        ]
        for phrase in production_phrases:
            content = content.replace(phrase, '')
        
        # Clean up multiple spaces and newlines
        content = re.sub(r'\s+', ' ', content)
        content = re.sub(r'\n\s*\n\s*\n+', '\n\n', content)

        # Ensure proper capitalization
        sentences = content.split(". ")
        capitalized_sentences = []

        for sentence in sentences:
            sentence = sentence.strip()
            if sentence and len(sentence) > 0:
                # Capitalize first letter if it makes sense
                if not sentence[0].isupper() and len(sentence) > 10:
                    sentence = sentence[0].upper() + sentence[1:]
                capitalized_sentences.append(sentence)

        return ". ".join(capitalized_sentences)

    def _generate_fallback_section(self, section: ContentSection) -> str:
        """Generate a basic fallback section if LLM fails."""
        return f"""
Now let's talk about {section.title.lower()}. This is an important aspect of {section.description.lower()}.

{chr(10).join(f"First, {point.lower()}" for point in section.key_points[:3])}

Remember, the key takeaway here is that this matters because it helps us understand the bigger picture of our topic.
"""

    def _combine_script_sections(self, script_sections: List[ScriptSection]) -> str:
        """Combine all script sections into a complete script."""
        combined_sections = []

        for i, section in enumerate(script_sections):
            # Add section header
            section_header = f"""
{'='*50}
{section.section_title.upper()}
{'='*50}

"""
            combined_sections.append(section_header + section.content.strip())

            # Add transition if not the last section
            if i < len(script_sections) - 1:
                next_section = script_sections[i + 1]
                transition = f"""

[Transition: Moving into our next topic - {next_section.section_title}]

"""
                combined_sections.append(transition)

        return "\n".join(combined_sections)

    async def _optimize_script_length(
        self,
        script_content: str,
        target_minutes: int,
        script_sections: List[ScriptSection]
    ) -> str:
        """Optimize script length to match target duration."""
        current_word_count = len(script_content.split())
        target_word_count = target_minutes * settings.words_per_minute

        self.logger.info(
            "Optimizing script length",
            current_words=current_word_count,
            target_words=target_word_count
        )

        # Check if optimization is needed
        word_ratio = current_word_count / target_word_count if target_word_count > 0 else 1.0

        # Tighter tolerance: must be within 90-110% of target
        if 0.9 <= word_ratio <= 1.1:  # Within 10% of target
            self.logger.info("Script length is within acceptable range, no optimization needed")
            return script_content

        # Generate optimized version
        optimization_prompt = self._build_optimization_prompt(
            script_content, target_minutes, word_ratio
        )

        try:
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.ollama_client.generate(
                    model=settings.ollama_model,
                    prompt=optimization_prompt,
                    options={
                        "temperature": 0.5,
                        "top_p": 0.8,
                        "repeat_penalty": 1.2,  # Reduce repetition
                        "num_predict": target_word_count * 2  # Base on target, not current length
                    }
                )
            )

            optimized_content = response.get("response", "").strip()
            optimized_word_count = len(optimized_content.split())

            self.logger.info(
                "Optimization complete",
                original_words=current_word_count,
                optimized_words=optimized_word_count,
                target_words=target_word_count
            )

            # Recalculate sections with optimized content
            updated_sections = await self._update_section_durations(
                optimized_content, script_sections
            )

            return optimized_content

        except Exception as e:
            self.logger.warning("Script optimization failed, using original", error=str(e))
            return script_content

    def _build_optimization_prompt(
        self,
        script_content: str,
        target_minutes: int,
        word_ratio: float
    ) -> str:
        """Build prompt for script length optimization."""
        target_word_count = target_minutes * settings.words_per_minute
        current_word_count = len(script_content.split())
        word_deficit = target_word_count - current_word_count
        
        if word_ratio > 1.1:  # Too long
            instruction = f"shorten the script by approximately {abs(word_deficit)} words while preserving all key points"
            expansion_notes = ""
        elif word_ratio < 0.9:  # Too short
            instruction = f"expand the script by approximately {abs(word_deficit)} words with more detail, examples, and depth"
            expansion_notes = f"""
EXPANSION STRATEGY (you need to add ~{abs(word_deficit)} more words):
- Add 2-3 detailed real-world examples or case studies
- Expand each main point with supporting context and explanation
- Include relevant data, statistics, or research findings
- Add analogies or metaphors to clarify complex concepts
- Anticipate and address potential questions or concerns
- Provide more context about why each point matters
- Include transition sentences between ideas
"""
        else:
            instruction = "refine the script for better flow and clarity while maintaining current length"
            expansion_notes = ""

        return f"""
I have a YouTube script that needs length optimization for SPOKEN CONTENT ONLY.

Current word count: {current_word_count} words
Target word count: {target_word_count} words (for {target_minutes} minutes of speaking)
Gap: {word_deficit} words ({"need to ADD" if word_deficit > 0 else "need to REMOVE"} approximately {abs(word_deficit)} words)
Reading pace: {settings.words_per_minute} words per minute

Task: {instruction}
{expansion_notes}

Original script:
{script_content}

CRITICAL REQUIREMENTS:
- Output ONLY spoken words that will be read by the presenter
- NO production notes, NO b-roll suggestions, NO [brackets], NO stage directions
- YOU MUST generate approximately {target_word_count} words total
- Maintain natural speaking rhythm and conversational flow
- Preserve all key points and information
- Use engaging, conversational language suitable for executives
- Every word should be meant to be spoken aloud
- If expanding: Add substantial content, not filler - provide real value and depth

Provide ONLY the optimized spoken script (~{target_word_count} words, no production notes):
"""

    async def _update_section_durations(
        self,
        optimized_content: str,
        original_sections: List[ScriptSection]
    ) -> List[ScriptSection]:
        """Update section durations based on optimized content."""
        # Simple proportional update - in practice, could be more sophisticated
        total_original_words = sum(section.word_count for section in original_sections)
        total_optimized_words = len(optimized_content.split())

        if total_original_words == 0:
            return original_sections

        ratio = total_optimized_words / total_original_words

        updated_sections = []
        for section in original_sections:
            new_duration = int(section.estimated_duration_seconds * ratio)
            new_word_count = int(section.word_count * ratio)

            updated_sections.append(ScriptSection(
                section_title=section.section_title,
                content=section.content,  # Keep original content reference
                word_count=new_word_count,
                estimated_duration_seconds=new_duration
            ))

        return updated_sections

    def _estimate_read_time(self, word_count: int) -> int:
        """Estimate speaking time in seconds."""
        words_per_minute = settings.words_per_minute
        minutes = word_count / words_per_minute
        return int(minutes * 60)

    def _estimate_section_duration(self, word_count: int, section: ContentSection) -> int:
        """Estimate duration for a specific section."""
        # Base calculation
        base_duration = self._estimate_read_time(word_count)

        # Adjust based on section type and complexity
        complexity_multipliers = {
            "technical_breakdown": 1.2,
            "examples": 1.1,
            "introduction": 1.0,
            "conclusion": 1.0
        }

        section_type = section.title.lower()
        multiplier = 1.0
        for key, mult in complexity_multipliers.items():
            if key in section_type:
                multiplier = mult
                break

        return int(base_duration * multiplier)

    def _generate_script_title(self, subject: str, content_plan: ContentPlan) -> str:
        """Generate an engaging title for the script."""
        audience = content_plan.target_audience
        return f"{subject.title()} for {audience.title()}: A Complete Guide"

    def _generate_apa_citations(self, search_results: List[SearchResult]) -> List[str]:
        """
        Generate APA 7th edition style citations from search results.
        
        Args:
            search_results: List of search results used in the script
            
        Returns:
            List of APA-formatted citations
        """
        citations = []
        current_year = datetime.now().year
        
        for result in search_results:
            try:
                # Extract domain name for author (when author is not available)
                parsed_url = urlparse(result.url)
                domain = parsed_url.netloc.replace('www.', '').replace('.com', '').replace('.org', '').replace('.edu', '').replace('.gov', '')
                
                # Clean up title
                title = result.title.strip()
                if not title.endswith('.'):
                    title += '.'
                
                # Extract year from metadata or URL, or use current year
                year = current_year
                if result.metadata and 'year' in result.metadata:
                    year = result.metadata['year']
                elif result.extracted_at:
                    year = result.extracted_at.year
                
                # Get retrieval date
                retrieval_date = result.extracted_at.strftime("%B %d, %Y") if result.extracted_at else datetime.now().strftime("%B %d, %Y")
                
                # Capitalize domain for author-like appearance
                author = domain.title()
                
                # Format APA citation
                # Format: Author/Organization. (Year). Title. Retrieved Month Day, Year, from URL
                citation = f"{author}. ({year}). {title} Retrieved {retrieval_date}, from {result.url}"
                
                citations.append(citation)
                
            except Exception as e:
                self.logger.warning(f"Failed to generate citation for {result.url}: {e}")
                # Fallback citation
                citations.append(f"Source. (n.d.). {result.title} Retrieved from {result.url}")
        
        # Remove duplicates and sort alphabetically
        citations = sorted(list(set(citations)))
        
        return citations

    async def _save_script_to_file(self, script: GeneratedScript):
        """Save the generated script to a file."""
        # Create filename from subject and audience
        safe_subject = "".join(c for c in script.subject if c.isalnum() or c in " -").strip()
        safe_audience = "".join(c for c in script.target_audience if c.isalnum() or c in " -").strip().replace(" ", "-")

        filename = f"{safe_subject[:50]}-{safe_audience[:30]}.txt"
        filepath = os.path.join(settings.scripts_directory, filename)

        try:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write("=" * 60 + "\n")
                f.write(f"SCRIPT: {script.title}\n")
                f.write(f"SUBJECT: {script.subject}\n")
                f.write(f"AUDIENCE: {script.target_audience}\n")
                f.write(f"TARGET LENGTH: {script.target_length_minutes} minutes\n")
                f.write(f"ESTIMATED READ TIME: {script.estimated_read_time_seconds // 60} minutes\n")
                f.write(f"WORD COUNT: {script.total_word_count}\n")
                f.write("=" * 60 + "\n\n")

                f.write(script.full_content)

                # Add citations section
                if script.citations:
                    f.write("\n\n" + "=" * 60 + "\n")
                    f.write("REFERENCES (APA 7th Edition)\n")
                    f.write("=" * 60 + "\n\n")
                    for i, citation in enumerate(script.citations, 1):
                        f.write(f"{citation}\n\n")

                f.write("\n" + "=" * 60 + "\n")
                f.write("SCRIPT METADATA\n")
                f.write(f"Generated: {script.created_at}\n")
                f.write(f"Sections: {len(script.sections)}\n")
                f.write(f"Sources Cited: {len(script.citations)}\n")
                f.write("=" * 60 + "\n")

            self.logger.info("Script saved to file", filepath=filepath)

        except Exception as e:
            self.logger.error("Failed to save script", error=str(e))
            raise
