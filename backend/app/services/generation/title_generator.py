"""
Intelligent chunk title generation service.
Uses Gemini to create concise, meaningful titles for chunks.
"""
from typing import Optional, List
import google.generativeai as genai
from tenacity import retry, stop_after_attempt, wait_exponential
from loguru import logger
import asyncio

from app.core.config import settings


class TitleGeneratorService:
    """
    Generate intelligent, context-aware titles for document chunks.
    Uses Gemini to analyze content and create concise titles.
    """
    
    def __init__(self):
        genai.configure(api_key=settings.gemini_api_key)
        
        # Use the model from settings (gemini-2.5-flash)
        model_name = settings.gemini_chat_model.replace('models/', '')
        
        self.model = genai.GenerativeModel(
            model_name=model_name,
            generation_config={
                "temperature": 0.3,
                "max_output_tokens": 50,
                "top_p": 0.95,
            }
        )
        
        logger.info(f"Title generation service initialized with model: {model_name}")
    
    def _extract_text_from_response(self, response) -> Optional[str]:
        """
        Safely extract text from Gemini response.
        Handles both simple and complex response structures.
        """
        try:
            # Try simple text accessor first
            if hasattr(response, 'text') and response.text:
                return response.text
        except Exception:
            pass
        
        try:
            # Try parts accessor
            if hasattr(response, 'parts') and response.parts:
                text_parts = []
                for part in response.parts:
                    if hasattr(part, 'text'):
                        text_parts.append(part.text)
                if text_parts:
                    return ''.join(text_parts)
        except Exception:
            pass
        
        try:
            # Try candidates accessor
            if hasattr(response, 'candidates') and response.candidates:
                for candidate in response.candidates:
                    if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
                        text_parts = []
                        for part in candidate.content.parts:
                            if hasattr(part, 'text'):
                                text_parts.append(part.text)
                        if text_parts:
                            return ''.join(text_parts)
        except Exception as e:
            logger.warning(f"Error extracting text from response: {e}")
        
        return None
    
    @retry(
        stop=stop_after_attempt(2),
        wait=wait_exponential(multiplier=1, min=1, max=3)
    )
    async def generate_title(
        self,
        content: str,
        chunk_type: str,
        section_title: Optional[str] = None,
        page_numbers: Optional[List[int]] = None,
        chunk_index: Optional[int] = None
    ) -> str:
        """
        Generate a concise, meaningful title for a chunk.
        
        Args:
            content: Chunk content
            chunk_type: Type of chunk (text, table, summary)
            section_title: Section context if available
            page_numbers: Page numbers for reference
            chunk_index: Index of chunk in document
            
        Returns:
            Generated title (max 60 characters)
        """
        # Handle special chunk types
        if chunk_type == 'summary':
            return "Document Overview"
        
        if chunk_type == 'table':
            # Try to extract table context
            table_title = await self._generate_table_title(content, section_title)
            return table_title
        
        # For regular text chunks
        return await self._generate_text_title(content, section_title, chunk_index)
    
    async def _generate_text_title(
        self,
        content: str,
        section_title: Optional[str],
        chunk_index: Optional[int]
    ) -> str:
        """Generate title for text chunk."""
        # Truncate content for prompt (use first 500 chars)
        preview = content[:500] if len(content) > 500 else content
        
        # Build context
        context_hint = ""
        if section_title:
            context_hint = f"This content is from the section: '{section_title}'.\n"
        
        prompt = f"""{context_hint}Analyze this text excerpt and create a concise, descriptive title (max 8 words):

TEXT:
{preview}

RULES:
- Maximum 8 words
- Minimum 3 words (avoid single-word titles)
- Describe the main topic/concept
- Be specific and actionable
- Use title case
- NO generic words like "Information", "Details", "Overview" unless necessary
- Focus on the KEY subject matter
- Respond with ONLY the title, nothing else

TITLE:"""
        
        try:
            loop = asyncio.get_running_loop()
            
            response = await loop.run_in_executor(
                None,
                lambda: self.model.generate_content(prompt)
            )
            
            # Use safe text extraction
            title_text = self._extract_text_from_response(response)
            
            if not title_text:
                raise ValueError("No text extracted from response")
            
            title = title_text.strip()
            
            # Clean up title
            title = title.replace('"', '').replace("'", '').strip()
            title = title.replace('TITLE:', '').replace('Title:', '').strip()
            
            # Validate minimum length (avoid single-word titles like "Mal", "Micro")
            if len(title) < 3 or len(title.split()) < 2:
                logger.warning(f"Generated title too short: '{title}', using fallback")
                raise ValueError("Generated title too short")
            
            # Ensure max length (60 chars)
            if len(title) > 60:
                title = title[:57] + "..."
            
            logger.debug(f"Generated title: '{title}'")
            return title
            
        except Exception as e:
            logger.warning(f"Title generation failed: {e}")
            # Fallback: Use first sentence or section title
            return self._generate_fallback_title(content, section_title, chunk_index)
    
    async def _generate_table_title(
        self,
        content: str,
        section_title: Optional[str]
    ) -> str:
        """Generate title for table chunk."""
        # Extract first few lines of table
        lines = content.split('\n')[:5]
        preview = '\n'.join(lines)
        
        context_hint = ""
        if section_title:
            context_hint = f"From section: '{section_title}'.\n"
        
        prompt = f"""{context_hint}This is a table. Create a concise title (max 8 words) describing what data it contains:

TABLE PREVIEW:
{preview}

RULES:
- Maximum 8 words
- Minimum 3 words
- Start with "Table:" if helpful
- Describe what the table shows
- Be specific about the data type
- Respond with ONLY the title, nothing else

TITLE:"""
        
        try:
            loop = asyncio.get_running_loop()
            
            response = await loop.run_in_executor(
                None,
                lambda: self.model.generate_content(prompt)
            )
            
            # Use safe text extraction
            title_text = self._extract_text_from_response(response)
            
            if not title_text:
                raise ValueError("No text extracted from response")
            
            title = title_text.strip()
            title = title.replace('"', '').replace("'", '').strip()
            title = title.replace('TITLE:', '').replace('Title:', '').strip()
            
            # Validate minimum length
            if len(title) < 3 or len(title.split()) < 2:
                raise ValueError("Generated title too short")
            
            if len(title) > 60:
                title = title[:57] + "..."
            
            logger.debug(f"Generated table title: '{title}'")
            return title
            
        except Exception as e:
            logger.warning(f"Table title generation failed: {e}")
            if section_title:
                return f"Table: {section_title[:50]}"
            return "Data Table"
    
    def _generate_fallback_title(
        self,
        content: str,
        section_title: Optional[str],
        chunk_index: Optional[int]
    ) -> str:
        """
        Generate fallback title if AI generation fails.
        
        Uses first sentence or section title as basis.
        """
        # Try to use section title
        if section_title and len(section_title) > 0:
            title = section_title[:60]
            if len(section_title) > 60:
                title += "..."
            logger.debug(f"Using section title as fallback: '{title}'")
            return title
        
        # Try to extract first sentence
        sentences = content.split('.')
        if sentences and len(sentences[0]) > 5:
            first_sentence = sentences[0].strip()
            if len(first_sentence) > 60:
                title = first_sentence[:57] + "..."
            else:
                title = first_sentence
            logger.debug(f"Using first sentence as fallback: '{title}'")
            return title
        
        # Last resort: Use first 60 characters
        if len(content) > 60:
            title = content[:57] + "..."
        else:
            title = content if content else f"Chunk {chunk_index + 1 if chunk_index is not None else ''}"
        
        logger.debug(f"Using content preview as fallback: '{title}'")
        return title
    
    async def batch_generate_titles(
        self,
        chunks: List[dict]
    ) -> List[str]:
        """
        Generate titles for multiple chunks efficiently.
        
        Args:
            chunks: List of chunk dictionaries with content, type, etc.
            
        Returns:
            List of generated titles in same order
        """
        titles = []
        
        for i, chunk in enumerate(chunks):
            try:
                logger.info(f"Generating title for chunk {i+1}/{len(chunks)}")
                
                title = await self.generate_title(
                    content=chunk.get('content', ''),
                    chunk_type=chunk.get('chunk_type', 'text'),
                    section_title=chunk.get('section_title'),
                    page_numbers=chunk.get('page_numbers'),
                    chunk_index=chunk.get('chunk_index')
                )
                titles.append(title)
                
                logger.success(f"Generated title {i+1}: '{title}'")
                
            except Exception as e:
                logger.error(f"Batch title generation failed for chunk {chunk.get('chunk_index')}: {e}")
                fallback_title = self._generate_fallback_title(
                    chunk.get('content', ''),
                    chunk.get('section_title'),
                    chunk.get('chunk_index')
                )
                titles.append(fallback_title)
                logger.warning(f"Using fallback title: '{fallback_title}'")
        
        logger.info(f"Batch title generation complete: {len(titles)} titles generated")
        return titles


# Global instance
title_generator = TitleGeneratorService()

__all__ = ['TitleGeneratorService', 'title_generator']