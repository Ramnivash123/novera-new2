"""
LLM-powered suggestion service for intelligent follow-up questions.
Generates context-aware suggestions using Gemini for natural conversation flow.
"""
from typing import List, Dict, Any, Optional
from loguru import logger
import asyncio


class SuggestionService:
    """
    Generates intelligent follow-up question suggestions using LLM.
    Analyzes conversation context, last response, and document sources
    to provide relevant, natural suggestions.
    """
    
    async def generate_suggestions(
        self,
        last_query: str,
        last_response: str,
        context_summary: Dict[str, Any],
        sources: List[Dict[str, Any]],
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> List[str]:
        """
        Generate 3-4 intelligent, context-aware follow-up suggestions using LLM.
        
        Args:
            last_query: User's last query
            last_response: AI's last response
            context_summary: Current conversation context from context_manager
            sources: Sources used in last response
            conversation_history: Recent conversation turns (last 2-3 exchanges)
            
        Returns:
            List of 3-4 suggested follow-up questions
        """
        logger.info("Generating LLM-powered context-aware suggestions")
        
        try:
            # Import here to avoid circular dependency
            from app.services.generation.llm_service import llm_service
            
            # Build context-rich prompt
            prompt = self._build_suggestion_prompt(
                last_query=last_query,
                last_response=last_response,
                context_summary=context_summary,
                sources=sources,
                conversation_history=conversation_history
            )
            
            # Generate suggestions using Gemini Flash (fast + cheap)
            suggestions = await llm_service.generate_follow_up_suggestions(prompt)
            
            # Validate and filter suggestions
            valid_suggestions = self._validate_suggestions(suggestions, last_query)
            
            # Return top 4
            final_suggestions = valid_suggestions[:4]
            
            logger.info(f"✅ Generated {len(final_suggestions)} context-aware suggestions")
            logger.debug(f"Suggestions: {final_suggestions}")
            
            return final_suggestions
            
        except Exception as e:
            logger.error(f"Suggestion generation failed: {str(e)}", exc_info=True)
            
            # Fallback to basic contextual suggestions
            return self._get_fallback_suggestions(context_summary, last_response)
    
    def _build_suggestion_prompt(
        self,
        last_query: str,
        last_response: str,
        context_summary: Dict[str, Any],
        sources: List[Dict[str, Any]],
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """
        Build a comprehensive prompt for LLM suggestion generation.
        
        Returns:
            Formatted prompt string
        """
        # Extract context information
        primary_doc = context_summary.get('primary_document', 'N/A')
        intent = context_summary.get('last_intent', 'general')
        entities = context_summary.get('entities', {})
        time_period = context_summary.get('recent_time_period')
        
        # Build source summary
        source_info = "No sources"
        if sources:
            unique_docs = set(s.get('document', 'Unknown') for s in sources)
            source_info = f"{len(sources)} chunks from: {', '.join(list(unique_docs)[:3])}"
        
        # Build conversation context
        conv_context = ""
        if conversation_history and len(conversation_history) > 0:
            conv_context = "**Recent Conversation:**\n"
            for msg in conversation_history[-4:]:  # Last 2 exchanges (4 messages)
                role = "User" if msg['role'] == 'user' else "AI"
                content = msg['content'][:150]  # Truncate for brevity
                conv_context += f"{role}: {content}...\n"
        
        # Truncate response for prompt (save tokens)
        response_preview = last_response[:800] if len(last_response) > 800 else last_response
        
        # Build comprehensive prompt
        prompt = f"""You are an AI assistant helping generate intelligent follow-up questions for a conversation about documents.

**Current Context:**
- Document Focus: {primary_doc}
- Query Intent: {intent}
- Sources Used: {source_info}
{f"- Time Period: {time_period}" if time_period else ""}

{conv_context}

**Latest Exchange:**
User Question: "{last_query}"

AI Response: "{response_preview}"

**Your Task:**
Generate 3-4 natural, intelligent follow-up questions that:

1. **Are HIGHLY RELEVANT** to what was just discussed
2. **Build naturally** on the AI's response
3. **Consider the document context** ({primary_doc})
4. **Match the conversation intent** ({intent})
5. **Are specific and actionable** - no generic questions
6. **Sound conversational** - like a real person asking
7. **Explore different angles** - clarification, deeper detail, related topics, practical application

**Examples of GOOD suggestions:**
- "How do I apply for this leave?"
- "What are the eligibility criteria?"
- "Can you show me the calculation for this?"
- "What documents do I need to submit?"

**Examples of BAD suggestions (avoid these):**
- "Tell me more" (too generic)
- "Explain in simpler terms" (not specific)
- "What else is there?" (too vague)
- "Can you elaborate?" (not actionable)

**Output Format:**
Provide ONLY the questions, one per line, no numbering, no bullets:

Question 1
Question 2
Question 3
Question 4

Generate the suggestions now:"""

        return prompt
    
    def _validate_suggestions(
        self,
        suggestions: List[str],
        last_query: str
    ) -> List[str]:
        """
        Validate and filter generated suggestions.
        
        Args:
            suggestions: Raw suggestions from LLM
            last_query: User's last query (to avoid duplicates)
            
        Returns:
            Filtered list of valid suggestions
        """
        valid = []
        last_query_lower = last_query.lower().strip()
        
        for suggestion in suggestions:
            # Clean suggestion
            cleaned = suggestion.strip()
            
            # Remove numbering/bullets if present
            cleaned = self._remove_numbering(cleaned)

            # Max 150 characters per suggestion
            MAX_SUGGESTION_LENGTH = 150
            if len(cleaned) > MAX_SUGGESTION_LENGTH:
                logger.warning(f"Suggestion too long ({len(cleaned)} chars), truncating: {cleaned[:50]}...")
                # Truncate at last complete word before 150 chars
                cleaned = cleaned[:MAX_SUGGESTION_LENGTH].rsplit(' ', 1)[0]
                if not cleaned.endswith('?'):
                    cleaned += '?'
            
            # Skip empty
            if not cleaned or len(cleaned) < 10:
                continue
            
            # Skip if too similar to last query
            if self._is_too_similar(cleaned, last_query_lower):
                logger.debug(f"Skipping similar suggestion: {cleaned}")
                continue
            
            # Skip generic/template phrases
            if self._is_generic(cleaned):
                logger.debug(f"Skipping generic suggestion: {cleaned}")
                continue
            
            # Add question mark if missing
            if not cleaned.endswith('?'):
                cleaned += '?'
            
            valid.append(cleaned)
        
        return valid
    
    def _remove_numbering(self, text: str) -> str:
        """Remove numbering/bullets from suggestion."""
        import re
        
        # Remove patterns like "1. ", "1) ", "- ", "• ", etc.
        patterns = [
            r'^\d+\.\s*',      # "1. "
            r'^\d+\)\s*',      # "1) "
            r'^[-•*]\s*',      # "- ", "• ", "* "
            r'^Question\s*\d+:\s*',  # "Question 1: "
        ]
        
        for pattern in patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)
        
        return text.strip()
    
    def _is_too_similar(self, suggestion: str, last_query: str) -> bool:
        """Check if suggestion is too similar to last query."""
        suggestion_lower = suggestion.lower()
        
        # Simple similarity check (can be enhanced)
        common_words = set(suggestion_lower.split()) & set(last_query.split())
        
        # If >70% words overlap, consider too similar
        if len(common_words) / max(len(suggestion_lower.split()), 1) > 0.7:
            return True
        
        return False
    
    def _is_generic(self, suggestion: str) -> bool:
        """Check if suggestion is too generic."""
        generic_phrases = [
            'tell me more',
            'explain in simpler terms',
            'can you elaborate',
            'give me an example',
            'what else',
            'tell me about',
            'can you explain',
            'show me more',
            'provide more details',
            'give more information',
        ]
        
        suggestion_lower = suggestion.lower()
        
        for phrase in generic_phrases:
            if phrase in suggestion_lower:
                return True
        
        return False
    
    def _get_fallback_suggestions(
        self,
        context_summary: Dict[str, Any],
        last_response: str
    ) -> List[str]:
        """
        Generate basic fallback suggestions if LLM generation fails.
        
        Args:
            context_summary: Context information
            last_response: Last AI response
            
        Returns:
            List of basic but contextual suggestions
        """
        logger.warning("Using fallback suggestion generation")
        
        suggestions = []
        
        intent = context_summary.get('last_intent', 'general')
        primary_doc = context_summary.get('primary_document')
        
        # Intent-based fallbacks (better than completely generic)
        if intent == 'financial':
            suggestions.extend([
                "Can you show me the detailed breakdown?",
                "How does this compare to previous periods?",
                "What are the key drivers of these numbers?",
            ])
        elif intent == 'procedural':
            suggestions.extend([
                "What are the step-by-step instructions?",
                "Who should I contact for this?",
                "What documents are required?",
            ])
        elif intent == 'compliance':
            suggestions.extend([
                "Are there any recent policy updates?",
                "What are the consequences of non-compliance?",
                "Who is responsible for enforcement?",
            ])
        else:
            # General fallbacks
            suggestions.extend([
                "Can you provide more specific details?",
                "What are the key points I should know?",
                "Where can I find the official documentation?",
            ])
        
        # Add document-specific suggestion if available
        if primary_doc:
            suggestions.append(f"What else does {primary_doc} cover?")
        
        # Response-based suggestions
        if any(word in last_response.lower() for word in ['must', 'required', 'mandatory']):
            suggestions.append("What if I cannot meet these requirements?")
        
        if any(word in last_response.lower() for word in ['deadline', 'due', 'by']):
            suggestions.append("Can the deadline be extended?")
        
        # Deduplicate and return top 4
        unique_suggestions = list(dict.fromkeys(suggestions))  # Preserve order
        
        return unique_suggestions[:4]
    
    def filter_suggestions_by_confidence(
        self,
        suggestions: List[str],
        confidence: str
    ) -> List[str]:
        """
        Filter suggestions based on response confidence.
        For low confidence, prioritize clarification questions.
        
        Args:
            suggestions: Generated suggestions
            confidence: Response confidence level
            
        Returns:
            Filtered suggestions
        """
        if confidence == 'low':
            # For low confidence, add clarification suggestions
            clarification_suggestions = [
                "Can you rephrase my question?",
                "Can you search in different documents?",
            ]
            
            # Combine with original suggestions
            return clarification_suggestions[:2] + suggestions[:2]
        
        return suggestions


# Global instance
suggestion_service = SuggestionService()

__all__ = ['SuggestionService', 'suggestion_service']