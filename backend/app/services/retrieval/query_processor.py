"""
Query processing and enhancement service.
Handles intent classification, entity extraction, and query expansion.
"""
from typing import Dict, Any, List, Optional, Tuple
import re
from datetime import datetime
from loguru import logger

class QueryProcessor:
    """
    Processes and enhances user queries for better retrieval.
    Implements intent classification, entity extraction, and query expansion.
    """
    
    def __init__(self):
        # Intent patterns
        self.intent_patterns = {
            'factual': [
                r'\bwhat is\b', r'\bwho is\b', r'\bwhere is\b', r'\bwhen\b',
                r'\bdefine\b', r'\bexplain\b', r'\btell me about\b'
            ],
            'procedural': [
                r'\bhow to\b', r'\bhow do i\b', r'\bsteps\b', r'\bprocess\b',
                r'\bprocedure\b', r'\bguidelines\b'
            ],
            'analytical': [
                r'\bcompare\b', r'\bdifference\b', r'\banalyze\b', r'\bcalculate\b',
                r'\bwhy\b', r'\breasons\b'
            ],
            'compliance': [
                r'\bpolicy\b', r'\bregulation\b', r'\brequirement\b', r'\bmust\b',
                r'\bshould\b', r'\bcompliance\b', r'\brule\b'
            ],
            'financial': [
                r'\bsalary\b', r'\bpayroll\b', r'\bexpense\b', r'\brevenue\b',
                r'\bbudget\b', r'\bcost\b', r'\bprice\b', r'\bpayment\b'
            ]
        }
        
        # Finance-specific entities
        self.financial_patterns = {
            'amount': r'\$[\d,]+(?:\.\d{2})?',
            'percentage': r'\d+(?:\.\d+)?%',
            'date': r'\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b',
            'year': r'\b20\d{2}\b',
            'quarter': r'\bQ[1-4]\s*20\d{2}\b'
        }
        
        # Domain synonyms for query expansion
        self.synonyms = {
            'pf': ['provident fund', 'epf', 'employee provident fund'],
            'hra': ['house rent allowance', 'housing allowance'],
            'leave': ['vacation', 'time off', 'absence', 'pto'],
            'salary': ['compensation', 'pay', 'wages', 'remuneration'],
            'policy': ['guideline', 'procedure', 'rule', 'regulation'],
            'expense': ['cost', 'expenditure', 'spending'],
            'revenue': ['income', 'earnings', 'sales']
        }
    
    def process_query(self, query: str) -> Dict[str, Any]:
        """
        Process and enhance a user query.
        
        Args:
            query: Raw user query
            
        Returns:
            Processed query information including intent, entities, and expansions
        """
        query_lower = query.lower().strip()
        
        # 1. Classify intent
        intent = self._classify_intent(query_lower)
        
        # 2. Extract entities
        entities = self._extract_entities(query)
        
        # 3. Expand query with synonyms
        expanded_terms = self._expand_query(query_lower)
        
        # 4. Detect question type
        is_question = query.strip().endswith('?') or any(
            query_lower.startswith(q) for q in ['what', 'who', 'where', 'when', 'why', 'how']
        )
        
        # 5. Extract key phrases
        key_phrases = self._extract_key_phrases(query)
        
        result = {
            'original_query': query,
            'processed_query': query_lower,
            'intent': intent,
            'entities': entities,
            'expanded_terms': expanded_terms,
            'is_question': is_question,
            'key_phrases': key_phrases,
            'query_length': len(query.split()),
            'complexity': self._assess_complexity(query_lower, entities)
        }
        
        logger.info(f"Query processed: intent={intent}, entities={len(entities)}, complexity={result['complexity']}")
        
        return result
    
    def _classify_intent(self, query: str) -> str:
        """
        Classify the intent of the query.
        
        Returns:
            Intent category: factual, procedural, analytical, compliance, financial, general
        """
        for intent, patterns in self.intent_patterns.items():
            for pattern in patterns:
                if re.search(pattern, query):
                    return intent
        
        return 'general'
    
    def _extract_entities(self, query: str) -> Dict[str, List[str]]:
        """
        Extract financial and temporal entities from query.
        
        Returns:
            Dictionary of entity types and their values
        """
        entities = {}
        
        for entity_type, pattern in self.financial_patterns.items():
            matches = re.findall(pattern, query)
            if matches:
                entities[entity_type] = matches
        
        return entities
    
    def _expand_query(self, query: str) -> List[str]:
        """
        Expand query with domain-specific synonyms.
        
        Returns:
            List of expanded terms
        """
        expanded = []
        
        for term, synonyms in self.synonyms.items():
            if term in query:
                expanded.extend(synonyms)
        
        return list(set(expanded))
    
    def _extract_key_phrases(self, query: str) -> List[str]:
        """
        Extract important phrases from query.
        
        Returns:
            List of key phrases (2-4 words)
        """
        # Remove common stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'is', 'are', 'was', 'were'}
        
        words = query.lower().split()
        
        # Extract phrases (2-3 word sequences)
        phrases = []
        for i in range(len(words) - 1):
            phrase = ' '.join(words[i:i+2])
            if not any(w in stop_words for w in words[i:i+2]):
                phrases.append(phrase)
        
        return phrases[:5]  # Return top 5 phrases
    
    def _assess_complexity(self, query: str, entities: Dict[str, List[str]]) -> str:
        """
        Assess query complexity.
        
        Returns:
            Complexity level: simple, moderate, complex
        """
        word_count = len(query.split())
        entity_count = sum(len(v) for v in entities.values())
        has_multiple_questions = query.count('?') > 1
        
        # Complex: multiple questions, many entities, or long query
        if has_multiple_questions or entity_count >= 3 or word_count > 20:
            return 'complex'
        
        # Moderate: some entities or medium length
        if entity_count > 0 or word_count > 10:
            return 'moderate'
        
        # Simple: short, no entities
        return 'simple'
    
    def enhance_query_for_retrieval(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Enhance query string for better retrieval.
        Adds context and expands with synonyms.
        
        Args:
            query: Original query
            context: Optional context (document type, department, etc.)
            
        Returns:
            Enhanced query string
        """
        processed = self.process_query(query)
        
        # Start with original query
        enhanced_parts = [query]
        
        # Add expanded terms
        if processed['expanded_terms']:
            enhanced_parts.extend(processed['expanded_terms'][:3])
        
        # Add context if provided
        if context:
            if 'doc_type' in context:
                enhanced_parts.append(context['doc_type'])
            if 'department' in context:
                enhanced_parts.append(context['department'])
        
        enhanced_query = ' '.join(enhanced_parts)
        
        logger.debug(f"Enhanced query: '{query}' → '{enhanced_query}'")
        
        return enhanced_query
    
    def should_use_semantic_only(self, processed_query: Dict[str, Any]) -> bool:
        """
        Determine if query should use semantic search only (skip keyword).
        
        Returns:
            True if semantic-only is recommended
        """
        # Use semantic only for:
        # - Very short queries (1-2 words)
        # - Conceptual questions
        # - No specific entities
        
        word_count = processed_query['query_length']
        has_entities = bool(processed_query['entities'])
        intent = processed_query['intent']
        
        return (
            word_count <= 2 or
            (intent in ['analytical', 'general'] and not has_entities)
        )
    
    def should_use_keyword_only(self, processed_query: Dict[str, Any]) -> bool:
        """
        Determine if query should use keyword search only (skip semantic).
        
        Returns:
            True if keyword-only is recommended
        """
        # Use keyword only for:
        # - Queries with specific IDs or codes
        # - Exact phrase matches (quotes)
        # - Many specific entities
        
        has_many_entities = sum(len(v) for v in processed_query['entities'].values()) >= 2
        original = processed_query['original_query']
        has_quotes = '"' in original or "'" in original
        
        return has_many_entities or has_quotes

    def reformulate_with_context(
        self,
        query: str,
        conversation_context: Optional['ConversationContext'] = None
    ) -> str:
        """
        Reformulate query using conversation context.
        
        Args:
            query: Original query
            conversation_context: Context from conversation
            
        Returns:
            Reformulated query
        """
        if not conversation_context:
            return query
        
        query_lower = query.lower().strip()
        
        # Detect if query is a follow-up (short, pronoun-heavy, incomplete)
        is_followup = self._is_followup_query(query_lower)
        
        if not is_followup:
            return query
        
        # Enhance with context
        enhanced_query = conversation_context.enhance_query_with_context(query)
        
        # Resolve coreferences
        resolved_query = self._resolve_coreferences(
            enhanced_query,
            conversation_context
        )
        
        logger.info(f"Reformulated: '{query}' -> '{resolved_query}'")
        
        return resolved_query
    
    def _is_followup_query(self, query: str) -> bool:
        """
        Detect if query is a follow-up question.
        
        Indicators:
        - Short length (< 5 words)
        - Contains pronouns (it, that, this, those)
        - Starts with "what about", "how about", "and"
        - No proper nouns or specific entities
        """
        words = query.split()
        
        # Very short queries
        if len(words) <= 4:
            # Check for follow-up patterns
            followup_patterns = [
                r'^(what|how)\s+about',
                r'^and\s+(what|how|the)',
                r'\b(it|that|this|those|them)\b',
                r'^(also|additionally)',
            ]
            
            for pattern in followup_patterns:
                if re.search(pattern, query):
                    return True
        
        # Check for vague references
        vague_words = ['it', 'that', 'this', 'those', 'them', 'same', 'previous']
        if any(word in words for word in vague_words):
            return True
        
        return False
    
    def _resolve_coreferences(
        self,
        query: str,
        context: 'ConversationContext'
    ) -> str:
        """
        Resolve pronouns and references using context.
        
        Examples:
        - "What about it?" -> "What about Q4 revenue in Annual_Report_2023?"
        - "And expenses?" -> "And expenses for Q4 2024?"
        """
        resolved = query
        
        # Replace "it" with last entity or topic
        if re.search(r'\bit\b', resolved, re.IGNORECASE):
            if context.entities:
                # Get most recent non-time entity
                for entity_type in ['amount', 'financial']:
                    if entity_type in context.entities and context.entities[entity_type]:
                        last_entity = context.entities[entity_type][-1]
                        resolved = re.sub(r'\bit\b', last_entity, resolved, flags=re.IGNORECASE)
                        break
        
        # Replace "that" with primary document or time period
        if re.search(r'\bthat\b', resolved, re.IGNORECASE):
            if context.last_time_reference:
                resolved = re.sub(
                    r'\bthat\b',
                    context.last_time_reference,
                    resolved,
                    flags=re.IGNORECASE
                )
        
        return resolved

# Global instance
query_processor = QueryProcessor()

__all__ = ['QueryProcessor', 'query_processor']