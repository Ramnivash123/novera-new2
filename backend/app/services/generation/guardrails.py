"""
Guardrails service for input and output validation.
Implements safety checks, hallucination detection, and content filtering.
Enhanced with user-friendly greeting detection and smart topic validation.
"""
from typing import Dict, Any, List, Optional, Tuple
import re
from loguru import logger

from app.core.config import settings


class GuardrailsService:
    """
    Service for validating inputs and outputs to ensure safety and accuracy.
    Implements multiple layers of protection with user-friendly interaction.
    """
    
    def __init__(self):
        self.input_enabled = settings.enable_input_guardrails
        self.output_enabled = settings.enable_output_guardrails
        self.hallucination_threshold = settings.hallucination_threshold
        
        # Greeting patterns (allowed - friendly interaction)
        self.greeting_patterns = [
            r'\b(hi|hello|hey|greetings|good\s+(morning|afternoon|evening)|howdy)\b',
            r'\bhow\s+are\s+you\b',
            r'\bwhat\'s\s+up\b',
            r'\bthanks?\b',
            r'\bthank\s+you\b',
            r'\bbye\b',
            r'\bgoodbye\b'
        ]
        
        # Casual conversation starters (allowed)
        self.casual_starters = [
            r'\bcan\s+you\s+help\b',
            r'\bi\s+need\s+help\b',
            r'\bwhat\s+can\s+you\s+do\b',
            r'\bhow\s+does\s+this\s+work\b',
            r'\btell\s+me\s+about\b'
        ]
        
        # Jailbreak patterns (blocked)
        self.jailbreak_patterns = [
            r'ignore\s+previous\s+instructions',
            r'disregard\s+.*\s+rules',
            r'act\s+as\s+(?:if|though)',
            r'pretend\s+(?:to\s+be|you\s+are)',
            r'bypass\s+restrictions',
            r'override\s+system\s+prompt',
            r'you\s+are\s+no\s+longer',
            r'forget\s+everything'
        ]
        
        # Sensitive data patterns (blocked)
        self.pii_patterns = {
            'SSN': r'\b\d{3}-\d{2}-\d{4}\b',
            'credit card': r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',
            'phone number': r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b'
        }
        
        # Strongly off-topic keywords (only block if multiple present)
        self.strongly_off_topic = [
            'recipe', 'cooking', 'weather forecast', 'sports score',
            'movie review', 'celebrity gossip', 'video game', 'dating advice',
            'astrolog', 'horoscope', 'lottery', 'gambling'
        ]
        
        # Domain keywords (finance/HRMS related)
        self.domain_keywords = [
            # HR/Employee related
            'salary', 'payroll', 'leave', 'vacation', 'employee', 'hr', 'human resources',
            'benefit', 'compensation', 'bonus', 'increment', 'promotion', 'appraisal',
            'attendance', 'timesheet', 'pto', 'sick leave', 'maternity', 'paternity',
            'onboarding', 'offboarding', 'resignation', 'termination', 'hiring',
            'recruitment', 'performance', 'kpi', 'feedback', 'training',
            
            # Finance related
            'finance', 'financial', 'expense', 'reimbursement', 'invoice', 'payment',
            'revenue', 'profit', 'loss', 'budget', 'forecast', 'tax', 'gst',
            'accounting', 'ledger', 'balance sheet', 'income statement', 'cash flow',
            'audit', 'compliance', 'deduction', 'allowance', 'claim',
            
            # Company/Policy related
            'policy', 'procedure', 'guideline', 'rule', 'regulation', 'compliance',
            'department', 'organization', 'company', 'corporate', 'office',
            'manager', 'supervisor', 'director', 'ceo', 'team',
            
            # General business
            'report', 'quarter', 'annual', 'monthly', 'cost', 'contract',
            'agreement', 'document', 'form', 'application', 'approval'
        ]
        
        logger.info(f"Guardrails initialized: input={self.input_enabled}, output={self.output_enabled}")
    
    def validate_input(self, query: str, metadata: Optional[Dict[str, Any]] = None) -> Tuple[bool, Optional[str]]:
        """
        Validate user input query with user-friendly approach.
        
        Args:
            query: User's query
            metadata: Optional metadata (user info, session, etc.)
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not self.input_enabled:
            return True, None
        
        logger.debug(f"Validating input: '{query[:50]}...'")
        
        # 1. Allow greetings and casual conversation starters
        if self._is_greeting_or_casual(query):
            logger.debug("✅ Greeting/casual starter detected - allowing")
            return True, None
        
        # 2. Check for jailbreak attempts (high priority block)
        is_jailbreak, jailbreak_msg = self._detect_jailbreak(query)
        if is_jailbreak:
            logger.warning(f"❌ Jailbreak detected: {jailbreak_msg}")
            return False, "I can't process this type of request. Please ask questions about company policies, finance, or HR matters."
        
        # 3. Check for PII (security concern)
        has_pii, pii_type = self._detect_pii(query)
        if has_pii:
            logger.warning(f"❌ PII detected: {pii_type}")
            return False, f"For security reasons, please don't include {pii_type} in your query. I can help without this sensitive information."
        
        # 4. Smart topic relevance check (lenient)
        is_relevant, confidence = self._check_topic_relevance_smart(query)
        if not is_relevant:
            logger.warning(f"⚠️  Off-topic query detected (confidence: {confidence})")
            return False, (
                "I'm specialized in helping with company finance and HR-related questions. "
                "I can assist with policies, payroll, benefits, expenses, reports, and employee matters. "
                "How can I help you with these topics?"
            )
        
        # 5. Basic length validation
        if len(query) > 2000:
            logger.warning("❌ Query too long")
            return False, "Your query is quite long. Could you please break it down into a shorter, more specific question?"
        
        if len(query.strip()) < 2:
            logger.warning("❌ Query too short")
            return False, "Could you please provide more details about what you'd like to know?"
        
        logger.debug("✅ Input validation passed")
        return True, None
    
    def _is_greeting_or_casual(self, query: str) -> bool:
        """
        Check if query is a greeting or casual conversation starter.
        These should always be allowed for friendly UX.
        
        Returns:
            True if greeting/casual, False otherwise
        """
        query_lower = query.lower().strip()
        
        # Check greeting patterns
        for pattern in self.greeting_patterns:
            if re.search(pattern, query_lower):
                return True
        
        # Check casual starters
        for pattern in self.casual_starters:
            if re.search(pattern, query_lower):
                return True
        
        # Very short queries that are likely greetings
        if len(query_lower.split()) <= 3 and len(query_lower) < 20:
            simple_greetings = ['hi', 'hello', 'hey', 'thanks', 'ok', 'okay', 'yes', 'no']
            if any(greeting in query_lower for greeting in simple_greetings):
                return True
        
        return False
    
    def _check_topic_relevance_smart(self, query: str) -> Tuple[bool, str]:
        """
        Smart topic relevance check with confidence levels.
        More lenient than before, blocks only clearly irrelevant queries.
        
        Returns:
            Tuple of (is_relevant, confidence_level)
        """
        query_lower = query.lower()
        words = query_lower.split()
        
        # 1. Check for strong off-topic indicators
        strong_off_topic_count = sum(
            1 for keyword in self.strongly_off_topic
            if keyword in query_lower
        )
        
        # Block only if multiple strong off-topic indicators
        if strong_off_topic_count >= 2:
            return False, "strongly_off_topic"
        
        # 2. Check for domain keywords (finance/HR)
        domain_matches = sum(
            1 for keyword in self.domain_keywords
            if keyword in query_lower
        )
        
        # If has clear domain keywords, definitely relevant
        if domain_matches >= 1:
            return True, "domain_match"
        
        # 3. Check for question words (suggests genuine query)
        question_words = ['what', 'how', 'when', 'where', 'why', 'who', 'which', 'can', 'is', 'are', 'do', 'does']
        has_question_word = any(word in words for word in question_words)
        
        # 4. Length-based heuristic
        # If it's a substantial question (5+ words) without strong off-topic indicators, allow it
        if len(words) >= 5 and has_question_word and strong_off_topic_count == 0:
            return True, "question_heuristic"
        
        # 5. Check for general business/work terms
        general_business = ['work', 'job', 'office', 'business', 'professional', 'career', 'company']
        if any(term in query_lower for term in general_business):
            return True, "business_context"
        
        # 6. If no strong off-topic indicators and reasonable length, be permissive
        if strong_off_topic_count == 0 and len(words) >= 3:
            return True, "permissive"
        
        # Only block if clearly off-topic
        return False, "unclear_topic"
    
    def validate_output(
        self,
        query: str,
        answer: str,
        context: str,
        sources: List[Dict[str, Any]]
    ) -> Tuple[bool, Optional[str], Dict[str, Any]]:
        """
        Validate generated output for hallucinations and quality.
        
        Args:
            query: Original query
            answer: Generated answer
            context: Context used for generation
            sources: Source documents
            
        Returns:
            Tuple of (is_valid, warning_message, validation_details)
        """
        if not self.output_enabled:
            return True, None, {}
        
        logger.debug("Validating output...")
        
        validation_details = {}
        
        # 1. Check for hallucination
        hallucination_score = self._detect_hallucination(answer, context)
        validation_details['hallucination_score'] = hallucination_score
        
        if hallucination_score > self.hallucination_threshold:
            logger.warning(f"❌ High hallucination score: {hallucination_score:.2f}")
            return False, "Answer may contain unverified information", validation_details
        
        # 2. Check source attribution (only if sources provided)
        if len(sources) > 0:
            has_citations = self._check_citations(answer, sources)
            validation_details['has_citations'] = has_citations
            
            if not has_citations:
                logger.debug("ℹ️  Answer missing citations (non-blocking)")
                validation_details['info'] = "Consider adding source citations"
        
        # 3. Check for speculation (non-blocking, just informational)
        has_speculation = self._detect_speculation(answer)
        validation_details['has_speculation'] = has_speculation
        
        if has_speculation:
            logger.debug("ℹ️  Answer contains some uncertainty language")
        
        # 4. Check answer length (very lenient)
        if len(answer.strip()) < 10:
            logger.warning("⚠️  Answer too short")
            return False, "Generated answer is too brief", validation_details
        
        logger.debug("✅ Output validation passed")
        return True, None, validation_details
    
    def _detect_jailbreak(self, query: str) -> Tuple[bool, Optional[str]]:
        """Detect jailbreak/prompt injection attempts."""
        query_lower = query.lower()
        
        for pattern in self.jailbreak_patterns:
            if re.search(pattern, query_lower):
                return True, f"Pattern matched: {pattern}"
        
        return False, None
    
    def _detect_pii(self, text: str) -> Tuple[bool, Optional[str]]:
        """Detect personally identifiable information."""
        for pii_type, pattern in self.pii_patterns.items():
            if re.search(pattern, text):
                return True, pii_type
        
        return False, None
    
    def _detect_hallucination(self, answer: str, context: str) -> float:
        """
        Detect potential hallucinations by comparing answer to context.
        
        Returns:
            Hallucination score (0-1, higher = more likely hallucinated)
        """
        # Extract numerical claims from answer
        numbers_in_answer = re.findall(r'\$?[\d,]+(?:\.\d+)?%?', answer)
        
        hallucination_indicators = 0
        total_checks = 0
        
        # Check if numbers in answer exist in context
        for number in numbers_in_answer:
            total_checks += 1
            if number not in context:
                hallucination_indicators += 1
        
        # Check for hedging language (indicates model uncertainty)
        hedging_phrases = [
            'i think', 'i believe', 'probably', 'possibly',
            'it seems', 'appears to be', 'might be', 'not sure'
        ]
        
        answer_lower = answer.lower()
        has_hedging = any(phrase in answer_lower for phrase in hedging_phrases)
        
        if has_hedging:
            hallucination_indicators += 1
            total_checks += 1
        
        # Calculate score
        if total_checks == 0:
            return 0.0
        
        score = hallucination_indicators / total_checks
        
        return min(score, 1.0)
    
    def _check_citations(self, answer: str, sources: List[Dict[str, Any]]) -> bool:
        """Check if answer includes proper citations."""
        citation_patterns = [
            r'\[Document:',
            r'\[Source:',
            r'\(Page \d+\)',
            r'according to',
            r'as stated in',
            r'based on'
        ]
        
        has_citation = any(
            re.search(pattern, answer, re.IGNORECASE)
            for pattern in citation_patterns
        )
        
        return has_citation
    
    def _detect_speculation(self, answer: str) -> bool:
        """Detect speculative or uncertain language."""
        speculation_indicators = [
            'may', 'might', 'could', 'possibly', 'perhaps',
            'it seems', 'appears to', 'likely', 'probably'
        ]
        
        answer_lower = answer.lower()
        
        # Count speculation words
        speculation_count = sum(
            answer_lower.count(indicator) for indicator in speculation_indicators
        )
        
        # Only flag if multiple instances (some uncertainty is normal)
        return speculation_count >= 3
    
    def sanitize_output(self, text: str) -> str:
        """Sanitize output text by removing system prompts or internal markers."""
        patterns_to_remove = [
            r'System:.*?\n',
            r'Assistant:.*?\n',
            r'You are an AI.*?\n',
            r'\[INTERNAL\].*?\[/INTERNAL\]'
        ]
        
        sanitized = text
        for pattern in patterns_to_remove:
            sanitized = re.sub(pattern, '', sanitized, flags=re.IGNORECASE | re.DOTALL)
        
        return sanitized.strip()


# Global instance
guardrails_service = GuardrailsService()

__all__ = ['GuardrailsService', 'guardrails_service']