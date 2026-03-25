"""
Keyword-based query classifier — zero LLM calls, zero quota cost.

Replaces the original LLM-based classifier which consumed 1 Gemini API
call before EVERY chat message just to decide whether to search documents.

Logic:
  - Default: 'document' (search docs — safe for a RAG system)
  - Override to 'conversational' only when query clearly matches
    greeting / meta / identity patterns
"""
from __future__ import annotations

import re
from typing import Any, Optional

from loguru import logger


# ---------------------------------------------------------------------------
# Patterns that unambiguously indicate a conversational / meta query.
# Anything that doesn't match defaults to document search.
# ---------------------------------------------------------------------------
_CONVERSATIONAL_RE = re.compile(
    r"""
    # Identity / capability questions directed at the AI
    \b(who|what)\s+are\s+you\b                              |
    \byour\s+(name|purpose|capabilit\w+|features?|function)\b |
    \bwhat\s+can\s+you\s+(do|help\s+with)\b                |
    \bhow\s+(do\s+you|can\s+you)\s+(work|help|assist)\b     |
    \btell\s+me\s+about\s+yourself\b                        |
    \bintroduce\s+yourself\b                                |
    \bwhat\s+is\s+novera\b                                  |

    # Greetings
    \b(hello|hi|hey|howdy|greetings?)\b                     |
    \bgood\s+(morning|afternoon|evening|day)\b              |

    # Closings / acknowledgements
    \b(thanks?|thank\s+you|cheers|appreciate(\s+it)?)\b     |
    \b(bye|goodbye|see\s+you|talk\s+later)\b                |

    # Chitchat
    \bhow\s+are\s+you\b                                     |
    \bwhat[' ]?s\s+up\b
    """,
    re.IGNORECASE | re.VERBOSE,
)


class QueryClassifier:
    """
    Instant keyword-based intent classifier.

    Latency  : ~0 ms  (no I/O)
    API cost : $0     (no LLM calls)
    Accuracy : sufficient for a RAG system — when in doubt, search docs.
    """

    async def classify_query(
        self,
        query: str,
        conversation_history: Optional[list] = None,
    ) -> dict[str, Any]:
        q = query.strip()
        word_count = len(q.split())

        if _CONVERSATIONAL_RE.search(q):
            result = {
                "type": "conversational",
                "reasoning": "Matches greeting / identity pattern",
                "confidence": "high",
                "cached": True,
            }
        elif word_count <= 3 and "?" not in q:
            # Very short non-question (e.g. "okay", "sure") — likely small talk
            result = {
                "type": "conversational",
                "reasoning": "Very short non-question input",
                "confidence": "medium",
                "cached": True,
            }
        else:
            # Default: search documents — this is a RAG system
            result = {
                "type": "document",
                "reasoning": "Default: search documents",
                "confidence": "high",
                "cached": True,
            }

        logger.info(
            f"Classification: {result['type'].upper()} "
            f"| '{q[:60]}' "
            f"| confidence={result['confidence']}"
        )
        return result


# Singleton used throughout the app
query_classifier = QueryClassifier()

__all__ = ["QueryClassifier", "query_classifier"]