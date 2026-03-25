"""
Intelligent chunking service with semantic awareness.
Implements multi-strategy chunking for optimal retrieval performance.
"""
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import tiktoken
from loguru import logger

from app.core.config import settings
from app.services.document_processing.text_extractor import ExtractedElement, DocumentStructure


@dataclass
class Chunk:
    """Represents a single text chunk with metadata."""
    content: str
    chunk_type: str  # 'text', 'table', 'summary', 'header'
    chunk_index: int
    page_numbers: List[int]
    section_title: Optional[str]
    token_count: int
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for database storage."""
        return {
            'content': self.content,
            'chunk_type': self.chunk_type,
            'chunk_index': self.chunk_index,
            'page_numbers': self.page_numbers,
            'section_title': self.section_title,
            'token_count': self.token_count,
            'metadata': self.metadata
        }


class SemanticChunker:
    """
    Advanced chunking with semantic awareness and context preservation.
    Handles tables, maintains context, and optimizes for retrieval.
    """
    
    def __init__(self):
        self.chunk_size = settings.chunk_size
        self.chunk_overlap = settings.chunk_overlap
        self.max_table_tokens = settings.max_table_tokens
        self.min_chunk_size = settings.min_chunk_size
        
        # Initialize tokenizer (OpenAI's tokenizer)
        try:
            self.encoding = tiktoken.encoding_for_model("gpt-4")
        except Exception:
            self.encoding = tiktoken.get_encoding("cl100k_base")
        
        logger.info(f"Chunker initialized: chunk_size={self.chunk_size}, overlap={self.chunk_overlap}")
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text using tiktoken."""
        return len(self.encoding.encode(text))
    
    def chunk_document(
        self, 
        doc_structure: DocumentStructure,
        document_title: Optional[str] = None
    ) -> List[Chunk]:
        """
        Chunk entire document with intelligent strategy selection.
        
        Args:
            doc_structure: Extracted document structure
            document_title: Title of document for context
            
        Returns:
            List of chunks ready for embedding
        """
        all_chunks = []
        chunk_index = 0
        current_section = None
        
        # Group consecutive elements for better context
        element_groups = self._group_elements(doc_structure.elements)
        
        for group in element_groups:
            group_type = group['type']
            elements = group['elements']
            
            if group_type == 'table':
                # Handle tables specially
                chunks = self._chunk_table_group(elements, chunk_index, current_section)
            elif group_type == 'title':
                # Update section context
                current_section = elements[0].content
                chunks = self._chunk_text_group(elements, chunk_index, current_section)
            else:
                # Regular text chunks
                chunks = self._chunk_text_group(elements, chunk_index, current_section)
            
            all_chunks.extend(chunks)
            chunk_index += len(chunks)
        
        # Add document-level summary chunk (if document has multiple chunks)
        if len(all_chunks) > 1:
            summary_chunk = self._create_document_summary(
                doc_structure, 
                document_title,
                chunk_index
            )
            all_chunks.insert(0, summary_chunk)
        
        logger.info(f"Created {len(all_chunks)} chunks from document")
        
        return all_chunks
    
    def _group_elements(self, elements: List[ExtractedElement]) -> List[Dict[str, Any]]:
        """
        Group consecutive elements by type for better chunking.
        
        Returns:
            List of element groups with type information
        """
        if not elements:
            return []
        
        groups = []
        current_group = {
            'type': elements[0].element_type,
            'elements': [elements[0]]
        }
        
        for elem in elements[1:]:
            # Group similar consecutive elements
            if elem.element_type == current_group['type']:
                current_group['elements'].append(elem)
            else:
                groups.append(current_group)
                current_group = {
                    'type': elem.element_type,
                    'elements': [elem]
                }
        
        # Add last group
        groups.append(current_group)
        
        return groups
    
    def _chunk_text_group(
        self, 
        elements: List[ExtractedElement],
        start_index: int,
        section_title: Optional[str]
    ) -> List[Chunk]:
        """
        Chunk a group of text elements with overlap.
        """
        # Combine elements into single text
        combined_text = "\n\n".join(e.content for e in elements)
        page_numbers = sorted(set(e.page_number for e in elements))
        
        # Split into sentences for semantic boundaries
        sentences = self._split_sentences(combined_text)
        
        chunks = []
        current_chunk = []
        current_tokens = 0
        
        for sentence in sentences:
            sentence_tokens = self.count_tokens(sentence)
            
            # If single sentence exceeds chunk size, split it
            if sentence_tokens > self.chunk_size:
                # Flush current chunk if any
                if current_chunk:
                    chunks.append(self._create_chunk(
                        " ".join(current_chunk),
                        "text",
                        start_index + len(chunks),
                        page_numbers,
                        section_title
                    ))
                    current_chunk = []
                    current_tokens = 0
                
                # Split long sentence by character limit
                long_chunks = self._split_long_text(sentence)
                for long_chunk in long_chunks:
                    chunks.append(self._create_chunk(
                        long_chunk,
                        "text",
                        start_index + len(chunks),
                        page_numbers,
                        section_title
                    ))
                continue
            
            # Check if adding sentence exceeds chunk size
            if current_tokens + sentence_tokens > self.chunk_size and current_chunk:
                # Create chunk
                chunks.append(self._create_chunk(
                    " ".join(current_chunk),
                    "text",
                    start_index + len(chunks),
                    page_numbers,
                    section_title
                ))
                
                # Start new chunk with overlap
                overlap_sentences = self._get_overlap_sentences(current_chunk)
                current_chunk = overlap_sentences + [sentence]
                current_tokens = sum(self.count_tokens(s) for s in current_chunk)
            else:
                current_chunk.append(sentence)
                current_tokens += sentence_tokens
        
        # Add remaining chunk
        if current_chunk:
            chunk_text = " ".join(current_chunk)
            if self.count_tokens(chunk_text) >= self.min_chunk_size:
                chunks.append(self._create_chunk(
                    chunk_text,
                    "text",
                    start_index + len(chunks),
                    page_numbers,
                    section_title
                ))
        
        return chunks
    
    def _chunk_table_group(
        self,
        elements: List[ExtractedElement],
        start_index: int,
        section_title: Optional[str]
    ) -> List[Chunk]:
        """
        Chunk tables with special handling.
        Preserves table structure and adds surrounding context.
        """
        chunks = []
        
        for elem in elements:
            table_content = elem.content
            table_tokens = self.count_tokens(table_content)
            
            # Add context prefix to table
            context_prefix = f"Table from page {elem.page_number}"
            if section_title:
                context_prefix += f" in section '{section_title}'"
            
            enhanced_content = f"{context_prefix}:\n\n{table_content}"
            
            # If table fits within limit, keep as single chunk
            if table_tokens <= self.max_table_tokens:
                chunks.append(self._create_chunk(
                    enhanced_content,
                    "table",
                    start_index + len(chunks),
                    [elem.page_number],
                    section_title
                ))
            else:
                # Split large table by rows
                table_chunks = self._split_table(table_content, context_prefix)
                for idx, table_chunk in enumerate(table_chunks):
                    chunks.append(self._create_chunk(
                        table_chunk,
                        "table",
                        start_index + len(chunks),
                        [elem.page_number],
                        section_title,
                        {'table_part': f"{idx+1}/{len(table_chunks)}"}
                    ))
        
        return chunks
    
    def _split_table(self, table_content: str, context: str) -> List[str]:
        """Split large table into smaller chunks while preserving headers."""
        lines = table_content.split('\n')
        
        # Detect header (usually first 1-3 lines)
        header_lines = []
        data_lines = []
        
        for idx, line in enumerate(lines):
            if idx < 3 and ('|' in line or '\t' in line):
                header_lines.append(line)
            else:
                data_lines.append(line)
        
        header = '\n'.join(header_lines)
        chunks = []
        current_chunk_lines = []
        
        for line in data_lines:
            test_chunk = f"{context}\n\n{header}\n" + '\n'.join(current_chunk_lines + [line])
            
            if self.count_tokens(test_chunk) > self.max_table_tokens and current_chunk_lines:
                # Finalize current chunk
                chunks.append(f"{context}\n\n{header}\n" + '\n'.join(current_chunk_lines))
                current_chunk_lines = [line]
            else:
                current_chunk_lines.append(line)
        
        # Add remaining
        if current_chunk_lines:
            chunks.append(f"{context}\n\n{header}\n" + '\n'.join(current_chunk_lines))
        
        return chunks if chunks else [f"{context}\n\n{table_content}"]
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences using simple heuristic."""
        import re
        
        # Split on sentence boundaries
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        # Filter empty sentences
        return [s.strip() for s in sentences if s.strip()]
    
    def _split_long_text(self, text: str) -> List[str]:
        """Split text that exceeds chunk size by character limit."""
        chunks = []
        words = text.split()
        current_chunk = []
        
        for word in words:
            current_chunk.append(word)
            chunk_text = " ".join(current_chunk)
            
            if self.count_tokens(chunk_text) >= self.chunk_size:
                chunks.append(chunk_text)
                current_chunk = []
        
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        return chunks
    
    def _get_overlap_sentences(self, sentences: List[str]) -> List[str]:
        """Get sentences for overlap based on token count."""
        overlap_sentences = []
        overlap_tokens = 0
        
        # Take sentences from end until we reach overlap size
        for sentence in reversed(sentences):
            sentence_tokens = self.count_tokens(sentence)
            if overlap_tokens + sentence_tokens <= self.chunk_overlap:
                overlap_sentences.insert(0, sentence)
                overlap_tokens += sentence_tokens
            else:
                break
        
        return overlap_sentences
    
    def _create_chunk(
        self,
        content: str,
        chunk_type: str,
        chunk_index: int,
        page_numbers: List[int],
        section_title: Optional[str],
        extra_metadata: Optional[Dict[str, Any]] = None
    ) -> Chunk:
        """Create a Chunk object with all metadata."""
        token_count = self.count_tokens(content)
        
        metadata = {
            'char_count': len(content),
            'has_numbers': any(c.isdigit() for c in content),
        }
        
        if extra_metadata:
            metadata.update(extra_metadata)
        
        return Chunk(
            content=content,
            chunk_type=chunk_type,
            chunk_index=chunk_index,
            page_numbers=page_numbers,
            section_title=section_title,
            token_count=token_count,
            metadata=metadata
        )
    
    def _create_document_summary(
        self,
        doc_structure: DocumentStructure,
        document_title: Optional[str],
        chunk_index: int
    ) -> Chunk:
        """Create a summary chunk for the entire document."""
        # Extract key information
        total_elements = len(doc_structure.elements)
        total_pages = doc_structure.total_pages
        
        # Get titles/headers for structure
        titles = [e.content for e in doc_structure.elements if e.element_type == 'title'][:5]
        
        summary = f"Document: {document_title or 'Unknown'}\n"
        summary += f"Total pages: {total_pages}\n"
        summary += f"Contains tables: {'Yes' if doc_structure.has_tables else 'No'}\n"
        
        if titles:
            summary += f"\nMain sections:\n" + "\n".join(f"- {t}" for t in titles)
        
        return Chunk(
            content=summary,
            chunk_type="summary",
            chunk_index=chunk_index,
            page_numbers=list(range(1, total_pages + 1)),
            section_title=None,
            token_count=self.count_tokens(summary),
            metadata={'is_document_summary': True}
        )


# Global instance
semantic_chunker = SemanticChunker()

__all__ = ['SemanticChunker', 'Chunk', 'semantic_chunker']