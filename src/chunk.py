"""
Text chunking module with intelligent segmentation and overlap.
"""

import re
from typing import List, Optional
import logging
from utils import Document, Chunk, log_step, estimate_tokens

logger = logging.getLogger(__name__)


class TextChunker:
    """Intelligently chunk text into retrieval-friendly segments."""
    
    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 128,
        min_chunk_size: int = 100
    ):
        """
        Initialize text chunker.
        
        Args:
            chunk_size: Target chunk size in characters
            chunk_overlap: Overlap between chunks in characters
            min_chunk_size: Minimum size for a chunk (to avoid tiny fragments)
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        log_step("CHUNK", f"Initialized chunker (size={chunk_size}, overlap={chunk_overlap})")
    
    def chunk(self, document: Document) -> List[Chunk]:
        """
        Chunk a document into segments.
        
        Args:
            document: Document to chunk
        
        Returns:
            List of Chunk objects
        """
        # Use smart splitting based on document type
        if document.doc_type == 'pdf':
            segments = self._split_by_sections(document.content)
        elif document.doc_type == 'md':
            segments = self._split_by_markdown_headers(document.content)
        else:
            segments = self._split_by_paragraphs(document.content)
        
        chunks = []
        chunk_index = 0
        
        for segment in segments:
            # Further chunk segments that are too large
            if len(segment) > self.chunk_size:
                sub_chunks = self._sliding_window_chunk(segment)
                for sub_chunk in sub_chunks:
                    chunk = Chunk(
                        content=sub_chunk,
                        source=document.source,
                        chunk_index=chunk_index,
                        doc_type=document.doc_type,
                        metadata={
                            **document.metadata,
                            'segment_type': 'subsplit',
                            'tokens_estimate': estimate_tokens(sub_chunk)
                        }
                    )
                    chunks.append(chunk)
                    chunk_index += 1
            else:
                if len(segment) >= self.min_chunk_size:
                    chunk = Chunk(
                        content=segment,
                        source=document.source,
                        chunk_index=chunk_index,
                        doc_type=document.doc_type,
                        metadata={
                            **document.metadata,
                            'segment_type': 'natural',
                            'tokens_estimate': estimate_tokens(segment)
                        }
                    )
                    chunks.append(chunk)
                    chunk_index += 1
        
        log_step("CHUNK", f"Created {len(chunks)} chunks from {document.source}")
        return chunks
    
    def _split_by_markdown_headers(self, text: str) -> List[str]:
        """Split markdown by headers, preserving header context."""
        segments = []
        current_segment = []
        
        # Split by markdown headers (# ## ### etc)
        lines = text.split('\n')
        
        for line in lines:
            if re.match(r'^#{1,6}\s', line):  # Markdown header
                if current_segment:
                    segments.append('\n'.join(current_segment))
                    current_segment = [line]
                else:
                    current_segment = [line]
            else:
                current_segment.append(line)
        
        if current_segment:
            segments.append('\n'.join(current_segment))
        
        return [s.strip() for s in segments if s.strip()]
    
    def _split_by_sections(self, text: str) -> List[str]:
        """Split text by section breaks (multiple newlines, page breaks)."""
        # Split by double+ newlines
        segments = re.split(r'\n{2,}|\[Page \d+\]', text)
        return [s.strip() for s in segments if s.strip()]
    
    def _split_by_paragraphs(self, text: str) -> List[str]:
        """Split text by paragraphs."""
        segments = text.split('\n\n')
        return [s.strip() for s in segments if s.strip()]
    
    def _sliding_window_chunk(self, text: str) -> List[str]:
        """
        Apply sliding window chunking for large segments.
        
        Args:
            text: Text to chunk
        
        Returns:
            List of overlapping chunks
        """
        chunks = []
        start = 0
        
        while start < len(text):
            # Find the end position
            end = start + self.chunk_size
            
            # Try to end at sentence boundary if possible
            if end < len(text):
                # Look for nearest period + space within 50 chars of end
                search_start = max(start, end - 50)
                sentence_end = text.rfind('.', search_start, end)
                
                if sentence_end > start:
                    end = sentence_end + 1
                else:
                    # Otherwise try to break at word boundary
                    space_end = text.rfind(' ', start, end)
                    if space_end > start:
                        end = space_end
            
            chunk = text[start:end].strip()
            if len(chunk) >= self.min_chunk_size:
                chunks.append(chunk)
            
            # Move start position with overlap
            start = end - self.chunk_overlap
        
        return chunks
    
    def chunk_batch(self, documents: List[Document]) -> List[Chunk]:
        """
        Chunk multiple documents.
        
        Args:
            documents: List of documents to chunk
        
        Returns:
            List of all chunks
        """
        all_chunks = []
        for doc in documents:
            chunks = self.chunk(doc)
            all_chunks.extend(chunks)
        
        log_step("CHUNK", f"Total chunks created: {len(all_chunks)}")
        return all_chunks
