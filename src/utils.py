"""
Utility functions for the RAG pipeline.
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


class Document:
    """Represents a raw document from storage."""
    
    def __init__(
        self,
        content: str,
        source: str,
        doc_type: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.content = content
        self.source = source  # e.g., "manuals/deviceA.pdf"
        self.doc_type = doc_type  # "pdf", "markdown", "txt"
        self.metadata = metadata or {}
        self.extraction_timestamp = datetime.utcnow().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'content': self.content,
            'source': self.source,
            'doc_type': self.doc_type,
            'metadata': self.metadata,
            'extraction_timestamp': self.extraction_timestamp
        }


class Chunk:
    """Represents a text chunk with metadata."""
    
    def __init__(
        self,
        content: str,
        source: str,
        chunk_index: int,
        doc_type: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.content = content
        self.source = source
        self.chunk_index = chunk_index
        self.doc_type = doc_type
        self.metadata = metadata or {}
        self.chunk_id = f"{source}_chunk_{chunk_index}"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'chunk_id': self.chunk_id,
            'content': self.content,
            'source': self.source,
            'chunk_index': self.chunk_index,
            'doc_type': self.doc_type,
            'metadata': self.metadata
        }


class EmbeddedChunk(Chunk):
    """Chunk with embedding vector."""
    
    def __init__(
        self,
        content: str,
        source: str,
        chunk_index: int,
        doc_type: str,
        embedding: List[float],
        metadata: Optional[Dict[str, Any]] = None
    ):
        super().__init__(content, source, chunk_index, doc_type, metadata)
        self.embedding = embedding
    
    def to_dict(self) -> Dict[str, Any]:
        base_dict = super().to_dict()
        base_dict['embedding'] = self.embedding
        return base_dict


class SearchResult:
    """Represents a search result with optional semantic caption."""
    
    def __init__(
        self,
        chunk_id: str,
        content: str,
        source: str,
        score: float,
        similarity_score: float,
        ranking_score: float,
        metadata: Optional[Dict[str, Any]] = None,
        semantic_caption: Optional[str] = None
    ):
        self.chunk_id = chunk_id
        self.content = content
        self.source = source
        self.score = score  # Combined score
        self.similarity_score = similarity_score  # Semantic
        self.ranking_score = ranking_score  # Lexical
        self.metadata = metadata or {}
        self.semantic_caption = semantic_caption  # AI-generated summary
    
    def to_dict(self) -> Dict[str, Any]:
        result = {
            'chunk_id': self.chunk_id,
            'content': self.content,
            'source': self.source,
            'combined_score': round(self.score, 4),
            'semantic_score': round(self.similarity_score, 4),
            'lexical_score': round(self.ranking_score, 4),
            'metadata': self.metadata
        }
        
        # Include semantic caption if available
        if self.semantic_caption:
            result['semantic_caption'] = self.semantic_caption
        
        return result
    
    def __repr__(self) -> str:
        return f"SearchResult(source={self.source}, score={self.score:.4f})"


def log_step(step: str, message: str):
    """Log a pipeline step."""
    logger.info(f"[{step}] {message}")


def log_error(step: str, message: str):
    """Log an error."""
    logger.error(f"[{step}] {message}")


def save_json(data: Any, filepath: str):
    """Save data to JSON file."""
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    log_step("IO", f"Saved to {filepath}")


def load_json(filepath: str) -> Any:
    """Load data from JSON file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def estimate_tokens(text: str) -> int:
    """Rough estimate of token count (1 token ≈ 4 chars)."""
    return len(text) // 4


def truncate_text(text: str, max_length: int = 500) -> str:
    """Truncate text to max length."""
    if len(text) <= max_length:
        return text
    return text[:max_length] + "..."
