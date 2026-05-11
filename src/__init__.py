"""
Azure ETL & Retrieval Augmented Search Pipeline

A complete end-to-end RAG pipeline for building retrieval augmented search systems
on documents stored in Azure Blob Storage. Supports mixed formats (PDF, Markdown, TXT)
with hybrid semantic and lexical search capabilities.

Main Components:
    - extract: Document extraction from multiple formats
    - chunk: Intelligent text chunking with overlap
    - embed: Semantic embedding generation
    - index: Vector and lexical index management
    - search: Hybrid search combining semantic + lexical
    - caption: Semantic caption generation (with mock and real LLM support)
    - ingest: End-to-end pipeline orchestration
    - utils: Utilities and data classes

Example:
    >>> from ingest import ETLPipeline
    >>> # Use mock captions (no API key needed)
    >>> pipeline = ETLPipeline(enable_captions=True, caption_provider="mock")
    >>> pipeline.ingest_from_files([('doc.txt', 'docs/doc.txt')])
    >>> results = pipeline.search("query", top_k=5)
    >>> for result in results:
    ...     print(f"Caption: {result.semantic_caption}")
"""

__version__ = "1.0.0"
__author__ = "RAG Team"

from .utils import Document, Chunk, EmbeddedChunk, SearchResult
from .extract import DocumentExtractor, AzureBlobExtractor
from .chunk import TextChunker
from .embed import EmbeddingGenerator, BatchEmbedder
from .index import VectorIndex, IndexBuilder
from .search import HybridSearchEngine
from .caption import CaptionGenerator, MockLLMCaption, AzureOpenAICaption, OpenAICaption, get_caption_generator
from .ingest import ETLPipeline, PipelineConfig

__all__ = [
    "Document",
    "Chunk",
    "EmbeddedChunk",
    "SearchResult",
    "DocumentExtractor",
    "AzureBlobExtractor",
    "TextChunker",
    "EmbeddingGenerator",
    "BatchEmbedder",
    "VectorIndex",
    "IndexBuilder",
    "HybridSearchEngine",
    "CaptionGenerator",
    "MockLLMCaption",
    "AzureOpenAICaption",
    "OpenAICaption",
    "get_caption_generator",
    "ETLPipeline",
    "PipelineConfig",
]
