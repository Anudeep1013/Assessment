"""
Orchestration module for the entire ETL pipeline.
"""

import os
from typing import List, Optional, Dict, Any
import logging
from utils import log_step, log_error
from extract import DocumentExtractor, AzureBlobExtractor
from chunk import TextChunker
from embed import EmbeddingGenerator, BatchEmbedder
from index import IndexBuilder
from search import HybridSearchEngine

logger = logging.getLogger(__name__)


class ETLPipeline:
    """End-to-end ETL pipeline orchestrator."""
    
    def __init__(
        self,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        chunk_size: int = 512,
        chunk_overlap: int = 128,
        enable_captions: bool = True,
        caption_provider: str = "mock"
    ):
        """
        Initialize ETL pipeline.
        
        Args:
            embedding_model: Model name for embeddings
            chunk_size: Target chunk size
            chunk_overlap: Overlap between chunks
            enable_captions: Whether to generate semantic captions
            caption_provider: Caption provider ("mock", "azure", or "openai")
        """
        self.extractor = DocumentExtractor()
        self.chunker = TextChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.embedding_generator = EmbeddingGenerator(model_name=embedding_model)
        self.batch_embedder = BatchEmbedder(self.embedding_generator)
        self.index_builder = IndexBuilder()
        self.search_engine = HybridSearchEngine(
            semantic_weight=0.6,
            lexical_weight=0.4,
            enable_captions=enable_captions,
            caption_provider=caption_provider
        )
        self.index = None
        
        log_step("PIPELINE", "Initialized ETLPipeline")
    
    def ingest_from_files(
        self,
        file_list: List[tuple],
        save_index_path: Optional[str] = None,
        batch_size: int = 32
    ):
        """
        Ingest documents from local files.
        
        Args:
            file_list: List of (filepath, source_path) tuples
            save_index_path: Optional path to save built index
            batch_size: Batch size for embeddings
        """
        log_step("PIPELINE", "Starting ingestion from files")
        
        # Extract
        documents = self.extractor.extract_batch(file_list)
        if not documents:
            log_error("PIPELINE", "No documents extracted")
            return False
        
        # Chunk
        chunks = self.chunker.chunk_batch(documents)
        if not chunks:
            log_error("PIPELINE", "No chunks created")
            return False
        
        # Embed
        embedded_chunks = self.batch_embedder.process_chunks(chunks, batch_size)
        if not embedded_chunks:
            log_error("PIPELINE", "No embeddings generated")
            return False
        
        # Index
        self.index = self.index_builder.build(embedded_chunks, save_index_path)
        
        # Setup search engine
        self.search_engine.build_lexical_index(self.index.metadata)
        
        log_step("PIPELINE", "Ingestion completed successfully")
        return True
    
    def ingest_from_azure(
        self,
        connection_string: str,
        container_name: str,
        save_index_path: Optional[str] = None,
        batch_size: int = 32
    ):
        """
        Ingest documents from Azure Blob Storage.
        
        Args:
            connection_string: Azure Storage connection string
            container_name: Blob container name
            save_index_path: Optional path to save index
            batch_size: Batch size for embeddings
        """
        log_step("PIPELINE", "Starting ingestion from Azure Blob")
        
        try:
            azure_extractor = AzureBlobExtractor(connection_string)
            
            # Extract from Azure
            documents = azure_extractor.extract_from_container(container_name)
            if not documents:
                log_error("PIPELINE", "No documents extracted from Azure")
                return False
            
            # Chunk
            chunks = self.chunker.chunk_batch(documents)
            
            # Embed
            embedded_chunks = self.batch_embedder.process_chunks(chunks, batch_size)
            
            # Index
            self.index = self.index_builder.build(embedded_chunks, save_index_path)
            
            # Setup search
            self.search_engine.build_lexical_index(self.index.metadata)
            
            log_step("PIPELINE", "Azure ingestion completed successfully")
            return True
        
        except Exception as e:
            log_error("PIPELINE", f"Azure ingestion failed: {str(e)}")
            return False
    
    def load_index(self, index_path: str):
        """
        Load pre-built index.
        
        Args:
            index_path: Path to saved index
        """
        from index import VectorIndex
        
        self.index = VectorIndex()
        self.index.load(index_path)
        self.search_engine.build_lexical_index(self.index.metadata)
        
        log_step("PIPELINE", f"Loaded index from {index_path}")
    
    def search(
        self,
        query: str,
        top_k: int = 5,
        search_type: str = "hybrid"
    ):
        """
        Search the indexed documents.
        
        Args:
            query: Query string
            top_k: Number of results
            search_type: "hybrid", "semantic", or "lexical"
        
        Returns:
            List of SearchResult objects
        """
        if self.index is None:
            log_error("PIPELINE", "Index not built. Run ingest first.")
            return []
        
        # Generate query embedding
        query_embedding = self.embedding_generator.embed_query(query)
        
        # Perform search
        if search_type == "hybrid":
            results = self.search_engine.search(query, query_embedding, self.index, top_k)
        elif search_type == "semantic":
            results = self.search_engine.semantic_search(query_embedding, self.index, top_k)
        elif search_type == "lexical":
            results = self.search_engine.lexical_search(query, self.index, top_k)
        else:
            log_error("PIPELINE", f"Unknown search type: {search_type}")
            return []
        
        log_step("PIPELINE", f"Search returned {len(results)} results")
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics."""
        if self.index is None:
            return {"status": "Index not built"}
        
        return {
            "status": "Ready",
            "index_stats": self.index.get_stats(),
            "embedding_model": self.embedding_generator.model_name,
            "chunk_size": self.chunker.chunk_size,
            "chunk_overlap": self.chunker.chunk_overlap
        }


class PipelineConfig:
    """Configuration for pipeline defaults."""
    
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    CHUNK_SIZE = 512
    CHUNK_OVERLAP = 128
    BATCH_SIZE = 32
    TOP_K_DEFAULT = 5
    SEMANTIC_WEIGHT = 0.6
    LEXICAL_WEIGHT = 0.4
