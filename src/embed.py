"""
Embedding generation module using sentence-transformers.
"""

from typing import List
import logging
import numpy as np
from utils import Chunk, EmbeddedChunk, log_step

logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """Generate embeddings for text chunks."""
    
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "cpu"
    ):
        """
        Initialize embedding generator.
        
        Args:
            model_name: Sentence-transformers model identifier
            device: "cpu" or "cuda"
        """
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError("sentence-transformers not installed")
        
        self.model = SentenceTransformer(model_name, device=device)
        self.model_name = model_name
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        
        log_step("EMBED", f"Loaded {model_name} (dim={self.embedding_dim})")
    
    def embed(self, chunk: Chunk) -> EmbeddedChunk:
        """
        Generate embedding for a chunk.
        
        Args:
            chunk: Chunk to embed
        
        Returns:
            EmbeddedChunk with embedding vector
        """
        embedding = self.model.encode(
            chunk.content,
            convert_to_tensor=False,
            show_progress_bar=False
        )
        
        # Normalize embedding
        embedding = embedding / (np.linalg.norm(embedding) + 1e-8)
        
        return EmbeddedChunk(
            content=chunk.content,
            source=chunk.source,
            chunk_index=chunk.chunk_index,
            doc_type=chunk.doc_type,
            embedding=embedding.tolist(),
            metadata=chunk.metadata
        )
    
    def embed_batch(
        self,
        chunks: List[Chunk],
        batch_size: int = 32
    ) -> List[EmbeddedChunk]:
        """
        Generate embeddings for multiple chunks efficiently.
        
        Args:
            chunks: List of chunks to embed
            batch_size: Batch size for encoding
        
        Returns:
            List of EmbeddedChunk objects
        """
        if not chunks:
            return []
        
        # Extract texts
        texts = [chunk.content for chunk in chunks]
        
        # Encode in batches
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            convert_to_tensor=False,
            show_progress_bar=True
        )
        
        # Normalize embeddings
        embeddings = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
        
        # Create embedded chunks
        embedded_chunks = []
        for chunk, embedding in zip(chunks, embeddings):
            embedded_chunk = EmbeddedChunk(
                content=chunk.content,
                source=chunk.source,
                chunk_index=chunk.chunk_index,
                doc_type=chunk.doc_type,
                embedding=embedding.tolist(),
                metadata=chunk.metadata
            )
            embedded_chunks.append(embedded_chunk)
        
        log_step("EMBED", f"Generated embeddings for {len(embedded_chunks)} chunks")
        return embedded_chunks
    
    def embed_query(self, query: str) -> np.ndarray:
        """
        Generate embedding for a query.
        
        Args:
            query: Query string
        
        Returns:
            Normalized embedding vector
        """
        embedding = self.model.encode(
            query,
            convert_to_tensor=False,
            show_progress_bar=False
        )
        
        # Normalize
        embedding = embedding / (np.linalg.norm(embedding) + 1e-8)
        return embedding


class BatchEmbedder:
    """Utility for embedding large numbers of chunks efficiently."""
    
    def __init__(self, generator: EmbeddingGenerator):
        self.generator = generator
    
    def process_chunks(
        self,
        chunks: List[Chunk],
        batch_size: int = 32,
        save_progress: bool = True
    ) -> List[EmbeddedChunk]:
        """
        Process chunks with progress tracking.
        
        Args:
            chunks: Chunks to embed
            batch_size: Processing batch size
            save_progress: Whether to log progress
        
        Returns:
            List of embedded chunks
        """
        embedded = self.generator.embed_batch(chunks, batch_size)
        
        if save_progress:
            log_step("EMBED", f"Completed embedding batch: {len(embedded)} chunks")
        
        return embedded
