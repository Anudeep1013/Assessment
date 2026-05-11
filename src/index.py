"""
Vector index management and storage.
"""

import pickle
import os
from typing import List, Dict, Any, Optional
import numpy as np
import logging
from utils import EmbeddedChunk, log_step

logger = logging.getLogger(__name__)


class VectorIndex:
    """In-memory vector index with persistence."""
    
    def __init__(self):
        self.embeddings: np.ndarray = np.array([]).reshape(0, 0)  # (n_chunks, embedding_dim)
        self.metadata: List[Dict[str, Any]] = []  # Parallel list with metadata
        self.chunk_ids: List[str] = []  # Chunk IDs for reference
        self.embedding_dim = 0
        
        log_step("INDEX", "Initialized VectorIndex")
    
    def add_chunks(self, embedded_chunks: List[EmbeddedChunk]):
        """
        Add embedded chunks to index.
        
        Args:
            embedded_chunks: List of EmbeddedChunk objects
        """
        if not embedded_chunks:
            return
        
        # Convert embeddings to numpy array
        new_embeddings = np.array([chunk.embedding for chunk in embedded_chunks])
        
        # Initialize embeddings if first addition
        if self.embeddings.size == 0:
            self.embeddings = new_embeddings
            self.embedding_dim = new_embeddings.shape[1]
        else:
            # Verify dimension consistency
            if new_embeddings.shape[1] != self.embedding_dim:
                raise ValueError(
                    f"Embedding dimension mismatch: {new_embeddings.shape[1]} vs {self.embedding_dim}"
                )
            self.embeddings = np.vstack([self.embeddings, new_embeddings])
        
        # Add metadata
        for chunk in embedded_chunks:
            self.metadata.append({
                'chunk_id': chunk.chunk_id,
                'source': chunk.source,
                'doc_type': chunk.doc_type,
                'content': chunk.content,
                'chunk_index': chunk.chunk_index,
                'metadata': chunk.metadata
            })
            self.chunk_ids.append(chunk.chunk_id)
        
        log_step("INDEX", f"Added {len(embedded_chunks)} chunks. Total: {len(self.chunk_ids)}")
    
    def search_semantic(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10
    ) -> List[tuple]:
        """
        Semantic search using cosine similarity.
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of top results
        
        Returns:
            List of (index, similarity_score) tuples sorted by score
        """
        if self.embeddings.size == 0:
            return []
        
        # Cosine similarity
        similarities = np.dot(self.embeddings, query_embedding) / (
            np.linalg.norm(self.embeddings, axis=1) * 
            np.linalg.norm(query_embedding) + 1e-8
        )
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        return [(idx, float(similarities[idx])) for idx in top_indices]
    
    def get_chunk_by_id(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        """Get chunk metadata by ID."""
        try:
            idx = self.chunk_ids.index(chunk_id)
            return self.metadata[idx]
        except (ValueError, IndexError):
            return None
    
    def get_by_index(self, idx: int) -> Optional[Dict[str, Any]]:
        """Get chunk by index."""
        if 0 <= idx < len(self.metadata):
            return self.metadata[idx]
        return None
    
    def get_stats(self) -> Dict[str, Any]:
        """Get index statistics."""
        return {
            'total_chunks': len(self.chunk_ids),
            'embedding_dimension': self.embedding_dim,
            'unique_sources': len(set(m['source'] for m in self.metadata)),
            'doc_types': list(set(m['doc_type'] for m in self.metadata))
        }
    
    def save(self, filepath: str):
        """
        Save index to disk.
        
        Args:
            filepath: Path to save index file
        """
        index_data = {
            'embeddings': self.embeddings,
            'metadata': self.metadata,
            'chunk_ids': self.chunk_ids,
            'embedding_dim': self.embedding_dim
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(index_data, f)
        
        log_step("INDEX", f"Saved index to {filepath}")
    
    def load(self, filepath: str):
        """
        Load index from disk.
        
        Args:
            filepath: Path to index file
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Index file not found: {filepath}")
        
        with open(filepath, 'rb') as f:
            index_data = pickle.load(f)
        
        self.embeddings = index_data['embeddings']
        self.metadata = index_data['metadata']
        self.chunk_ids = index_data['chunk_ids']
        self.embedding_dim = index_data['embedding_dim']
        
        log_step("INDEX", f"Loaded index from {filepath}")


class IndexBuilder:
    """Helper class to build index from chunks."""
    
    def __init__(self):
        self.index = VectorIndex()
    
    def build(
        self,
        embedded_chunks: List[EmbeddedChunk],
        save_path: Optional[str] = None
    ) -> VectorIndex:
        """
        Build index from embedded chunks.
        
        Args:
            embedded_chunks: List of embedded chunks
            save_path: Optional path to save index
        
        Returns:
            Built VectorIndex
        """
        self.index.add_chunks(embedded_chunks)
        
        if save_path:
            self.index.save(save_path)
        
        return self.index
