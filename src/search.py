"""
Hybrid search module combining lexical and semantic search.
"""

from typing import List, Dict, Any, Optional
import logging
import numpy as np
from utils import SearchResult, log_step

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    BM25Okapi = None

try:
    from caption import get_caption_generator, CaptionGenerator
    CAPTION_AVAILABLE = True
except ImportError:
    CAPTION_AVAILABLE = False

logger = logging.getLogger(__name__)


class HybridSearchEngine:
    """Combine lexical (BM25) and semantic (embeddings) search with optional captions."""
    
    def __init__(
        self,
        semantic_weight: float = 0.6,
        lexical_weight: float = 0.4,
        enable_captions: bool = True,
        caption_provider: str = "mock"
    ):
        """
        Initialize search engine.
        
        Args:
            semantic_weight: Weight for semantic similarity (0-1)
            lexical_weight: Weight for lexical ranking (0-1)
            enable_captions: Whether to generate semantic captions
            caption_provider: "mock" (default), "azure", or "openai"
                - "mock": Use mock captions (no API key needed)
                - "azure": Use Azure OpenAI (requires AZURE_OPENAI_KEY env var)
                - "openai": Use OpenAI (requires OPENAI_API_KEY env var)
        """
        self.semantic_weight = semantic_weight
        self.lexical_weight = lexical_weight
        self.bm25_model = None
        self.tokenized_corpus = None
        self.enable_captions = enable_captions and CAPTION_AVAILABLE
        self.caption_generator: Optional[CaptionGenerator] = None
        
        if self.enable_captions:
            try:
                self.caption_generator = get_caption_generator(caption_provider)
                log_step("SEARCH", f"Enabled semantic captions via {caption_provider}")
            except Exception as e:
                logger.warning(f"Failed to initialize caption generator: {e}. Disabling captions.")
                self.enable_captions = False
        
        log_step("SEARCH", f"Initialized HybridSearchEngine (semantic={semantic_weight}, lexical={lexical_weight}, captions={self.enable_captions})")
    
    def build_lexical_index(self, chunks_metadata: List[Dict[str, Any]]):
        """
        Build BM25 index for lexical search.
        
        Args:
            chunks_metadata: List of chunk metadata dictionaries
        """
        if BM25Okapi is None:
            raise ImportError("rank-bm25 not installed")
        
        # Tokenize documents
        self.tokenized_corpus = [
            self._tokenize(chunk['content'])
            for chunk in chunks_metadata
        ]
        
        # Build BM25 model
        self.bm25_model = BM25Okapi(self.tokenized_corpus)
        
        log_step("SEARCH", f"Built BM25 index for {len(self.tokenized_corpus)} documents")
    
    def search(
        self,
        query: str,
        query_embedding: np.ndarray,
        index,
        top_k: int = 10
    ) -> List[SearchResult]:
        """
        Perform hybrid search.
        
        Args:
            query: Query string
            query_embedding: Query embedding vector
            index: VectorIndex instance
            top_k: Number of top results to return
        
        Returns:
            List of SearchResult objects ranked by combined score
        """
        if index.metadata and not self.bm25_model:
            self.build_lexical_index(index.metadata)
        
        # Semantic search results
        semantic_results = index.search_semantic(query_embedding, top_k=top_k * 2)
        
        # Lexical search results
        if self.bm25_model:
            tokenized_query = self._tokenize(query)
            bm25_scores = self.bm25_model.get_scores(tokenized_query)
            lexical_results = [
                (idx, float(score))
                for idx, score in enumerate(bm25_scores)
                if score > 0
            ]
            lexical_results = sorted(lexical_results, key=lambda x: x[1], reverse=True)[:top_k * 2]
        else:
            lexical_results = []
        
        # Combine and normalize scores
        combined_scores = {}
        
        # Add semantic scores (already 0-1 from cosine similarity)
        for idx, score in semantic_results:
            combined_scores[idx] = {
                'semantic': score,
                'lexical': 0.0
            }
        
        # Add lexical scores (normalize BM25)
        if lexical_results:
            max_lexical_score = max([score for _, score in lexical_results])
            for idx, score in lexical_results:
                if max_lexical_score > 0:
                    normalized_score = score / max_lexical_score
                else:
                    normalized_score = 0.0
                
                if idx not in combined_scores:
                    combined_scores[idx] = {'semantic': 0.0, 'lexical': normalized_score}
                else:
                    combined_scores[idx]['lexical'] = normalized_score
        
        # Calculate combined scores and create results
        results = []
        for idx, scores in combined_scores.items():
            combined_score = (
                self.semantic_weight * scores['semantic'] +
                self.lexical_weight * scores['lexical']
            )
            
            chunk_info = index.get_by_index(idx)
            if chunk_info:
                # Generate semantic caption if enabled
                caption = self._generate_caption(chunk_info['content'])
                
                result = SearchResult(
                    chunk_id=chunk_info['chunk_id'],
                    content=chunk_info['content'],
                    source=chunk_info['source'],
                    score=combined_score,
                    similarity_score=scores['semantic'],
                    ranking_score=scores['lexical'],
                    metadata={
                        'doc_type': chunk_info['doc_type'],
                        'chunk_index': chunk_info['chunk_index'],
                        **chunk_info['metadata']
                    },
                    semantic_caption=caption
                )
                results.append(result)
        
        # Sort by combined score
        results = sorted(results, key=lambda x: x.score, reverse=True)[:top_k]
        
        log_step("SEARCH", f"Found {len(results)} hybrid search results")
        return results
    
    def semantic_search(
        self,
        query_embedding: np.ndarray,
        index,
        top_k: int = 10
    ) -> List[SearchResult]:
        """
        Pure semantic search.
        
        Args:
            query_embedding: Query embedding
            index: VectorIndex instance
            top_k: Number of results
        
        Returns:
            List of SearchResult objects
        """
        results_raw = index.search_semantic(query_embedding, top_k=top_k)
        
        results = []
        for idx, similarity in results_raw:
            chunk_info = index.get_by_index(idx)
            if chunk_info:
                # Generate semantic caption if enabled
                caption = self._generate_caption(chunk_info['content'])
                
                result = SearchResult(
                    chunk_id=chunk_info['chunk_id'],
                    content=chunk_info['content'],
                    source=chunk_info['source'],
                    score=similarity,
                    similarity_score=similarity,
                    ranking_score=0.0,
                    metadata={
                        'doc_type': chunk_info['doc_type'],
                        'chunk_index': chunk_info['chunk_index'],
                        **chunk_info['metadata']
                    },
                    semantic_caption=caption
                )
                results.append(result)
        
        return results
    
    def lexical_search(
        self,
        query: str,
        index,
        top_k: int = 10
    ) -> List[SearchResult]:
        """
        Pure lexical search using BM25.
        
        Args:
            query: Query string
            index: VectorIndex instance
            top_k: Number of results
        
        Returns:
            List of SearchResult objects
        """
        if not self.bm25_model:
            self.build_lexical_index(index.metadata)
        
        tokenized_query = self._tokenize(query)
        scores = self.bm25_model.get_scores(tokenized_query)
        
        # Get top-k
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                chunk_info = index.get_by_index(idx)
                if chunk_info:
                    # Generate semantic caption if enabled
                    caption = self._generate_caption(chunk_info['content'])
                    
                    result = SearchResult(
                        chunk_id=chunk_info['chunk_id'],
                        content=chunk_info['content'],
                        source=chunk_info['source'],
                        score=float(scores[idx]),
                        similarity_score=0.0,
                        ranking_score=float(scores[idx]),
                        metadata={
                            'doc_type': chunk_info['doc_type'],
                            'chunk_index': chunk_info['chunk_index'],
                            **chunk_info['metadata']
                        },
                        semantic_caption=caption
                    )
                    results.append(result)
        
        return results
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization."""
        return text.lower().split()
    
    def _generate_caption(self, content: str) -> Optional[str]:
        """
        Generate semantic caption for content.
        
        Args:
            content: The chunk content to caption
        
        Returns:
            Semantic caption or None if captions disabled
        """
        if not self.enable_captions or not self.caption_generator:
            return None
        
        try:
            return self.caption_generator.generate_caption(content, max_length=150)
        except Exception as e:
            logger.warning(f"Failed to generate caption: {e}")
            return None
