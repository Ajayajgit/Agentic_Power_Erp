"""
ingestion/retriever.py

Retrieval logic: embed a query, search the vector store scoped to one
tenant's namespace only, apply a minimum similarity threshold. Mirrors
Section 1.3 of the design document - hybrid retrieval (vector similarity +
tenant filter), and "flag low evidence" rather than force a weak match.
"""

from typing import List

from prototype.models import RetrievedChunk
from prototype.providers.embedding_provider import ResilientEmbeddingProvider
from prototype.vectorstore.base import VectorStore


class HistoricalRetriever:
    def __init__(self, embedder: ResilientEmbeddingProvider, vector_store: VectorStore,
                 min_score_threshold: float = 0.15):
        self._embedder = embedder
        self._vector_store = vector_store
        self._min_score_threshold = min_score_threshold

    def retrieve(self, query_text: str, tenant_id: str, top_k: int) -> List[RetrievedChunk]:
        query_vector = self._embedder.embed([query_text])[0]
        raw_matches = self._vector_store.query(vector=query_vector, top_k=top_k, namespace=tenant_id)

        results = []
        for score, metadata in raw_matches:
            if score >= self._min_score_threshold:
                results.append(RetrievedChunk(
                    text=metadata.get("text", ""),
                    source=metadata.get("source", "unknown"),
                    content_type=metadata.get("content_type", "unknown"),
                    score=score,
                ))
        return results
