"""
ingestion/pipeline.py

Orchestrates ingestion: chunk -> embed -> upsert, tagging every chunk with
tenant_id / engagement_id / content_type metadata. This tagging is what
makes tenant-scoped retrieval possible later (Section 2 of the design doc).
"""

import uuid
from typing import List

from prototype.ingestion.chunker import TextChunker
from prototype.providers.embedding_provider import ResilientEmbeddingProvider
from prototype.vectorstore.base import VectorStore


class IngestionPipeline:
    def __init__(self, chunker: TextChunker, embedder: ResilientEmbeddingProvider, vector_store: VectorStore):
        self._chunker = chunker
        self._embedder = embedder
        self._vector_store = vector_store

    def ingest_document(self, text: str, tenant_id: str, engagement_id: str,
                         content_type: str, source: str) -> int:
        """Chunk, embed, and store one document. Returns number of chunks stored."""
        chunks = self._chunker.chunk(text)
        if not chunks:
            return 0

        vectors = self._embedder.embed(chunks)
        ids = [f"{source}-{uuid.uuid4().hex[:8]}-{i}" for i in range(len(chunks))]
        metadatas = [
            {
                "text": chunk,
                "source": source,
                "content_type": content_type,
                "tenant_id": tenant_id,
                "engagement_id": engagement_id,
            }
            for chunk in chunks
        ]

        self._vector_store.upsert(ids=ids, vectors=vectors, metadatas=metadatas, namespace=tenant_id)
        return len(chunks)
