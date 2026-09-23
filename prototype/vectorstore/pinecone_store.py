"""
vectorstore/pinecone_store.py

Pinecone implementation of VectorStore. Uses one Pinecone namespace per
tenant - this mirrors the multi-tenancy isolation design in the main
design document (Section 2): namespaces are a hard boundary enforced by
Pinecone itself, not just application-level filtering.
"""

import logging
from typing import Dict, List, Tuple

from prototype.vectorstore.base import VectorStore

logger = logging.getLogger(__name__)


class PineconeVectorStore(VectorStore):
    name = "pinecone"

    def __init__(self, api_key: str, index_name: str, cloud: str, region: str, dimension: int):
        from pinecone import Pinecone, ServerlessSpec  # lazy import

        self._pc = Pinecone(api_key=api_key)
        existing = [idx["name"] for idx in self._pc.list_indexes()]
        if index_name not in existing:
            logger.info("Creating Pinecone index '%s' (dim=%d)...", index_name, dimension)
            self._pc.create_index(
                name=index_name,
                dimension=dimension,
                metric="cosine",
                spec=ServerlessSpec(cloud=cloud, region=region),
            )
        self._index = self._pc.Index(index_name)

    def upsert(self, ids: List[str], vectors: List[List[float]], metadatas: List[Dict], namespace: str) -> None:
        records = [
            {"id": i, "values": v, "metadata": m}
            for i, v, m in zip(ids, vectors, metadatas)
        ]
        self._index.upsert(vectors=records, namespace=namespace)

    def query(self, vector: List[float], top_k: int, namespace: str) -> List[Tuple[float, Dict]]:
        result = self._index.query(
            vector=vector, top_k=top_k, namespace=namespace, include_metadata=True
        )
        return [(match["score"], match["metadata"]) for match in result["matches"]]
