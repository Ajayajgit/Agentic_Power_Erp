"""
vectorstore/local_store.py

Local, in-memory vector store fallback - plain numpy cosine similarity.
Used automatically when PINECONE_API_KEY is not configured, so this
prototype can run with zero cloud dependencies. Namespaces are still
respected (implemented as separate in-memory buckets) to keep the
tenant-isolation behavior consistent with the Pinecone implementation.
"""

from typing import Dict, List, Tuple

import numpy as np

from prototype.vectorstore.base import VectorStore


class LocalVectorStore(VectorStore):
    name = "local-inmemory"

    def __init__(self):
        # namespace -> {"ids": [...], "vectors": np.ndarray, "metadatas": [...]}
        self._buckets: Dict[str, Dict] = {}

    def upsert(self, ids: List[str], vectors: List[List[float]], metadatas: List[Dict], namespace: str) -> None:
        bucket = self._buckets.setdefault(namespace, {"ids": [], "vectors": [], "metadatas": []})
        bucket["ids"].extend(ids)
        bucket["vectors"].extend(vectors)
        bucket["metadatas"].extend(metadatas)

    def query(self, vector: List[float], top_k: int, namespace: str) -> List[Tuple[float, Dict]]:
        bucket = self._buckets.get(namespace)
        if not bucket or not bucket["vectors"]:
            return []

        matrix = np.array(bucket["vectors"], dtype=float)
        query_vec = np.array(vector, dtype=float)

        # cosine similarity
        matrix_norms = np.linalg.norm(matrix, axis=1)
        query_norm = np.linalg.norm(query_vec)
        denom = matrix_norms * query_norm
        denom[denom == 0] = 1e-10
        scores = (matrix @ query_vec) / denom

        top_indices = np.argsort(-scores)[:top_k]
        return [(float(scores[i]), bucket["metadatas"][i]) for i in top_indices]
