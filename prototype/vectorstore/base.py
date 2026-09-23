"""
vectorstore/base.py

Vector store interface. Retrieval logic depends only on this interface,
never on Pinecone directly - this is what makes the "use Pinecone if
available, fall back locally otherwise" behavior possible without
touching any calling code.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple


class VectorStore(ABC):
    name: str = "base"

    @abstractmethod
    def upsert(self, ids: List[str], vectors: List[List[float]], metadatas: List[Dict], namespace: str) -> None:
        """Store vectors with associated metadata under a tenant namespace."""
        raise NotImplementedError

    @abstractmethod
    def query(self, vector: List[float], top_k: int, namespace: str) -> List[Tuple[float, Dict]]:
        """Return up to top_k (score, metadata) pairs, most similar first,
        scoped to the given namespace only (this is the tenant isolation
        boundary at the retrieval layer)."""
        raise NotImplementedError
