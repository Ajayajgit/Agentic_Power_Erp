"""
vectorstore/factory.py

Chooses Pinecone if configured and reachable, otherwise falls back to the
local in-memory store. This keeps the "use Pinecone if needed" requirement
satisfied while still letting the prototype run with no cloud account.
"""

import logging

from prototype.vectorstore.local_store import LocalVectorStore
from prototype.vectorstore.pinecone_store import PineconeVectorStore

logger = logging.getLogger(__name__)


class VectorStoreFactory:
    @staticmethod
    def create(settings, embedding_dimension: int):
        if settings.pinecone_api_key:
            try:
                return PineconeVectorStore(
                    api_key=settings.pinecone_api_key,
                    index_name=settings.pinecone_index_name,
                    cloud=settings.pinecone_cloud,
                    region=settings.pinecone_region,
                    dimension=embedding_dimension,
                )
            except Exception as exc:
                logger.warning("Pinecone unavailable (%s). Falling back to local in-memory vector store.", exc)
                return LocalVectorStore()
        else:
            logger.info("No PINECONE_API_KEY found - using local in-memory vector store.")
            return LocalVectorStore()
