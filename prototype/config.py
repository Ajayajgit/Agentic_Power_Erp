"""
config.py

Single source of truth for all runtime configuration. Everything is loaded
from environment variables (via a .env file) so no keys or model names are
hardcoded anywhere else in the codebase.
"""

import os
from dataclasses import dataclass
from dotenv import load_dotenv

# Ensure we load the .env file that lives alongside this config module
_HERE = os.path.dirname(__file__)
_DOTENV_PATH = os.path.join(_HERE, ".env")
load_dotenv(dotenv_path=_DOTENV_PATH)


def _get_bool(name: str, default: bool) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes")


@dataclass(frozen=True)
class Settings:
    # --- LLM: primary (OpenAI) + fallback (Groq, open-source models) ---
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_llm_model: str = os.getenv("OPENAI_LLM_MODEL", "gpt-4o-mini")

    groq_api_key: str = os.getenv("GROQ_API_KEY", "")
    groq_llm_model: str = os.getenv("GROQ_LLM_MODEL", "llama-3.3-70b-versatile")

    # --- Embeddings: primary (OpenAI) + fallback (local sentence-transformers) ---
    embedding_model_primary: str = os.getenv("EMBEDDING_MODEL_PRIMARY", "text-embedding-3-small")
    embedding_model_fallback: str = os.getenv("EMBEDDING_MODEL_FALLBACK", "all-MiniLM-L6-v2")
    # --- Gemini (Google) embeddings (secondary) ---
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")
    gemini_embedding_model: str = os.getenv("GEMINI_EMBEDDING_MODEL", "embed-gecko-001")

    # --- Vector store: Pinecone if configured, else local in-memory fallback ---
    pinecone_api_key: str = os.getenv("PINECONE_API_KEY", "")
    pinecone_index_name: str = os.getenv("PINECONE_INDEX_NAME", "netsuite-plan-rag-prototype")
    pinecone_cloud: str = os.getenv("PINECONE_CLOUD", "aws")
    pinecone_region: str = os.getenv("PINECONE_REGION", "us-east-1")

    # --- Workflow tuning ---
    top_k_retrieval: int = int(os.getenv("TOP_K_RETRIEVAL", "5"))
    max_estimate_retries: int = int(os.getenv("MAX_ESTIMATE_RETRIES", "3"))

    # --- Tenant / engagement identifiers for this run ---
    tenant_id: str = os.getenv("TENANT_ID", "power_cloud")
    engagement_id: str = os.getenv("ENGAGEMENT_ID", "meridian_outdoor_supply_2026")


settings = Settings()
