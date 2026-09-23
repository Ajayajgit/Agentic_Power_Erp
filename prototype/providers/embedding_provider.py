"""
providers/embedding_provider.py

Embedding provider abstraction (Dependency Inversion: the rest of the system
depends on the EmbeddingProvider interface, never on OpenAI or
sentence-transformers directly).

- OpenAIEmbeddingProvider: primary, hosted, requires OPENAI_API_KEY.
- SentenceTransformerEmbeddingProvider: fallback, open-source, runs locally,
  no API key required.
- ResilientEmbeddingProvider: tries primary first, transparently falls back
  on any failure (missing key, network error, rate limit, etc.) and reports
  which provider actually served each call.
"""

import logging
from abc import ABC, abstractmethod
from typing import List
from typing import List

logger = logging.getLogger(__name__)


class EmbeddingProvider(ABC):
    """Interface every embedding backend must implement."""

    name: str = "base"

    @abstractmethod
    def embed(self, texts: List[str]) -> List[List[float]]:
        """Return one embedding vector per input text, same order."""
        raise NotImplementedError


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """Primary embedding provider - OpenAI hosted embeddings."""

    name = "openai"

    def __init__(self, api_key: str, model: str):
        if not api_key:
            raise ValueError("OPENAI_API_KEY is not set - cannot use OpenAIEmbeddingProvider")
        from openai import OpenAI  # lazy import so the package is only required if actually used
        self._client = OpenAI(api_key=api_key)
        self._model = model

    def embed(self, texts: List[str]) -> List[List[float]]:
        response = self._client.embeddings.create(model=self._model, input=texts)
        return [item.embedding for item in response.data]


class SentenceTransformerEmbeddingProvider(EmbeddingProvider):
    """
    Fallback embedding provider - open-source, runs locally, no API key
    needed. Used automatically whenever the primary is unavailable.
    """

    name = "sentence-transformers"

    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer  # lazy import
        self._model = SentenceTransformer(model_name)

    def embed(self, texts: List[str]) -> List[List[float]]:
        vectors = self._model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        return [v.tolist() for v in vectors]


class LazySentenceTransformerEmbeddingProvider(EmbeddingProvider):
    """Lazy-loading wrapper for sentence-transformers to avoid heavy imports at startup.

    The actual `SentenceTransformer` model is only instantiated on the first call
    to `embed(...)`. Subsequent calls reuse the loaded model.
    """

    name = "sentence-transformers (lazy)"

    def __init__(self, model_name: str):
        self._model_name = model_name
        self._model = None

    def _ensure_model(self):
        if self._model is None:
            logger.info("Initializing SentenceTransformer model '%s' (lazy)", self._model_name)
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self._model_name)

    def embed(self, texts: List[str]) -> List[List[float]]:
        self._ensure_model()
        vectors = self._model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        return [v.tolist() for v in vectors]


class GeminiEmbeddingProvider(EmbeddingProvider):
    """
    Secondary embedding provider - Google Gemini via `google-genai` or `google.generativeai`.
    """

    name = "gemini"

    def __init__(self, api_key: str, model: str):
        if not api_key:
            raise ValueError("GEMINI_API_KEY is not set - cannot use GeminiEmbeddingProvider")
        
        self._model = model
        self._is_new_sdk = False

        # 1. Try modern google-genai SDK
        try:
            from google import genai
            self._client = genai.Client(api_key=api_key)
            self._is_new_sdk = True
            return
        except (ImportError, Exception):
            pass

        # 2. Fall back to legacy google.generativeai package
        try:
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            self._client = genai
        except Exception as exc:
            raise ImportError(
                "google-genai or google-generativeai is required for GeminiEmbeddingProvider"
            ) from exc

    def embed(self, texts: List[str]) -> List[List[float]]:
        # Ensure model name doesn't double-prefix 'models/'
        model_name = self._model if self._model.startswith("models/") else f"models/{self._model}"
        clean_model_name = self._model.replace("models/", "")

        # Option A: Modern google-genai SDK (google.genai)
        if self._is_new_sdk:
            try:
                # Batch embed via client.models.embed_content
                response = self._client.models.embed_content(
                    model=clean_model_name,
                    contents=texts,
                )
                if hasattr(response, "embeddings"):
                    return [e.values for e in response.embeddings]
                if hasattr(response, "embedding"):
                    return [response.embedding.values]
            except Exception as exc:
                raise RuntimeError(f"google-genai embed failed: {exc}") from exc

        # Option B: Legacy google.generativeai package
        try:
            response = self._client.embed_content(
                model=model_name,
                content=texts,
            )
            # Response is typically a dict with key 'embedding'
            if isinstance(response, dict) and "embedding" in response:
                embeddings = response["embedding"]
                # If single string was sent or wrapped differently
                if len(embeddings) > 0 and isinstance(embeddings[0], float):
                    return [embeddings]
                return embeddings
            elif hasattr(response, "embedding"):
                return [response.embedding]
        except Exception as exc:
            raise RuntimeError(f"google.generativeai embed failed: {exc}") from exc

        raise RuntimeError("No supported embeddings API found on google genai client")


class ResilientEmbeddingProvider(EmbeddingProvider):
    """
    Wraps a primary + fallback EmbeddingProvider. Tries primary first;
    on any exception, logs a warning and retries with the fallback.
    This is the object the rest of the system actually uses.
    """

    def __init__(self, primary: EmbeddingProvider, fallback: EmbeddingProvider):
        self._primary = primary
        self._fallback = fallback
        self.last_provider_used = None

    def embed(self, texts: List[str]) -> List[List[float]]:
        try:
            vectors = self._primary.embed(texts)
            self.last_provider_used = self._primary.name
            return vectors
        except Exception as exc:
            logger.warning(
                "Primary embedding provider '%s' failed (%s). Falling back to '%s'.",
                self._primary.name, exc, self._fallback.name,
            )
            vectors = self._fallback.embed(texts)
            self.last_provider_used = self._fallback.name
            return vectors

    @property
    def name(self) -> str:
        return self.last_provider_used or "unresolved"


class EmbeddingProviderFactory:
    """Builds a ResilientEmbeddingProvider from settings."""

    @staticmethod
    def create(settings) -> ResilientEmbeddingProvider:
        # Primary: OpenAI, only constructed if a key exists. If no key, we
        # skip straight to the fallback as primary so there's no wasted call.
        # Primary selection: OpenAI -> Gemini -> SentenceTransformers
        if settings.openai_api_key:
            primary = OpenAIEmbeddingProvider(settings.openai_api_key, settings.embedding_model_primary)
        elif settings.gemini_api_key:
            logger.info("No OPENAI_API_KEY found - using Gemini as the effective primary.")
            primary = GeminiEmbeddingProvider(settings.gemini_api_key, settings.gemini_embedding_model)
        else:
            logger.info("No OPENAI_API_KEY or GEMINI_API_KEY found - using sentence-transformers as the effective primary.")
            primary = LazySentenceTransformerEmbeddingProvider(settings.embedding_model_fallback)

        # Fallback selection: try Gemini if available, otherwise sentence-transformers
        if settings.gemini_api_key and not isinstance(primary, GeminiEmbeddingProvider):
            fallback = GeminiEmbeddingProvider(settings.gemini_api_key, settings.gemini_embedding_model)
        else:
            fallback = LazySentenceTransformerEmbeddingProvider(settings.embedding_model_fallback)

        return ResilientEmbeddingProvider(primary=primary, fallback=fallback)
