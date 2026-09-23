"""
providers/llm_provider.py

LLM provider abstraction. The reasoning workflow depends only on the
LLMProvider interface (Dependency Inversion), never on OpenAI or Groq
directly.

- OpenAILLMProvider: primary, hosted.
- GroqLLMProvider: fallback, serves open-source models (e.g. Llama 3.3)
  through Groq's fast inference API. This satisfies the requirement that
  the system supports open-source LLMs via Groq as a fallback route.
- ResilientLLMProvider: tries primary first, falls back automatically.
"""

import logging
from abc import ABC, abstractmethod
from dotenv import load_dotenv

logger = logging.getLogger(__name__)


class LLMProvider(ABC):
    name: str = "base"

    @abstractmethod
    def generate(self, system_prompt: str, user_prompt: str, max_tokens: int = 2000) -> str:
        """Return the raw text completion for the given prompts."""
        raise NotImplementedError


class OpenAILLMProvider(LLMProvider):
    """Primary LLM provider - OpenAI hosted chat models."""

    name = "openai"

    def __init__(self, api_key: str, model: str):
        if not api_key:
            raise ValueError("OPENAI_API_KEY is not set - cannot use OpenAILLMProvider")
        from openai import OpenAI  # lazy import
        self._client = OpenAI(api_key=api_key)
        self._model = model

    def generate(self, system_prompt: str, user_prompt: str, max_tokens: int = 2000) -> str:
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=max_tokens,
            temperature=0.2,
        )
        return response.choices[0].message.content


class GroqLLMProvider(LLMProvider):
    """
    Fallback LLM provider - serves open-source models (e.g. Llama 3.3 70B)
    via Groq's inference API. Used automatically if the primary is
    unavailable or fails.
    """

    name = "groq"

    def __init__(self, api_key: str, model: str):
        if not api_key:
            raise ValueError("GROQ_API_KEY is not set - cannot use GroqLLMProvider")
        from groq import Groq  # lazy import
        self._client = Groq(api_key=api_key)
        self._model = model

    def generate(self, system_prompt: str, user_prompt: str, max_tokens: int = 2000) -> str:
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=max_tokens,
            temperature=0.2
        )
        return response.choices[0].message.content


class ResilientLLMProvider(LLMProvider):
    """Wraps a primary + fallback LLMProvider, falling back on any failure."""

    def __init__(self, primary: LLMProvider, fallback: LLMProvider):
        self._primary = primary
        self._fallback = fallback
        self.last_provider_used = None

    def generate(self, system_prompt: str, user_prompt: str, max_tokens: int = 2000) -> str:
        try:
            result = self._primary.generate(system_prompt, user_prompt, max_tokens)
            self.last_provider_used = self._primary.name
            return result
        except Exception as exc:
            logger.warning(
                "Primary LLM provider '%s' failed (%s). Falling back to '%s'.",
                self._primary.name, exc, self._fallback.name,
            )
            result = self._fallback.generate(system_prompt, user_prompt, max_tokens)
            self.last_provider_used = self._fallback.name
            return result

    @property
    def name(self) -> str:
        return self.last_provider_used or "unresolved"


class LLMProviderFactory:
    """Builds a ResilientLLMProvider from settings."""

    @staticmethod
    def create(settings) -> ResilientLLMProvider:
        if settings.openai_api_key:
            primary = OpenAILLMProvider(settings.openai_api_key, settings.openai_llm_model)
        elif settings.groq_api_key:
            # No OpenAI key at all - use Groq directly as the effective primary.
            logger.info("No OPENAI_API_KEY found - using Groq as the effective primary LLM.")
            primary = GroqLLMProvider(settings.groq_api_key, settings.groq_llm_model)
        else:
            raise RuntimeError(
                "No LLM credentials found. Set OPENAI_API_KEY and/or GROQ_API_KEY in .env"
            )

        if not settings.groq_api_key:
            # No fallback key available - fall back to the same primary
            # (ResilientLLMProvider still works, it just has nowhere further to go).
            fallback = primary
        else:
            fallback = GroqLLMProvider(settings.groq_api_key, settings.groq_llm_model)

        return ResilientLLMProvider(primary=primary, fallback=fallback)
