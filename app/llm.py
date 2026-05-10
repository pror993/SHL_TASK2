from dataclasses import dataclass
from typing import Optional, Protocol
import logging

from .config import GEMINI_API_KEY, LLM_MODEL, OPENAI_API_KEY, OPENROUTER_API_KEY, OPENROUTER_URL


class LLMClient(Protocol):
    async def generate_json(self, system_prompt: str, user_prompt: str) -> str:
        ...


@dataclass
class GeminiClient:
    model_name: str
    api_key: str

    async def generate_json(self, system_prompt: str, user_prompt: str) -> str:
        import google.generativeai as genai

        logger = logging.getLogger(__name__)
        try:
            genai.configure(api_key=self.api_key)
            model = genai.GenerativeModel(
                model_name=self.model_name,
                generation_config={
                    "response_mime_type": "application/json",
                    "temperature": 0.2,
                },
                system_instruction=system_prompt,
            )
            response = await model.generate_content_async(user_prompt)
            text = response.text or ""
            logger.debug("Gemini response text: %s", (text[:2000] + "...") if len(text) > 2000 else text)
            return text
        except Exception:
            logger.exception("Gemini client error during generate_json")
            raise


@dataclass
class OpenAIClient:
    model_name: str
    api_key: str
    base_url: Optional[str] = None

    async def generate_json(self, system_prompt: str, user_prompt: str) -> str:
        try:
            from openai import AsyncOpenAI
        except ImportError as exc:
            raise ImportError(
                "openai package is not installed. Add it to requirements.txt."
            ) from exc
        logger = logging.getLogger(__name__)
        try:
            if self.base_url:
                client = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)
            else:
                client = AsyncOpenAI(api_key=self.api_key)
            response = await client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={"type": "json_object"},
                temperature=0.2,
            )
            message = response.choices[0].message
            text = message.content or ""
            logger.debug("OpenAI/OpenRouter response text: %s", (text[:2000] + "...") if len(text) > 2000 else text)
            return text
        except Exception:
            logger.exception("OpenAI/OpenRouter client error during generate_json")
            raise


def _infer_provider(model_name: str) -> str:
    lowered = model_name.strip().lower()
    if lowered.startswith("gemini") or "gemini" in lowered:
        return "gemini"
    if lowered.startswith("openrouter") or "openrouter" in lowered:
        return "openrouter"
    return "openai"


def _split_model_name(model_name: str) -> tuple[str, str]:
    if ":" in model_name:
        prefix, actual = model_name.split(":", 1)
        prefix = prefix.strip().lower()
        if prefix in {"openai", "gemini", "openrouter"} and actual.strip():
            return prefix, actual.strip()
    return _infer_provider(model_name), model_name


def get_llm_client(model_name: Optional[str] = None) -> LLMClient:
    name = (model_name or LLM_MODEL).strip()
    if not name:
        raise ValueError("LLM_MODEL is required in .env")
    provider, resolved_name = _split_model_name(name)

    if provider == "gemini":
        if not GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY is required for Gemini models")
        return GeminiClient(resolved_name, GEMINI_API_KEY)
    if provider == "openrouter":
        if not OPENROUTER_API_KEY:
            raise ValueError("OPENROUTER_API_KEY is required for OpenRouter models")
        return OpenAIClient(resolved_name, OPENROUTER_API_KEY, base_url=OPENROUTER_URL or None)

    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY is required for OpenAI models")
    return OpenAIClient(resolved_name, OPENAI_API_KEY)
