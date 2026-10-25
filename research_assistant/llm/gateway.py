"""
Provider-agnostic LLM gateway.
Orchestration code calls get_llm_gateway() — never imports langchain_* directly.
Supports: OpenAI-compatible APIs, Gemini, Ollama, Perplexity (via OpenAI base URL).
"""

from __future__ import annotations

import asyncio
import base64
import os
from typing import Optional

import httpx

from config import Config


class LLMGateway:
    """Unified async text completion interface."""

    def __init__(
        self,
        provider: str,
        model: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        self.provider = provider.lower()
        self.model = model
        self.api_key = api_key
        self.base_url = base_url

    async def complete(
        self,
        system: str,
        user: str,
        *,
        temperature: float = 0.3,
        max_tokens: int = 4096,
    ) -> str:
        if self.provider in ("openai", "perplexity", "groq", "together", "openai_compat"):
            return await self._complete_openai_compat(system, user, temperature, max_tokens)
        if self.provider == "gemini":
            return await self._complete_gemini(system, user, temperature, max_tokens)
        if self.provider == "ollama":
            return await self._complete_ollama(system, user, temperature, max_tokens)
        raise ValueError(f"Unsupported LLM provider: {self.provider}")

    async def complete_with_image(
        self,
        system: str,
        user: str,
        image: bytes,
        *,
        mime_type: str = "image/png",
        temperature: float = 0.1,
        max_tokens: int = 1200,
    ) -> str:
        """Generate a response grounded in both the prompt and one image."""
        encoded = base64.b64encode(image).decode("ascii")
        if self.provider in ("openai", "openai_compat"):
            url = (self.base_url or "https://api.openai.com/v1").rstrip("/") + "/chat/completions"
            headers = {"Content-Type": "application/json"}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            payload = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": user},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{mime_type};base64,{encoded}",
                                },
                            },
                        ],
                    },
                ],
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.post(url, headers=headers, json=payload)
                response.raise_for_status()
                return response.json()["choices"][0]["message"]["content"]

        if self.provider == "gemini":
            from io import BytesIO

            import google.generativeai as genai
            from PIL import Image

            genai.configure(api_key=self.api_key)
            model = genai.GenerativeModel(
                model_name=self.model,
                system_instruction=system,
                generation_config=genai.GenerationConfig(
                    temperature=temperature,
                    max_output_tokens=max_tokens,
                ),
            )
            image_part = Image.open(BytesIO(image))
            response = await asyncio.to_thread(
                model.generate_content,
                [user, image_part],
            )
            return (response.text or "").strip()

        if self.provider == "ollama":
            url = (self.base_url or Config.OLLAMA_BASE_URL).rstrip("/") + "/api/chat"
            payload = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user, "images": [encoded]},
                ],
                "stream": False,
                "options": {"temperature": temperature, "num_predict": max_tokens},
            }
            async with httpx.AsyncClient(timeout=180.0) as client:
                response = await client.post(url, json=payload)
                response.raise_for_status()
                return response.json()["message"]["content"]

        raise ValueError(
            f"Provider '{self.provider}' does not support image input. "
            "Configure VISION_PROVIDER as openai, gemini, or a vision-capable ollama model."
        )

    async def _complete_openai_compat(
        self, system: str, user: str, temperature: float, max_tokens: int
    ) -> str:
        url = (self.base_url or "https://api.openai.com/v1").rstrip("/") + "/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(url, headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]

    async def _complete_gemini(
        self, system: str, user: str, temperature: float, max_tokens: int
    ) -> str:
        import google.generativeai as genai

        genai.configure(api_key=self.api_key)
        model = genai.GenerativeModel(
            model_name=self.model,
            system_instruction=system,
            generation_config=genai.GenerationConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
            ),
        )
        response = await asyncio.to_thread(model.generate_content, user)
        return (response.text or "").strip()

    async def _complete_ollama(
        self, system: str, user: str, temperature: float, max_tokens: int
    ) -> str:
        url = (self.base_url or Config.OLLAMA_BASE_URL).rstrip("/") + "/api/chat"
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }
        async with httpx.AsyncClient(timeout=180.0) as client:
            resp = await client.post(url, json=payload)
            resp.raise_for_status()
            return resp.json()["message"]["content"]

    def as_langchain_llm(self):
        """Optional bridge for legacy code paths during migration."""
        return Config.get_llm()


def _detect_provider() -> tuple[str, str, Optional[str], Optional[str]]:
    explicit = os.getenv("LLM_PROVIDER", "auto").lower()
    openai_key = os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY".lower())
    gemini_key = Config.GEMINI_API_KEY
    perplexity_key = Config.PERPLEXITY_API_KEY

    if explicit != "auto":
        if explicit == "openai":
            return "openai", os.getenv("OPENAI_MODEL", "gpt-4o-mini"), openai_key, None
        if explicit == "gemini":
            return "gemini", Config.GEMINI_MODEL, gemini_key, None
        if explicit == "perplexity":
            return "perplexity", Config.PERPLEXITY_MODEL, perplexity_key, Config.PERPLEXITY_BASE_URL
        if explicit == "ollama":
            return "ollama", Config.OLLAMA_MODEL, None, Config.OLLAMA_BASE_URL

    # Auto-detect from available keys (priority order)
    if openai_key:
        return "openai", os.getenv("OPENAI_MODEL", "gpt-4o-mini"), openai_key, None
    if gemini_key:
        return "gemini", Config.GEMINI_MODEL, gemini_key, None
    if perplexity_key:
        return "perplexity", Config.PERPLEXITY_MODEL, perplexity_key, Config.PERPLEXITY_BASE_URL
    return "ollama", Config.OLLAMA_MODEL, None, Config.OLLAMA_BASE_URL


def get_llm_gateway(
    provider: Optional[str] = None,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
) -> LLMGateway:
    """
    Return configured LLM gateway. Optional overrides for per-user BYOK (future UI).
    """
    if provider and model:
        base_url = None
        if provider == "perplexity":
            base_url = Config.PERPLEXITY_BASE_URL
        elif provider == "ollama":
            base_url = Config.OLLAMA_BASE_URL
        return LLMGateway(provider, model, api_key, base_url)

    if Config.LLM_BACKEND == "gemini" and Config.GEMINI_API_KEY:
        return LLMGateway("gemini", Config.GEMINI_MODEL, Config.GEMINI_API_KEY)
    if Config.LLM_BACKEND == "perplexity" and Config.PERPLEXITY_API_KEY:
        return LLMGateway(
            "perplexity", Config.PERPLEXITY_MODEL, Config.PERPLEXITY_API_KEY, Config.PERPLEXITY_BASE_URL
        )
    if Config.LLM_BACKEND == "ollama":
        return LLMGateway("ollama", Config.OLLAMA_MODEL, None, Config.OLLAMA_BASE_URL)

    p, m, k, b = _detect_provider()
    return LLMGateway(p, m, k, b)


def get_vision_gateway() -> LLMGateway:
    """Return an explicitly vision-capable provider configured for PDF page analysis."""
    provider = Config.VISION_PROVIDER
    model = Config.VISION_MODEL

    if provider == "auto":
        if Config.OPENAI_API_KEY:
            provider = "openai"
        elif Config.GEMINI_API_KEY:
            provider = "gemini"
        elif Config.LLM_BACKEND == "ollama":
            provider = "ollama"
        else:
            raise RuntimeError(
                "Page vision needs an image-capable model. Set OPENAI_API_KEY or "
                "GEMINI_API_KEY, or configure VISION_PROVIDER=ollama with a vision model."
            )

    if provider == "openai":
        api_key = Config.OPENAI_API_KEY
        if not api_key:
            raise RuntimeError("VISION_PROVIDER=openai requires OPENAI_API_KEY")
        return LLMGateway(
            provider,
            model or "gpt-4o-mini",
            api_key,
            Config.OPENAI_BASE_URL,
        )
    if provider == "gemini":
        api_key = Config.GEMINI_API_KEY
        if not api_key:
            raise RuntimeError("VISION_PROVIDER=gemini requires GEMINI_API_KEY")
        return LLMGateway(provider, model or Config.GEMINI_MODEL, api_key)
    if provider == "ollama":
        return LLMGateway(
            provider,
            model or Config.OLLAMA_MODEL,
            None,
            Config.OLLAMA_BASE_URL,
        )
    raise ValueError(
        f"Unsupported VISION_PROVIDER '{provider}'. Use auto, openai, gemini, or ollama."
    )
