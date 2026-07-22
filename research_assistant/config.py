"""
Configuration for Agentic RAG Research Assistant.
Reads from .env file. Perplexity API is primary LLM backend.
"""

import os
from typing import Optional
from dotenv import load_dotenv

load_dotenv()


def _strip_env(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    return value.strip().strip('"').strip("'")


class Config:
    # ── LLM Backend ───────────────────────────────────────────────────────────
    LLM_BACKEND: str = os.getenv("LLM_BACKEND", "perplexity").lower()

    # Perplexity (primary)
    PERPLEXITY_API_KEY: Optional[str] = os.getenv("PERPLEXITY_API_KEY")
    PERPLEXITY_MODEL: str = os.getenv("PERPLEXITY_MODEL", "sonar-pro")
    PERPLEXITY_BASE_URL: str = "https://api.perplexity.ai"

    # Google Gemini
    GEMINI_API_KEY: Optional[str] = _strip_env(os.getenv("GEMINI_API_KEY"))
    GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

    # Ollama (local fallback)
    OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "mistral")
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

    # ── Embeddings ────────────────────────────────────────────────────────────
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

    # ── Chunking ──────────────────────────────────────────────────────────────
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "800"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "150"))

    # ── Hybrid Retrieval ──────────────────────────────────────────────────────
    TOP_K_DENSE: int = int(os.getenv("TOP_K_DENSE", "20"))   # candidates from FAISS
    TOP_K_SPARSE: int = int(os.getenv("TOP_K_SPARSE", "20"))  # candidates from BM25
    TOP_K_FINAL: int = int(os.getenv("TOP_K_FINAL", "6"))    # final after RRF
    RRF_K: int = int(os.getenv("RRF_K", "60"))               # RRF constant
    MIN_RETRIEVAL_SCORE: float = float(os.getenv("MIN_RETRIEVAL_SCORE", "0.35"))
    MIN_RETRIEVAL_SCORE_NORMAL: float = float(os.getenv("MIN_RETRIEVAL_SCORE_NORMAL", "0.35"))
    MIN_RETRIEVAL_SCORE_LOW: float = float(os.getenv("MIN_RETRIEVAL_SCORE_LOW", "0.25"))
    LOG_FILTERED_RESULTS: bool = os.getenv("LOG_FILTERED_RESULTS", "true").lower() in ("1", "true", "yes")

    # ── Gemini verification & resilience ─────────────────────────────────────
    GEMINI_SAFETY_THRESHOLD: str = os.getenv("GEMINI_SAFETY_THRESHOLD", "BLOCK_MEDIUM_AND_ABOVE")
    GEMINI_TEMPERATURE: float = float(os.getenv("GEMINI_TEMPERATURE", "0.1"))
    GEMINI_TEMPERATURE_CREATIVE: float = float(os.getenv("GEMINI_TEMPERATURE_CREATIVE", "0.7"))
    MAX_RETRIES: int = int(os.getenv("MAX_RETRIES", "3"))
    RETRY_BACKOFF: float = float(os.getenv("RETRY_BACKOFF", "1.0"))

    # ── Agent ─────────────────────────────────────────────────────────────────
    MAX_SUB_QUERIES: int = int(os.getenv("MAX_SUB_QUERIES", "4"))
    MAX_TOKENS_ANSWER: int = int(os.getenv("MAX_TOKENS_ANSWER", "1200"))
    MAX_TOKENS_SYNTHESIS: int = int(os.getenv("MAX_TOKENS_SYNTHESIS", "3000"))
    LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.3"))

    # ── arXiv ─────────────────────────────────────────────────────────────────
    ARXIV_MAX_RESULTS: int = int(os.getenv("ARXIV_MAX_RESULTS", "20"))

    # ── Level 5 Autonomous RAG ───────────────────────────────────────────────
    REDIS_URL: Optional[str] = os.getenv("REDIS_URL")
    QUERY_CACHE_TTL: int = int(os.getenv("QUERY_CACHE_TTL", "3600"))
    SLACK_WEBHOOK_URL: Optional[str] = os.getenv("SLACK_WEBHOOK_URL")
    GEMINI_TOKEN_BUCKET_RATE: float = float(os.getenv("GEMINI_TOKEN_BUCKET_RATE", "10"))
    GEMINI_TOKEN_BUCKET_CAPACITY: float = float(os.getenv("GEMINI_TOKEN_BUCKET_CAPACITY", "20"))
    GEMINI_IMAGE_SAFETY_THRESHOLD: str = os.getenv(
        "GEMINI_IMAGE_SAFETY_THRESHOLD", "BLOCK_MEDIUM_AND_ABOVE"
    )
    LEVEL5_ENABLED: bool = os.getenv("LEVEL5_ENABLED", "true").lower() in ("1", "true", "yes")

    # ── Provider-agnostic LLM (Deep-Read + future features) ─────────────────
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "auto").lower()
    OPENAI_API_KEY: Optional[str] = _strip_env(os.getenv("OPENAI_API_KEY"))
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    OPENAI_BASE_URL: Optional[str] = os.getenv("OPENAI_BASE_URL")

    @classmethod
    def get_effective_retrieval_threshold(cls, query: str = "") -> float:
        """Lower threshold for technical/definitional queries."""
        try:
            from retrieval.query_utils import is_technical_query

            if is_technical_query(query):
                return cls.MIN_RETRIEVAL_SCORE_LOW
        except ImportError:
            pass
        return cls.MIN_RETRIEVAL_SCORE_NORMAL

    @classmethod
    def load_runtime_overrides(cls) -> None:
        """Apply dynamic overrides from optimization/runtime_config.json."""
        try:
            from optimization.common import load_runtime_config

            runtime = load_runtime_config()
            if "min_retrieval_score" in runtime:
                cls.MIN_RETRIEVAL_SCORE = float(runtime["min_retrieval_score"])
        except Exception as exc:
            print(f"[config] Runtime overrides skipped: {exc}")

    @classmethod
    def validate(cls) -> bool:
        allowed = {"perplexity", "ollama", "gemini"}
        if cls.LLM_BACKEND not in allowed:
            raise ValueError(
                f"Invalid LLM_BACKEND '{cls.LLM_BACKEND}'. "
                f"Use one of: {', '.join(sorted(allowed))}."
            )
        if cls.LLM_BACKEND == "perplexity" and not cls.PERPLEXITY_API_KEY:
            raise ValueError(
                "PERPLEXITY_API_KEY is required when LLM_BACKEND=perplexity.\n"
                "Set it in research_assistant/.env"
            )
        if cls.LLM_BACKEND == "gemini" and not cls.GEMINI_API_KEY:
            raise ValueError(
                "GEMINI_API_KEY is required when LLM_BACKEND=gemini.\n"
                "Set it in research_assistant/.env"
            )
        return True

    @classmethod
    def get_llm(cls):
        """Return configured LangChain LLM instance."""
        if cls.LLM_BACKEND == "perplexity":
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(
                model=cls.PERPLEXITY_MODEL,
                openai_api_key=cls.PERPLEXITY_API_KEY,
                openai_api_base=cls.PERPLEXITY_BASE_URL,
                temperature=cls.LLM_TEMPERATURE,
                max_tokens=cls.MAX_TOKENS_ANSWER,
            )
        if cls.LLM_BACKEND == "gemini":
            from langchain_google_genai import ChatGoogleGenerativeAI
            return ChatGoogleGenerativeAI(
                model=cls.GEMINI_MODEL,
                google_api_key=cls.GEMINI_API_KEY,
                temperature=cls.GEMINI_TEMPERATURE_CREATIVE,
                max_output_tokens=cls.MAX_TOKENS_ANSWER,
            )
        from langchain_community.chat_models import ChatOllama
        return ChatOllama(
            model=cls.OLLAMA_MODEL,
            base_url=cls.OLLAMA_BASE_URL,
            temperature=cls.LLM_TEMPERATURE,
        )


Config.load_runtime_overrides()
