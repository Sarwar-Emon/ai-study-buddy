"""Application settings, read from environment variables (or a .env file)."""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Any provider that speaks the OpenAI API. The names say "openai" because the
    # client library is OpenAI's, but the key can come from another provider (for
    # example Google Gemini, which has a free tier).
    openai_api_key: str = ""
    openai_base_url: str = ""  # empty = OpenAI. Set it to use another provider.
    embedding_model: str = "text-embedding-3-small"
    chat_model: str = "gpt-4o-mini"
    # Optional speed knob for models that "think" before answering (for example Gemini 3:
    # try "low" or "minimal"). Empty = do not send it. Not every model accepts it.
    reasoning_effort: str = ""

    # "openai" uses real models. "fake" runs fully offline (demo and tests only).
    provider: str = "openai"

    # Storage
    data_dir: str = "./data"

    # Retrieval
    chunk_size: int = 1000  # characters per chunk
    chunk_overlap: int = 200  # characters shared between neighbouring chunks
    top_k: int = 4  # chunks sent to the LLM
    min_score: float = 0.2  # below this cosine similarity, we refuse to answer

    # Trust check: one extra model call per answer that flags unsupported claims.
    verify_answers: bool = True

    # API safety
    api_key: str = ""  # if set, requests must send X-API-Key
    max_upload_mb: int = 20
    max_question_chars: int = 1000


@lru_cache
def get_settings() -> Settings:
    return Settings()
