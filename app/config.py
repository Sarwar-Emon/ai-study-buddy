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

    # Public demo mode (for a hosted copy that strangers can open). It protects your model
    # quota: a shared daily cap on requests that call the model, a small document limit,
    # and a banner that tells visitors it is a demo.
    demo_mode: bool = False
    demo_daily_requests: int = 150  # model-using requests per day, shared by all visitors
    demo_max_docs: int = 4
    sample_pdf: str = ""  # loaded on startup in demo mode when the library is empty

    # Serve the built frontend from this folder (used by the single-container deploy)
    static_dir: str = ""


@lru_cache
def get_settings() -> Settings:
    return Settings()
