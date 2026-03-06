from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings
from pydantic_settings import SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_env: str = Field(default="local", alias="APP_ENV")
    api_host: str = Field(default="0.0.0.0", alias="API_HOST")
    api_port: int = Field(default=5001, alias="API_PORT")
    postgres_dsn: str = Field(
        default="postgresql+asyncpg://adk:adk@localhost:5432/adk",
        alias="POSTGRES_DSN",
    )
    mongo_uri: str = Field(default="mongodb://localhost:27017", alias="MONGO_URI")
    mongo_db: str = Field(default="adk", alias="MONGO_DB")
    mongo_memory_collection: str = Field(
        default="memory_chunks", alias="MONGO_MEMORY_COLLECTION"
    )
    mongo_profile_collection: str = Field(
        default="user_profiles", alias="MONGO_PROFILE_COLLECTION"
    )
    embedding_provider: str = Field(default="azure_openai", alias="EMBEDDING_PROVIDER")
    embedding_dim: int = Field(default=3072, alias="EMBEDDING_DIM")
    openai_api_key: str | None = Field(default=None, alias="OPENAI_API_KEY")

    azure_openai_endpoint: str | None = Field(
        default=None, alias="AZURE_OPENAI_ENDPOINT"
    )
    azure_openai_key: str | None = Field(default=None, alias="AZURE_OPENAI_KEY")
    azure_openai_embedding_model: str = Field(
        default="text-embedding-3-large", alias="AZURE_OPENAI_EMBEDDING_MODEL"
    )
    azure_openai_api_version: str = Field(
        default="2024-06-01", alias="AZURE_OPENAI_API_VERSION"
    )

    api_endpoint: str = Field(default="https://biuprodapi.piramalfinance.com/open-ai/invocations", alias="API_ENDPOINT")
    api_key: str | None = Field(default=None, alias="API_KEY")

    vertex_project: str | None = Field(default=None, alias="VERTEX_PROJECT")
    vertex_location: str | None = Field(default=None, alias="VERTEX_LOCATION")
    vector_index_name: str = Field(
        default="memory_vector_index", alias="MONGO_VECTOR_INDEX"
    )
    vector_score_threshold: float = Field(
        default=0.45, alias="VECTOR_SCORE_THRESHOLD"
    )
    vector_top_k: int = Field(default=8, alias="VECTOR_TOP_K")
    memory_chunk_max_tokens: int = Field(
        default=600, alias="MEMORY_CHUNK_MAX_TOKENS"
    )
    memory_chunk_overlap_tokens: int = Field(
        default=100, alias="MEMORY_CHUNK_OVERLAP_TOKENS"
    )
    temp_state_prefix: str = Field(default="temp:", alias="TEMP_STATE_PREFIX")


def get_settings() -> Settings:
    return Settings()
