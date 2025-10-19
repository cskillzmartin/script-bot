"""Configuration management for the content generation pipeline."""

import os
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings with environment variable support."""

    # API Keys and Services
    serpapi_api_key: Optional[str] = Field(None, env="SERPAPI_API_KEY")

    # Database connections
    mongodb_url: str = Field("mongodb://admin:adminpass@localhost:27017", env="MONGODB_URL")
    qdrant_url: str = Field("http://localhost:6333", env="QDRANT_URL")
    qdrant_api_key: Optional[str] = Field(None, env="QDRANT_API_KEY")

    # Ollama configuration
    ollama_base_url: str = Field("http://localhost:11434", env="OLLAMA_BASE_URL")
    ollama_model: str = Field("llama3", env="OLLAMA_MODEL")

    # Search configuration
    max_search_results: int = Field(10, env="MAX_SEARCH_RESULTS")
    search_timeout_seconds: int = Field(30, env="SEARCH_TIMEOUT_SECONDS")
    content_extraction_timeout_seconds: int = Field(60, env="CONTENT_EXTRACTION_TIMEOUT_SECONDS")

    # Content processing
    max_content_length: int = Field(8000, env="MAX_CONTENT_LENGTH")
    embedding_model: str = Field("all-MiniLM-L6-v2", env="EMBEDDING_MODEL")
    similarity_threshold: float = Field(0.7, env="SIMILARITY_THRESHOLD")

    # Script generation
    words_per_minute: int = Field(150, env="WORDS_PER_MINUTE")
    max_script_iterations: int = Field(3, env="MAX_SCRIPT_ITERATIONS")

    # File paths
    output_directory: str = Field("outputs", env="OUTPUT_DIRECTORY")
    scripts_directory: str = Field("outputs/scripts", env="SCRIPTS_DIRECTORY")
    logs_directory: str = Field("outputs/logs", env="LOGS_DIRECTORY")

    # Logging
    log_level: str = Field("INFO", env="LOG_LEVEL")
    enable_structured_logging: bool = Field(True, env="ENABLE_STRUCTURED_LOGGING")

    # Performance
    max_concurrent_searches: int = Field(5, env="MAX_CONCURRENT_SEARCHES")
    request_timeout_seconds: int = Field(30, env="REQUEST_TIMEOUT_SECONDS")

    class Config:
        """Pydantic configuration."""

        env_file = ".env"
        case_sensitive = False


# Global settings instance
settings = Settings()


def ensure_directories():
    """Ensure all required directories exist."""
    directories = [
        settings.output_directory,
        settings.scripts_directory,
        settings.logs_directory,
    ]

    for directory in directories:
        os.makedirs(directory, exist_ok=True)


def get_database_name() -> str:
    """Get MongoDB database name."""
    return "content_generation_pipeline"


def get_qdrant_collection_name() -> str:
    """Get Qdrant collection name."""
    return "content_embeddings"
