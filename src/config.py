import os
from dataclasses import dataclass, field


@dataclass(frozen=True)
class EmbeddingConfig:
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"


@dataclass(frozen=True)
class ChunkingConfig:
    chunk_size: int = 500
    chunk_overlap: int = 50


@dataclass(frozen=True)
class VectorStoreConfig:
    persist_directory: str = os.path.join(os.path.dirname(__file__), "..", "chroma_db")
    collection_name: str = "contracts"


@dataclass(frozen=True)
class LLMConfig:
    provider: str = "gemini"
    model_name: str = "gemini-2.5-flash"
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")
    ollama_base_url: str = "http://localhost:11434"
    temperature: float = 0.0
    top_k_results: int = 4


@dataclass(frozen=True)
class AppConfig:
    contracts_dir: str = os.path.join(os.path.dirname(__file__), "..", "contracts")
    supported_extensions: tuple = (".pdf", ".txt", ".docx")
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    vector_store: VectorStoreConfig = field(default_factory=VectorStoreConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)


config = AppConfig()
