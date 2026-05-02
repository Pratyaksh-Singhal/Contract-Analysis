from abc import ABC, abstractmethod
from typing import List
from dataclasses import dataclass


@dataclass
class Document:
    content: str
    source: str
    page: int = 0
    chunk_index: int = 0


@dataclass
class QueryResult:
    answer: str
    source_documents: List[Document]
    query: str


class BaseDocumentLoader(ABC):
    @abstractmethod
    def load(self, file_path: str) -> List[Document]:
        pass

    @abstractmethod
    def supports(self, file_path: str) -> bool:
        pass


class BaseVectorStore(ABC):
    @abstractmethod
    def add_documents(self, documents: List[Document]) -> None:
        pass

    @abstractmethod
    def similarity_search(self, query: str, k: int) -> List[Document]:
        pass

    @abstractmethod
    def is_empty(self) -> bool:
        pass

    @abstractmethod
    def clear(self) -> None:
        pass


class BaseEmbedder(ABC):
    @abstractmethod
    def embed(self, texts: List[str]) -> List[List[float]]:
        pass


class BaseLLM(ABC):
    @abstractmethod
    def generate(self, prompt: str) -> str:
        pass
