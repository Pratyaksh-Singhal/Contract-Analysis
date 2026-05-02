import os
from typing import List
from .interfaces import BaseVectorStore, Document
from .config import VectorStoreConfig, EmbeddingConfig


class _LocalFallbackEmbedder:
    def name(self) -> str:
        return "local-fallback"

    def embed_query(self, input: List[str]) -> List[List[float]]:
        return self.__call__(input)

    def __call__(self, input: List[str]) -> List[List[float]]:
        import hashlib
        dim = 384
        result = []
        for text in input:
            vec = [0.0] * dim
            for i in range(max(1, len(text) - 2)):
                trigram = text[i:i+3]
                h = int(hashlib.md5(trigram.encode()).hexdigest(), 16)
                vec[h % dim] += 1.0
            norm = (sum(x*x for x in vec) ** 0.5) or 1.0
            result.append([x / norm for x in vec])
        return result


class ChromaVectorStore(BaseVectorStore):
    def __init__(self, vector_config: VectorStoreConfig, embedding_config: EmbeddingConfig):
        import chromadb
        from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

        os.makedirs(vector_config.persist_directory, exist_ok=True)

        try:
            self._embedding_fn = SentenceTransformerEmbeddingFunction(
                model_name=embedding_config.model_name
            )
        except Exception:
            print("  [!] sentence-transformers unavailable, using local fallback embedder.")
            self._embedding_fn = _LocalFallbackEmbedder()

        self._client = chromadb.PersistentClient(path=vector_config.persist_directory)
        self._collection = self._client.get_or_create_collection(
            name=vector_config.collection_name,
            embedding_function=self._embedding_fn,
            metadata={"hnsw:space": "cosine"}
        )

    def add_documents(self, documents: List[Document]) -> None:
        if not documents:
            return
        self._collection.upsert(
            ids=[f"{doc.source}__p{doc.page}__c{doc.chunk_index}" for doc in documents],
            documents=[doc.content for doc in documents],
            metadatas=[{"source": doc.source, "page": doc.page, "chunk_index": doc.chunk_index} for doc in documents]
        )

    def similarity_search(self, query: str, k: int) -> List[Document]:
        results = self._collection.query(
            query_texts=[query],
            n_results=min(k, self._collection.count())
        )
        documents = []
        if not results["documents"] or not results["documents"][0]:
            return documents
        for text, metadata in zip(results["documents"][0], results["metadatas"][0]):
            documents.append(Document(
                content=text,
                source=metadata.get("source", "unknown"),
                page=metadata.get("page", 0),
                chunk_index=metadata.get("chunk_index", 0)
            ))
        return documents

    def is_empty(self) -> bool:
        return self._collection.count() == 0

    def clear(self) -> None:
        self._client.delete_collection(self._collection.name)
        self._collection = self._client.get_or_create_collection(
            name="contracts",
            embedding_function=self._embedding_fn,
            metadata={"hnsw:space": "cosine"}
        )

    def document_count(self) -> int:
        return self._collection.count()
