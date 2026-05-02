import os
from typing import List, Tuple
from .interfaces import BaseVectorStore, Document
from .loaders import LoaderRegistry
from .chunker import TextChunker
from .config import AppConfig


class IngestionPipeline:
    def __init__(self, config: AppConfig, vector_store: BaseVectorStore):
        self._config = config
        self._vector_store = vector_store
        self._loader_registry = LoaderRegistry()
        self._chunker = TextChunker(config.chunking)

    def ingest_directory(self, directory: str) -> Tuple[int, List[str]]:
        files = self._get_supported_files(directory)
        if not files:
            return 0, []

        all_chunks: List[Document] = []
        processed_files: List[str] = []

        for file_path in files:
            chunks = self._process_file(file_path)
            if chunks:
                all_chunks.extend(chunks)
                processed_files.append(os.path.basename(file_path))
                print(f"  [+] {os.path.basename(file_path)} → {len(chunks)} chunks")

        if all_chunks:
            self._vector_store.add_documents(all_chunks)

        return len(all_chunks), processed_files

    def ingest_file(self, file_path: str) -> int:
        chunks = self._process_file(file_path)
        if chunks:
            self._vector_store.add_documents(chunks)
        return len(chunks)

    def _process_file(self, file_path: str) -> List[Document]:
        try:
            loader = self._loader_registry.get_loader(file_path)
            raw_docs = loader.load(file_path)
            if not raw_docs:
                print(f"  [!] {os.path.basename(file_path)} — empty or unreadable")
                return []
            return self._chunker.chunk_all(raw_docs)
        except Exception as e:
            print(f"  [!] Failed to load {os.path.basename(file_path)}: {e}")
            return []

    def _get_supported_files(self, directory: str) -> List[str]:
        if not os.path.exists(directory):
            return []
        return [
            os.path.join(directory, f)
            for f in os.listdir(directory)
            if self._loader_registry.supports(f)
            and os.path.isfile(os.path.join(directory, f))
        ]
