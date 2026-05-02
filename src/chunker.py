from typing import List
from .interfaces import Document
from .config import ChunkingConfig


class TextChunker:
    def __init__(self, config: ChunkingConfig):
        self._chunk_size = config.chunk_size
        self._overlap = config.chunk_overlap

    def chunk(self, document: Document) -> List[Document]:
        words = document.content.split()
        if not words:
            return []

        if len(words) <= self._chunk_size:
            return [Document(
                content=document.content,
                source=document.source,
                page=document.page,
                chunk_index=0
            )]

        chunks = []
        start = 0
        chunk_index = 0
        step = self._chunk_size - self._overlap

        if step <= 0:
            step = self._chunk_size

        while start < len(words):
            chunk_words = words[start:start + self._chunk_size]
            chunks.append(Document(
                content=" ".join(chunk_words),
                source=document.source,
                page=document.page,
                chunk_index=chunk_index
            ))
            chunk_index += 1
            start += step

        return chunks

    def chunk_all(self, documents: List[Document]) -> List[Document]:
        all_chunks = []
        for doc in documents:
            all_chunks.extend(self.chunk(doc))
        return all_chunks
