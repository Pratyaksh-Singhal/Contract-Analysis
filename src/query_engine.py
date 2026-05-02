from typing import List
from .interfaces import BaseVectorStore, BaseLLM, Document, QueryResult
from .prompt_builder import PromptBuilder
from .config import LLMConfig


class QueryEngine:
    def __init__(self, vector_store: BaseVectorStore, llm: BaseLLM, llm_config: LLMConfig):
        self._vector_store = vector_store
        self._llm = llm
        self._top_k = llm_config.top_k_results
        self._prompt_builder = PromptBuilder()

    def query(self, question: str) -> QueryResult:
        question = question.strip()

        if not question:
            return QueryResult(answer="Please provide a question.", source_documents=[], query=question)

        if self._vector_store.is_empty():
            return QueryResult(
                answer="No documents have been ingested yet. Run: python main.py ingest",
                source_documents=[],
                query=question
            )

        retrieved_docs = self._vector_store.similarity_search(query=question, k=self._top_k)
        prompt = self._prompt_builder.build(query=question, context_documents=retrieved_docs)
        answer = self._llm.generate(prompt)

        return QueryResult(answer=answer, source_documents=retrieved_docs, query=question)

    def format_result(self, result: QueryResult) -> str:
        lines = [
            "\n" + "─" * 60,
            f"Question: {result.query}",
            "─" * 60,
            f"\nAnswer:\n{result.answer}",
        ]

        if result.source_documents:
            lines.append("\n" + "─" * 60)
            lines.append("Sources used:")
            seen = set()
            for doc in result.source_documents:
                key = (doc.source, doc.page)
                if key not in seen:
                    seen.add(key)
                    page_info = f" (page {doc.page})" if doc.page > 0 else ""
                    lines.append(f"  • {doc.source}{page_info}")

        lines.append("─" * 60 + "\n")
        return "\n".join(lines)
