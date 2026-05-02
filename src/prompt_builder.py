from typing import List
from .interfaces import Document


class PromptBuilder:
    SYSTEM_INSTRUCTION = """You are a precise legal contract analyst.
Answer questions based STRICTLY on the contract documents provided below.

Rules:
1. Answer ONLY using information from the provided context. Do not use outside knowledge.
2. If the answer is not in the context, say: "This information is not found in the provided documents."
3. Mention which document the information came from.
4. Be concise and direct.
5. For dates, numbers, names, and obligations — quote the exact text from the document.
"""

    def build(self, query: str, context_documents: List[Document]) -> str:
        if not context_documents:
            return self._no_context_prompt(query)

        return f"""{self.SYSTEM_INSTRUCTION}
--- DOCUMENT CONTEXT ---
{self._format_context(context_documents)}
--- END CONTEXT ---

Question: {query}

Answer:"""

    def _format_context(self, documents: List[Document]) -> str:
        parts = []
        for i, doc in enumerate(documents, start=1):
            page_info = f", Page {doc.page}" if doc.page > 0 else ""
            parts.append(f"[Source {i}: {doc.source}{page_info}]\n{doc.content.strip()}")
        return "\n\n".join(parts)

    def _no_context_prompt(self, query: str) -> str:
        return (
            f"{self.SYSTEM_INSTRUCTION}\n\n"
            "No relevant document sections were found.\n\n"
            f"Question: {query}\n\n"
            "Answer: This information is not found in the provided documents."
        )
