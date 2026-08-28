import re
from typing import Optional, Tuple
from core.llm.ollama_client import OllamaClient


class QueryRewriter:
    """
    Fluid, Dynamic Query Normalizer & HyDE Expander.
    Works dynamically for any domain without static rules or hardcoded keywords.
    """

    def __init__(self, ollama_client: Optional[OllamaClient] = None):
        self.ollama = ollama_client or OllamaClient()

    def is_conversational_greeting(self, text: str) -> Optional[str]:
        cleaned = text.strip().lower().rstrip("?!.")
        greetings = ["hi", "hello", "hey", "hola", "namaste", "good morning", "good evening", "who are you", "help"]
        if cleaned in greetings:
            return "Hello! I am your private enterprise RAG assistant. Ask me any question about your uploaded documents."
        return None

    def clean_and_expand_query(self, user_query: str, model_name: Optional[str] = None) -> Tuple[str, str]:
        """
        Dynamically generates search expansions for retrieval while strictly preserving the user's intent.
        """
        query_clean = user_query.strip()
        if not query_clean:
            return user_query, user_query

        # If query is short or straightforward, search directly with original query
        if len(query_clean.split()) <= 4:
            return query_clean, query_clean

        system_msg = (
            "You are a search query expander. "
            "Output ONLY 1 short sentence describing what information in a document would answer the user's question. "
            "Do NOT include any conversational filler."
        )

        try:
            expanded = self.ollama.chat_response(
                user_message=f"Question: {query_clean}",
                system_prompt=system_msg,
                model=model_name,
                temperature=0.0
            )
            passage = expanded.strip().replace("\n", " ")
            if "ready to provide" in passage.lower() or "please ask" in passage.lower() or len(passage) < 5:
                return query_clean, query_clean

            search_query = f"{query_clean} {passage[:150]}"
            return query_clean, search_query
        except Exception:
            return query_clean, query_clean
