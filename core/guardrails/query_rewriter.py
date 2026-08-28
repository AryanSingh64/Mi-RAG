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
        Dynamically generates a hypothetical passage and corrected question using HyDE.
        Fluidly adapts to resumes, contracts, invoices, medical records, or tech specs.
        """
        system_msg = (
            "You are a search intelligence engine. "
            "Given a user's question (which may contain typos or be short), output TWO lines:\n"
            "Line 1: The corrected, clean English question.\n"
            "Line 2: A short hypothetical 1-sentence passage from a document that answers this question."
        )

        user_msg = f"User Question: {user_query}"

        try:
            expanded = self.ollama.chat_response(
                user_message=user_msg,
                system_prompt=system_msg,
                model=model_name,
                temperature=0.0
            )
            lines = [l.strip() for l in expanded.strip().split("\n") if l.strip()]
            
            cleaned_question = lines[0].replace("Line 1:", "").replace("Question:", "").strip().strip('"')
            hypothetical_passage = lines[1].replace("Line 2:", "").replace("Passage:", "").strip().strip('"') if len(lines) > 1 else cleaned_question

            # search_query combines the question and hypothetical context
            search_query = f"{cleaned_question} {hypothetical_passage}"
            return cleaned_question, search_query
        except Exception:
            return user_query, user_query
