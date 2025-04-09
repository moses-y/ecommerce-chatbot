# src/llm/interface.py
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

class LLMInterface(ABC):
    """Abstract Base Class for Large Language Model services."""

    @abstractmethod
    def generate_response(
        self,
        prompt: str,
        history: Optional[List[Dict[str, str]]] = None,
        temperature: float = 0.7,
        max_output_tokens: int = 1024
    ) -> str:
        """
        Generates a response from the LLM based on a prompt and optional history.

        Args:
            prompt: The user's input or a specific instruction for the LLM.
            history: A list of previous turns in the conversation, typically
                     in the format [{'role': 'user', 'parts': ['text']},
                                   {'role': 'model', 'parts': ['text']}].
            temperature: Controls the randomness of the output.
            max_output_tokens: The maximum number of tokens to generate.

        Returns:
            The generated text response from the LLM.
        """
        pass

    @abstractmethod
    def determine_intent(
        self,
        user_input: str,
        available_intents: List[str],
        history: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """
        Determines the user's primary intent based on their input and available options.

        Args:
            user_input: The user's message.
            available_intents: A list of possible intents the chatbot can handle
                               (e.g., 'check_order_status', 'ask_return_policy', 'request_human').
            history: Optional conversation history.

        Returns:
            The identified intent string (must be one of the available_intents)
            or 'unknown' / 'general_query' if no specific intent matches.
        """
        pass

    # You could add other common LLM tasks here, e.g., summarization, classification