# src/llm/gemini_service.py
import google.generativeai as genai
import logging
from typing import List, Dict, Optional, Any
from src.llm.interface import LLMInterface
from src.core.config import (
    GOOGLE_API_KEY,
    GEMINI_MODEL_NAME,
    GEMINI_TEMPERATURE,
    GEMINI_MAX_OUTPUT_TOKENS,
    GEMINI_TOP_P,
    GEMINI_TOP_K,
    SYSTEM_PROMPT # Import the system prompt
)

logger = logging.getLogger(__name__)

class GeminiService(LLMInterface):
    """Implementation of LLMInterface using Google's Gemini models."""

    def __init__(self):
        """Initializes the Gemini client and model."""
        try:
            genai.configure(api_key=GOOGLE_API_KEY)
            # Set up the model configuration
            self.generation_config = genai.GenerationConfig(
                temperature=GEMINI_TEMPERATURE,
                top_p=GEMINI_TOP_P,
                top_k=GEMINI_TOP_K,
                max_output_tokens=GEMINI_MAX_OUTPUT_TOKENS,
                # response_mime_type="text/plain", # Ensure text output if needed
            )
            # Safety settings - adjust as needed, be cautious with blocking too much
            self.safety_settings = [
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            ]

            self.model = genai.GenerativeModel(
                model_name=GEMINI_MODEL_NAME,
                generation_config=self.generation_config,
                safety_settings=self.safety_settings,
                # Add the system instruction here
                system_instruction=SYSTEM_PROMPT
            )
            logger.info(f"Gemini model '{GEMINI_MODEL_NAME}' initialized successfully.")
            logger.debug(f"Gemini Config: Temp={GEMINI_TEMPERATURE}, MaxTokens={GEMINI_MAX_OUTPUT_TOKENS}, TopP={GEMINI_TOP_P}, TopK={GEMINI_TOP_K}")
            logger.debug(f"Gemini System Prompt: {SYSTEM_PROMPT[:100]}...") # Log beginning of prompt

        except Exception as e:
            logger.error(f"Failed to initialize Gemini model: {e}", exc_info=True)
            # Depending on the application, you might want to raise the error
            # or handle it gracefully (e.g., set self.model to None and check later)
            raise ConnectionError(f"Failed to initialize Gemini: {e}") from e

    def generate_response(
        self,
        prompt: str, # In conversational use, this might be the latest user message
        history: Optional[List[Dict[str, str]]] = None,
        temperature: Optional[float] = None, # Allow overriding config
        max_output_tokens: Optional[int] = None # Allow overriding config
    ) -> str:
        """
        Generates a response using the Gemini model's chat session.

        Args:
            prompt: The latest user message or instruction.
            history: The conversation history in Gemini format
                     ([{'role': 'user', 'parts': ['text']}, {'role': 'model', 'parts': ['text']}]).
            temperature: Override the default temperature if provided.
            max_output_tokens: Override the default max tokens if provided.

        Returns:
            The generated text response from the LLM.
        """
        if not self.model:
            logger.error("Gemini model not initialized. Cannot generate response.")
            return "Error: The AI service is currently unavailable."

        # Use the history provided, or start fresh if none
        chat_history = history or []

        # Create a chat session *for this specific request* using the history
        # Note: Gemini API manages history state within the session object if you reuse it,
        # but passing history explicitly like this is often clearer for stateless web apps.
        # Moved inside try block

        # Prepare generation config overrides if any
        current_gen_config = self.generation_config
        if temperature is not None or max_output_tokens is not None:
             # Create a temporary config dictionary for overrides
             config_override_dict = {
                 "temperature": temperature if temperature is not None else self.generation_config.temperature,
                 "max_output_tokens": max_output_tokens if max_output_tokens is not None else self.generation_config.max_output_tokens,
                 "top_p": self.generation_config.top_p, # Keep others from default
                 "top_k": self.generation_config.top_k
             }
             # Convert dict to GenerationConfig object if necessary for send_message
             # Note: send_message might accept a dict directly for generation_config override
             # Check google-generativeai documentation for the exact expected type.
             # Assuming send_message accepts a dict for overrides:
             generation_config_override = config_override_dict
             logger.debug(f"Using generation config override: {generation_config_override}")
        else:
             generation_config_override = None # Use model's default


        try:
            # Start the chat session *inside* the try block
            chat_session = self.model.start_chat(history=chat_history)
            logger.debug(f"Sending prompt to Gemini: '{prompt[:100]}...' with history length: {len(chat_history)}")
            # Send the new prompt to the chat session
            response = chat_session.send_message(
                prompt,
                generation_config=generation_config_override # Pass overrides if any
            )

            # Check for potential safety blocks or empty responses
            if not response.parts:
                 logger.warning("Gemini response blocked or empty. Check safety settings or prompt.")
                 # Check candidate details if available
                 try:
                     finish_reason = response.candidates[0].finish_reason.name
                     safety_ratings = response.candidates[0].safety_ratings
                     logger.warning(f"Finish Reason: {finish_reason}")
                     logger.warning(f"Safety Ratings: {safety_ratings}")
                     if finish_reason == "SAFETY":
                         return "I cannot provide a response to that request due to safety guidelines."
                 except (IndexError, AttributeError) as e:
                     logger.warning(f"Could not retrieve detailed block reason: {e}")
                 return "I'm sorry, I couldn't generate a response for that."


            response_text = response.text
            logger.debug(f"Received response from Gemini: '{response_text[:100]}...'")
            return response_text

        except Exception as e:
            logger.error(f"Error during Gemini API call: {e}", exc_info=True)
            # Check for specific API errors if possible (e.g., quota, authentication)
            # Add more specific error handling based on google.api_core.exceptions if needed
            return "Sorry, I encountered an error while communicating with the AI service."


    def determine_intent(
        self,
        user_input: str,
        available_intents: List[str],
        history: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """
        Uses the LLM to determine the user's intent from a list of possibilities.

        Args:
            user_input: The user's message.
            available_intents: A list of possible intents (strings).
            history: Optional conversation history.

        Returns:
            The identified intent string or 'unknown'.
        """
        if not self.model:
            logger.error("Gemini model not initialized. Cannot determine intent.")
            return "unknown"

        # Construct a prompt specifically for intent detection
        intent_list_str = ", ".join([f"'{intent}'" for intent in available_intents])
        prompt = f"""
Analyze the following user message and determine the primary intent.
The available intents are: {intent_list_str}, 'general_query'.

User Message: "{user_input}"

Based *only* on the user message and the available intents, which intent best describes the user's goal?
Respond with *only* the single intent name from the list (e.g., 'check_order_status', 'ask_return_policy', 'request_human', 'general_query').
Do not add any explanation or other text.
Intent:"""

        # Use generate_response for this, but with specific settings
        # Lower temperature might be better for classification tasks
        intent_temperature = 0.1
        intent_max_tokens = 20 # Intent name should be short

        try:
            logger.debug(f"Determining intent for input: '{user_input}'")
            # We don't necessarily need history for simple intent detection,
            # but it could be added if context is important.
            # For now, call generate_response without history for simplicity.
            raw_intent = self.generate_response(
                prompt=prompt,
                history=None, # Or pass history if needed for context
                temperature=intent_temperature,
                max_output_tokens=intent_max_tokens
            )

            # Clean up the response - LLM might add quotes or extra spaces
            cleaned_intent = raw_intent.strip().replace("'", "").replace('"', '').lower()

            # Validate against available intents
            valid_intents_lower = [intent.lower() for intent in available_intents] + ['general_query']
            if cleaned_intent in valid_intents_lower:
                # Return the original casing if matched
                for original_intent in available_intents + ['general_query']:
                     if original_intent.lower() == cleaned_intent:
                         logger.info(f"Determined intent: '{original_intent}'")
                         return original_intent
            else:
                logger.warning(f"LLM returned unrecognized intent '{raw_intent}' (cleaned: '{cleaned_intent}'). Defaulting to 'general_query'.")
                return 'general_query' # Default if LLM fails or returns garbage

        except Exception as e:
            logger.error(f"Error during intent determination: {e}", exc_info=True)
            return 'unknown' # Indicate an error occurred

# Example Usage (for testing)
# if __name__ == "__main__":
#     logging.basicConfig(level=logging.DEBUG) # Use DEBUG for detailed logs
#     try:
#         gemini_service = GeminiService()
#         # Test generate_response
#         # response = gemini_service.generate_response("Hello there!", history=[])
#         # print(f"Generate Response Test: {response}")
#
#         # Test intent detection
#         test_input = "Can you tell me about returning items?"
#         intents = ["check_order_status", "ask_return_policy", "request_human"]
#         detected_intent = gemini_service.determine_intent(test_input, intents)
#         print(f"Intent Detection Test for '{test_input}': {detected_intent}")
#
#         test_input_2 = "Where is my package e481f51cbdc54678b7cc49136f2d6af7?"
#         detected_intent_2 = gemini_service.determine_intent(test_input_2, intents)
#         print(f"Intent Detection Test for '{test_input_2}': {detected_intent_2}")
#
#         test_input_3 = "I want to talk to a person"
#         detected_intent_3 = gemini_service.determine_intent(test_input_3, intents)
#         print(f"Intent Detection Test for '{test_input_3}': {detected_intent_3}")
#
#         test_input_4 = "What's the weather like?"
#         detected_intent_4 = gemini_service.determine_intent(test_input_4, intents)
#         print(f"Intent Detection Test for '{test_input_4}': {detected_intent_4}")
#
#     except Exception as main_e:
#         print(f"An error occurred during testing: {main_e}")