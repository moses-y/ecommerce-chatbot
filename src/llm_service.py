"""
llm_service.py - Enhanced LLM service with better context handling
"""
import os
import logging
from typing import List, Dict, Any
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from .state_management import ConversationMemory
from .config import SYSTEM_PROMPT, GEMINI_CONFIG
from src.credentials import verify_credentials

logger = logging.getLogger(__name__)

class LLMService:
    def __init__(self):
        """Initialize the LLM service with proper credential verification."""
        try:
            # Verify credentials before initializing
            cred_results = verify_credentials(["GOOGLE_API_KEY"])
            if not all(cred_results.values()):
                raise ValueError("Missing required Google API credentials")

            self.llm = ChatGoogleGenerativeAI(
                model=GEMINI_CONFIG["model"],
                temperature=GEMINI_CONFIG["temperature"],
                google_api_key=os.getenv("GOOGLE_API_KEY"),
                convert_system_message_to_human=True,
                timeout=15,
                max_output_tokens=GEMINI_CONFIG["max_output_tokens"],
                top_p=GEMINI_CONFIG["top_p"],
                top_k=GEMINI_CONFIG["top_k"]
            )
            logger.info("LLM service initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize LLM service: {e}")
            raise RuntimeError(f"LLM service initialization failed: {str(e)}")
        
    def generate_response(self, 
                         messages: List[Dict[str, Any]], 
                         conversation_memory: ConversationMemory) -> str:
        """Generate response with enhanced context handling."""
        try:
            # Format conversation history
            prompt_messages = [
                SystemMessage(content=self._get_enhanced_system_prompt(conversation_memory))
            ]
            
            # Add relevant context from conversation memory
            context_window = conversation_memory.get_context_window(
                GEMINI_CONFIG["context_window"]
            )
            
            for message in context_window:
                if message["role"] == "user":
                    prompt_messages.append(HumanMessage(content=message["content"]))
                elif message["role"] == "assistant":
                    prompt_messages.append(AIMessage(content=message["content"]))
                    
            # Add the current message
            if messages and messages[-1]["content"].strip():
                prompt_messages.append(HumanMessage(content=messages[-1]["content"]))
                
            response = self.llm.invoke(prompt_messages)
            return response.content
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return self._handle_fallback(messages[-1]["content"])
            
    def _get_enhanced_system_prompt(self, conversation_memory: ConversationMemory) -> str:
        """Create a context-aware system prompt."""
        base_prompt = SYSTEM_PROMPT
        
        # Add relevant context from conversation memory
        if conversation_memory.key_details:
            context_additions = []
            if "current_order_id" in conversation_memory.key_details:
                context_additions.append(
                    f"Current order being discussed: {conversation_memory.key_details['current_order_id']}"
                )
            if "customer_name" in conversation_memory.key_details:
                context_additions.append(
                    f"Customer name: {conversation_memory.key_details['customer_name']}"
                )
                
            if context_additions:
                base_prompt += "\n\nRelevant context:\n" + "\n".join(context_additions)
                
        return base_prompt
    
    def _handle_fallback(self, user_input: str) -> str:
        """Provide a fallback response when the main LLM call fails."""
        try:
            # Simplified prompt for fallback
            simple_prompt = [
                SystemMessage(content="You are a helpful e-commerce support assistant."),
                HumanMessage(content=user_input)
            ]
            response = self.llm.invoke(simple_prompt)
            return response.content
        except Exception as e:
            logger.error(f"Fallback also failed: {e}")
            return "I'm having trouble processing your request. Could you please try again or rephrase your question?"
