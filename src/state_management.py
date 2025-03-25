"""
state_management.py - Handles conversation memory and state management
"""
from typing import List, Dict, Any, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class ConversationMemory:
    def __init__(self, max_history: int = 10):
        self.max_history = max_history
        self.messages: List[Dict[str, Any]] = []
        self.conversation_summary: Optional[str] = None
        self.key_details: Dict[str, Any] = {}
        
    def add_message(self, role: str, content: str) -> None:
        """Add a message while maintaining history limit."""
        self.messages.append({
            "role": role, 
            "content": content, 
            "timestamp": datetime.now().isoformat()
        })
        if len(self.messages) > self.max_history:
            self.messages.pop(0)
            
    def get_context_window(self, window_size: int = 5) -> List[Dict[str, Any]]:
        """Get the most recent n messages."""
        return self.messages[-window_size:]
    
    def update_key_details(self, key: str, value: Any) -> None:
        """Track important details like order IDs, customer info, etc."""
        self.key_details[key] = value
        logger.debug(f"Updated key detail: {key}={value}")
    
    def get_conversation_context(self) -> Dict[str, Any]:
        """Get the full conversation context including key details."""
        return {
            "messages": self.messages,
            "key_details": self.key_details,
            "summary": self.conversation_summary
        }

    def clear_history(self) -> None:
        """Clear conversation history but maintain key details."""
        self.messages = []
        self.conversation_summary = None