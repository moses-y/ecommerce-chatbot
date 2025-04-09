# src/core/state.py
from typing import List, Dict, Optional, Any
# Import ConfigDict for Pydantic v2 config
from pydantic import BaseModel, Field, ConfigDict
import datetime

class ConversationState(BaseModel):
    """Represents the state of a conversation."""
    session_id: str = Field(..., description="Unique identifier for the conversation session.")
    history: List[Dict[str, str]] = Field(default_factory=list, description="History of user/model messages.")
    # Example: [{'role': 'user', 'parts': ['Hi there!']}, {'role': 'model', 'parts': ['Hello! How can I help?']}]

    current_intent: Optional[str] = Field(None, description="The detected intent for the current turn.")
    last_agent: Optional[str] = Field(None, description="The last agent that handled the conversation.")
    extracted_entities: Dict[str, Any] = Field(default_factory=dict, description="Entities extracted during the conversation (e.g., {'order_id': 'xyz'}).")
    last_interaction_time: datetime.datetime = Field(default_factory=datetime.datetime.now, description="Timestamp of the last interaction.")

    # --- Pydantic V2 Configuration ---
    model_config = ConfigDict(
        # Allows Pydantic models to work with ORM objects if needed later
        # For now, primarily useful for validation on assignment
        from_attributes=True,
        # Optional: Add other configurations like validate_assignment if needed
        # validate_assignment=True
    )

    def add_message(self, role: str, text: str):
        """Adds a message to the history."""
        # Basic validation
        if role not in ['user', 'model']:
            raise ValueError("Role must be 'user' or 'model'")
        if not isinstance(text, str):
            raise TypeError("Text must be a string")

        self.history.append({"role": role, "parts": [text]})
        self.last_interaction_time = datetime.datetime.now()

    def get_history(self) -> List[Dict[str, str]]:
        """Returns the conversation history."""
        return self.history

    def update_state(self, intent: Optional[str] = None, agent: Optional[str] = None, entities: Optional[Dict[str, Any]] = None):
        """Updates the state after processing a turn."""
        if intent is not None:
            self.current_intent = intent
        if agent is not None:
            self.last_agent = agent
        if entities is not None:
            self.extracted_entities.update(entities) # Merge new entities
        self.last_interaction_time = datetime.datetime.now()

    def clear_transient_state(self):
        """Resets state elements that might be turn-specific."""
        self.current_intent = None
        # Decide if extracted_entities should persist across turns or be cleared
        # self.extracted_entities = {} # Uncomment if entities should reset each turn

    # --- Removed old class Config ---
    # class Config:
    #     from_attributes = True