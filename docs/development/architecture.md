### `docs/development/architecture.md`


# Project Architecture

This document outlines the high-level architecture of the Ari chatbot.

## Core Components

*   **`app.py`:** The main entry point for the application. It initializes services and launches the Gradio UI.
*   **`src/ui/gradio_app.py`:** Defines the Gradio web interface, handles UI events, manages chat state display, and interacts with the `ConversationManager`.
*   **`src/core/conversation.py` (`ConversationManager`):** Orchestrates the conversation flow. It manages conversation state, uses the LLM service to determine user intent, routes requests to appropriate agents, injects required services into agents, and formats final responses.
*   **`src/core/state.py` (`ConversationState`):** Manages the state of a single user conversation session, primarily storing message history.
*   **`src/agents/`:** Contains specialized agents inheriting from `BaseAgent`. Each agent handles a specific user intent (e.g., `OrderStatusAgent`, `ReturnPolicyAgent`, `HumanRepAgent`). Agents receive necessary services via dependency injection from the `ConversationManager`.
    *   **`base_agent.py` (`BaseAgent`):** Abstract base class defining the agent interface (`process` method) and specifying required services (`get_required_service_keys`).
*   **`src/llm/gemini_service.py` (`GeminiService`):** Implementation for interacting with the Google Gemini API. Primarily used by the `ConversationManager` for intent detection based on user input and conversation history.
*   **`src/services/`:** Contains modules encapsulating business logic or interactions with external resources/data.
    *   **`order_service.py` (`OrderService`):** Handles database queries related to orders (e.g., fetching status by ID) using SQLAlchemy.
    *   **`policy_service.py` (`PolicyService`):** Reads and provides information from the `data/policies.json` file.
    *   *(Other services like `contact_service.py` might exist)*
*   **`src/db/`:** Database interaction layer using SQLAlchemy.
    *   **`database.py`:** Manages database sessions.
    *   **`models.py`:** Defines SQLAlchemy ORM models (e.g., `Order`).
    *   **`setup_db.py`:** Script to initialize the database schema.
*   **`src/utils/helpers.py`:** Contains utility functions used across the application (e.g., `extract_order_id`).
*   **`data/`:** Stores data files.
    *   `policies.json`: Contains return policy information.
    *   `chatbot_data.db`: Default SQLite database file (if used).
*   **`assets/`:** Stores static assets for the UI (icons, images).

## Data Flow (Simplified)

1.  User sends a message via the Gradio UI (`gradio_app.py`).
2.  The UI's event handler calls `ConversationManager.handle_message()`.
3.  `ConversationManager` retrieves or creates `ConversationState` for the session.
4.  `ConversationManager` calls `GeminiService` to determine the user's intent.
5.  `ConversationManager` identifies the appropriate `Agent` based on the intent.
6.  `ConversationManager` injects required `Services` (e.g., `OrderService`, `PolicyService`) into the selected `Agent`.
7.  `ConversationManager` calls the `Agent.process()` method.
8.  The `Agent` executes its logic:
    *   It might call helper functions (`utils`).
    *   It interacts with its injected `Services` (e.g., `agent.order_service.get_order_status_by_id(...)`).
    *   `Services` interact with the database (`db`) or data files (`data`).
9.  The `Agent` returns a response string to the `ConversationManager`.
10. `ConversationManager` updates the `ConversationState` (history) and returns the response data to the Gradio UI.
11. Gradio UI (`gradio_app.py`) updates the chat display.