# src/ui/gradio_app.py
import os
import logging
import asyncio
import gradio as gr
import uuid
from typing import Dict, Any, List, Tuple, Optional, AsyncIterator
import html

# Ensure imports happen relative to the root when run via app.py
try:
    from src.core.conversation import ConversationManager
    from src.llm.gemini_service import GeminiService
    from src.core.config import GOOGLE_API_KEY
except ModuleNotFoundError:
    # Handle case where script might be run directly
    import sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
    from src.core.conversation import ConversationManager
    from src.llm.gemini_service import GeminiService
    from src.core.config import GOOGLE_API_KEY

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# --- Service Initialization ---
conversation_manager = None
CONV_MANAGER_ERROR_MSG = "Error: The chatbot service is currently unavailable. Please try again later or contact support."
try:
    if not GOOGLE_API_KEY:
        logger.error("GOOGLE_API_KEY not found. Please set it in your .env file.")
        # Don't raise here, let the UI handle showing an error
    else:
        llm_service = GeminiService()
        conversation_manager = ConversationManager(llm_service=llm_service)
        logger.info("Successfully initialized LLM service and conversation manager.")
except Exception as e:
    logger.error(f"CRITICAL: Failed to initialize core services: {e}", exc_info=True)
    # conversation_manager remains None

# --- Constants ---
INITIAL_WELCOME_MESSAGE = "Welcome! How can I help you today?"
USER_PLACEHOLDER_MESSAGE = "..." # Placeholder while bot is thinking
DEFAULT_ERROR_MESSAGE = "I'm sorry, an internal error occurred. Please try again."
# Construct absolute paths for assets relative to this file's location
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.abspath(os.path.join(CURRENT_DIR, '..', '..', 'assets'))
BOT_AVATAR = os.path.join(ASSETS_DIR, "bot-icon.png") if os.path.exists(os.path.join(ASSETS_DIR, "bot-icon.png")) else None
USER_AVATAR = os.path.join(ASSETS_DIR, "user-icon.png") if os.path.exists(os.path.join(ASSETS_DIR, "user-icon.png")) else None
FAVICON_PATH = os.path.join(ASSETS_DIR, "favicon.ico") if os.path.exists(os.path.join(ASSETS_DIR, "favicon.ico")) else None

# --- Helper Functions ---

def generate_new_session_id() -> str:
    """Generates a new unique session ID."""
    return str(uuid.uuid4())

def get_initial_chat_history() -> List[Tuple[Optional[str], Optional[str]]]:
    """Returns the initial chat history structure for Gradio Chatbot."""
    return [(None, INITIAL_WELCOME_MESSAGE)]

async def handle_chat_interaction(
    message: str,
    history: List[Tuple[Optional[str], Optional[str]]],
    session_id: str
) -> AsyncIterator[Tuple[
        List[Tuple[Optional[str], Optional[str]]], # Updated History
        str,                                       # Updated Session ID
        gr.Textbox,                                # Textbox update
        gr.Button                                  # Button update
    ]]:
    """
    Handles user messages, interacts with the backend, and updates UI state. (Async Generator)
    """
    # 1. Input Validation & Sanitization
    if not message or not message.strip():
        return # Stop generator for empty messages

    sanitized_message = html.escape(message.strip())

    # 2. Ensure Session ID
    if not session_id:
        session_id = generate_new_session_id()
        logger.info(f"Started new session: {session_id}")

    # 3. Update History & UI for Loading State
    updated_history = history + [(sanitized_message, None)] # Add user message
    thinking_history = updated_history + [(None, USER_PLACEHOLDER_MESSAGE)] # Add placeholder

    # Disable input fields while processing
    ui_updates_processing = (
        thinking_history,
        session_id,
        gr.Textbox(value="", interactive=False, placeholder="Ari is thinking..."),
        gr.Button(interactive=False)
    )
    yield ui_updates_processing

    # 4. Check Conversation Manager Status
    if conversation_manager is None:
        logger.error(f"Session {session_id}: ConversationManager not available.")
        final_history = updated_history + [(None, CONV_MANAGER_ERROR_MSG)]
        ui_updates_final = (
            final_history,
            session_id,
            gr.Textbox(value="", interactive=True, placeholder="Type your message..."), # Re-enable
            gr.Button(interactive=True) # Re-enable
        )
        yield ui_updates_final
        return # Stop processing

    # 5. Process Message with Backend
    logger.info(f"Session {session_id}: Processing message: '{sanitized_message[:50]}...'")
    bot_response = DEFAULT_ERROR_MESSAGE
    try:
        response_data = await conversation_manager.handle_message(
            user_input=sanitized_message,
            session_id=session_id
        )
        bot_response = response_data.get("response", DEFAULT_ERROR_MESSAGE)
        session_id = response_data.get("session_id", session_id)
        logger.info(f"Session {session_id}: Bot response: '{bot_response[:100]}...'")

    except Exception as e:
        logger.error(f"Session {session_id}: Error during conversation_manager.handle_message: {e}", exc_info=True)

    # 6. Final UI Update with Bot Response
    final_history = updated_history + [(None, bot_response)]
    ui_updates_final = (
        final_history,
        session_id,
        gr.Textbox(value="", interactive=True, placeholder="Type your message..."), # Re-enable
        gr.Button(interactive=True) # Re-enable
    )
    yield ui_updates_final


def clear_chat_action() -> Tuple[List[Tuple[Optional[str], Optional[str]]], str, gr.Textbox]:
    """Clears the chat history and resets the session."""
    new_session_id = generate_new_session_id()
    logger.info(f"Chat cleared. New session ID: {new_session_id}")
    return get_initial_chat_history(), new_session_id, gr.Textbox(placeholder="Type your message...")


# --- UI Components ---

# Minimal CSS for responsiveness and slight adjustments
modern_css = """
/* Limit overall width on larger screens */
.gradio-container {
    max-width: 900px !important;
    margin-left: auto !important;
    margin-right: auto !important;
}

/* Ensure input row elements have some minimum width */
.input-row > * {
    min-width: 50px;
}

/* Mobile responsiveness: stack input row */
@media (max-width: 640px) {
    .input-row {
        flex-direction: column !important;
        gap: 0.5rem !important; /* Add gap when stacked */
    }
    .input-row > * {
        width: 100% !important; /* Make elements full width when stacked */
        min-width: unset !important;
    }
    /* Reduce padding on mobile */
    .gradio-container {
        padding: 1rem !important;
    }
    /* Adjust chatbot height slightly on mobile if needed */
    /* .chatbot-area { height: 70vh !important; } */
}

/* Style the clear button */
.clear-button-row {
    justify-content: center !important; /* Center the button */
    margin-top: 0.5rem !important;
}
"""

def create_modern_demo():
    """Creates a modern, responsive Gradio Chat Interface."""

    # Use a clean, modern theme - REMOVED the faulty .set() call
    theme = gr.themes.Soft(
        primary_hue=gr.themes.colors.blue,
        secondary_hue=gr.themes.colors.neutral,
        font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"]
    )

    with gr.Blocks(theme=theme, css=modern_css, title="Ari E-commerce Assistant", analytics_enabled=False) as demo:
        # State to store the unique session ID for the conversation
        session_state = gr.State(value=generate_new_session_id)

        # Optional: Simple Header
        with gr.Row():
             gr.Markdown("# Ari E-commerce Assistant", elem_id="app-title")
             # Could add logo here if desired: gr.Image(value=BOT_AVATAR, width=40, height=40, show_label=False, container=False)

        # Main Chat Area
        with gr.Column(elem_classes="chatbot-area"): # Added class for potential targeting
            chatbot = gr.Chatbot(
                value=get_initial_chat_history,
                label="Chat History",
                bubble_full_width=False,
                height=600, # Max height constraint
                avatar_images=(USER_AVATAR, BOT_AVATAR),
                show_copy_button=True,
                layout="bubble"
                # render=False # Only needed if placing it manually later
            )

        # Input Area
        with gr.Row(elem_classes="input-row"): # Added class for targeting
            msg_textbox = gr.Textbox(
                placeholder="Type your message...",
                label="Your Message",
                show_label=False,
                scale=5, # Give textbox more space
                autofocus=True,
                lines=1,
                max_lines=5,
                container=False
            )
            submit_button = gr.Button("Send", variant="primary", scale=1) # Send button takes less space

        # Clear Button Area
        with gr.Row(elem_classes="clear-button-row"): # Added class for targeting
            clear_button = gr.Button("🗑️ Clear Conversation", variant="secondary", size="sm")


        # --- Event handlers ---

        # Combine Textbox submit and Button click handlers
        submit_triggers = [msg_textbox.submit, submit_button.click]
        chat_outputs = [chatbot, session_state, msg_textbox, submit_button]

        for trigger in submit_triggers:
            trigger(
                fn=handle_chat_interaction,
                inputs=[msg_textbox, chatbot, session_state],
                outputs=chat_outputs,
                # queue=True # Handled by demo.queue() later
            )

        # Clear button action
        clear_button.click(
            fn=clear_chat_action,
            inputs=None,
            outputs=[chatbot, session_state, msg_textbox],
            queue=False # Clearing is fast
        )

    return demo

# --- Launch App ---

def main():
    """Main function to launch the Gradio app."""
    if conversation_manager is None:
         logger.warning("Conversation Manager failed to initialize. Chatbot will show an error message.")

    logger.info("Creating Gradio demo...")
    demo = create_modern_demo()

    # Enable queue for handling multiple users/requests gracefully
    demo.queue()

    logger.info("Launching Gradio demo...")
    try:
        demo.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=False, # Set to True for a public link (e.g., for Hugging Face Spaces)
            debug=False,
            max_threads=20,
            favicon_path=FAVICON_PATH # Use constant defined earlier
        )
        logger.info("Gradio demo launched. Access locally via http://localhost:7860 or http://127.0.0.1:7860")
    except Exception as e:
        logger.error(f"Failed to launch Gradio demo: {e}", exc_info=True)
        print(f"Error launching Gradio: {e}")

if __name__ == "__main__":
    main()