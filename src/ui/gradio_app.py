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
:root {
    --primary: #6366f1;
    --primary-dark: #4f46e5;
    --text: white; /* Changed to white */
    --text-light: #cccccc; /* Light gray for secondary text */
    --glass: rgba(0, 0, 0, 0.25); /* Dark translucent glass */
    --glass-border: rgba(255, 255, 255, 0.1); /* Subtle light border */
    --glass-shadow: rgba(0, 0, 0, 0.1);
    --glass-highlight: rgba(255, 255, 255, 0.4);
    --input-bg: rgba(0, 0, 0, 0.2); /* Darker input background */
    --input-border: rgba(255, 255, 255, 0.15);
    --input-focus-bg: rgba(0, 0, 0, 0.3);
    --input-focus-border: var(--primary); /* Keep primary focus border */
    --input-focus-shadow: rgba(99, 102, 241, 0.2); /* Adjust focus shadow */
    --bubble-bot-bg: rgba(255, 255, 255, 0.7);
    --bubble-user-bg: rgba(99, 102, 241, 0.2);
}

body {
    background: linear-gradient(135deg, #1a202c, #2d3748) !important; /* Dark gradient */
    font-family: 'Inter', sans-serif !important;
    color: var(--text) !important; /* Ensure body text defaults to black */
}

/* --- Base Styles (Simplified) --- */
/* Removed .glass-card base as we are simplifying */

/* Optional: Add hover effect if desired (Can be applied selectively)
.element-with-hover:hover {
    transform: translateY(-5px);
    box-shadow: 0 12px 40px 0 rgba(0, 0, 0, 0.15) !important;
} */

/* --- Glass Input Base --- */
.glass-input {
    background: var(--input-bg) !important;
    backdrop-filter: blur(5px) !important;
    -webkit-backdrop-filter: blur(5px) !important;
    border: 1px solid var(--input-border) !important;
    transition: all 0.3s ease !important;
    border-radius: 0.75rem !important; /* 12px */
}
.glass-input:focus, .glass-input:focus-within { /* Apply to container on focus */
    background: var(--input-focus-bg) !important;
    border-color: var(--input-focus-border) !important;
    box-shadow: 0 0 0 3px var(--input-focus-shadow) !important;
}

/* --- Glass Button Base --- */
.glass-button {
    background: linear-gradient(135deg, var(--primary), var(--primary-dark)) !important;
    color: white !important;
    border: none !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1) !important;
    border-radius: 0.75rem !important; /* 12px */
    padding: 0.6rem 1.2rem !important; /* Adjust padding */
    font-weight: 500 !important;
}
.glass-button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15) !important;
    background: linear-gradient(135deg, var(--primary-dark), var(--primary)) !important;
}
.glass-button:active {
    transform: translateY(0) !important;
}

/* --- Gradio Specific Styling --- */

/* Main Container - Simplified for a single outline */
.gradio-container {
    max-width: 1000px !important; /* Slightly reduced width */
    margin: 3rem auto !important;
    padding: 2rem 2.5rem !important;
    border-radius: 1.5rem !important;
    box-shadow: 0 10px 40px rgba(0, 0, 0, 0.3) !important; /* Slightly stronger shadow */
    background: var(--glass) !important; /* Dark glass */
    backdrop-filter: blur(12px) !important;
    -webkit-backdrop-filter: blur(12px) !important;
    border: 1px solid var(--glass-border) !important; /* Light border on dark glass */
    overflow: visible !important; /* Allow shadow to show */
}

/* Hero Section (Keep distinct background, adjust padding) */
.hero-section {
    background: transparent !important; /* Make hero background transparent */
    backdrop-filter: none !important;
    -webkit-backdrop-filter: none !important;
    padding: 0 0 2rem 0 !important; /* Padding only at the bottom */
    text-align: center;
    border-bottom: 1px solid var(--glass-border) !important; /* Use glass border for separator */
}
.hero-section h1 {
    font-size: 2.25rem !important;
    font-weight: 600 !important;
    color: var(--text) !important; /* White text */
    margin-bottom: 0.5rem !important;
    /* Optional: Gradient text (removed for black text request) */
    /* background: linear-gradient(135deg, var(--primary), var(--primary-dark));
    -webkit-background-clip: text;
    background-clip: text;
    color: transparent !important; */
}
.hero-section p {
    font-size: 1.1rem !important;
    color: var(--text-light) !important; /* Light gray text */
    margin-bottom: 0 !important;
    max-width: 600px;
    margin-left: auto;
    margin-right: auto;
}

/* Chat Container (Holds Chatbot, Input, Clear) */
.chat-container {
    padding: 1.5rem 0 0 0 !important; /* Padding only at the top */
}

/* Chatbot Area (The message history box) */
.chatbot-area > .wrap { /* Ensure chat history area is visually gone */
    background: transparent !important;
    backdrop-filter: none !important;
    -webkit-backdrop-filter: none !important;
    border: none !important;
    border-radius: 0 !important;
    padding: 0 !important; /* Remove padding */
    box-shadow: none !important;
    margin: 0 !important; /* Remove margin */
}
.chatbot-area .message-wrap { /* Container for bubbles */
    background: transparent !important; /* Make inner container transparent */
}

/* Chat Bubbles */
.chatbot-area .message {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    padding: 0 !important;
    margin: 0.5rem 0 !important;
    max-width: 100% !important; /* Allow text to span wider */
    border-radius: 0 !important; /* No radius on the message container */
}
.chatbot-area .message > div:first-child { /* Style the text content block */
    padding: 0.5rem 0.8rem !important; /* Reduced padding */
    border-radius: 8px !important; /* Subtle rounding for text block */
    backdrop-filter: none !important; /* No blur on text block */
    -webkit-backdrop-filter: none !important;
    transition: none !important; /* No transition */
    background: transparent !important; /* Make bubble background transparent */
    border: none !important; /* Remove bubble border */
    box-shadow: none !important; /* Remove bubble shadow */
    color: var(--text) !important; /* Ensure text is white */
    display: inline-block; /* Allow text block to size naturally */
    max-width: 80%; /* Limit width of the text block itself */
}
/* Remove specific user/bot bubble backgrounds/borders */
.chatbot-area .message.user > div:first-child {
    /* background: transparent !important; */ /* Already set above */
    margin-left: auto !important;
    /* border: none !important; */ /* Already set above */
    text-align: right; /* Align text right for user */
}
.chatbot-area .message.bot > div:first-child {
    /* background: transparent !important; */ /* Already set above */
    margin-right: auto !important;
    /* border: none !important; */ /* Already set above */
    text-align: left; /* Align text left for bot */
}
/* Adjust avatar spacing if needed */
.chatbot-area .message .avatar-container { margin: 0 0.5rem; }

/* Input Row */
.input-row {
    gap: 0.75rem !important; /* Increase gap slightly */
    align-items: stretch !important; /* Align items top */
    margin-top: 1rem !important;
}
.input-row .gr-textbox { /* Container for textarea */
    flex-grow: 1 !important;
    border-radius: 0.75rem !important; /* Match glass-input */
    background: transparent !important; /* Make container transparent */
    border: none !important; /* Remove container border */
    padding: 0 !important; /* Remove container padding */
}
.input-row .gr-textbox textarea {
    background: var(--input-bg) !important;
    backdrop-filter: blur(5px) !important;
    -webkit-backdrop-filter: blur(5px) !important;
    border: 1px solid var(--input-border) !important;
    transition: all 0.3s ease !important;
    border-radius: 0.75rem !important; /* 12px */
    padding: 0.8rem 1rem !important; /* Adjust padding */
    min-height: 50px !important; /* Ensure decent height */
    color: var(--text) !important; /* Ensure input text is white */
}
.input-row .gr-textbox textarea:focus {
    background: var(--input-focus-bg) !important;
    border-color: var(--input-focus-border) !important;
    box-shadow: 0 0 0 3px var(--input-focus-shadow) !important;
    outline: none !important;
}

/* Send Button */
.input-row .gr-button.primary {
    background: linear-gradient(135deg, var(--primary), var(--primary-dark)) !important;
    color: white !important;
    border: none !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1) !important;
    border-radius: 0.75rem !important; /* 12px */
    padding: 0.8rem 1.2rem !important; /* Match textarea padding */
    font-weight: 500 !important;
    align-self: stretch !important; /* Make button full height of row */
    min-width: 100px !important;
}
.input-row .gr-button.primary:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15) !important;
    background: linear-gradient(135deg, var(--primary-dark), var(--primary)) !important;
}
.input-row .gr-button.primary:active {
    transform: translateY(0) !important;
}

/* Clear Button */
.clear-button-row {
    justify-content: center !important;
    margin-top: 1.5rem !important;
    padding-bottom: 0.5rem !important; /* Add padding below clear button */
}
.clear-button-row .gr-button {
    background: rgba(0, 0, 0, 0.03) !important; /* Very subtle background */
    backdrop-filter: none !important;
    -webkit-backdrop-filter: none !important;
    border: 1px solid rgba(0, 0, 0, 0.08) !important; /* Subtle border */
    color: var(--text-light) !important; /* Light gray text for clear button */
    border-radius: 0.75rem !important; /* 12px */
    padding: 0.5rem 1rem !important;
    font-size: 0.9rem !important;
    font-weight: 500 !important;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05) !important;
    transition: all 0.3s ease !important;
}
.clear-button-row .gr-button:hover {
    background: rgba(0, 0, 0, 0.06) !important; /* Slightly darker on hover */
    border-color: rgba(0, 0, 0, 0.12) !important;
    color: var(--text) !important; /* White text on hover */
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.08) !important;
}

/* Mobile Adjustments */
@media (max-width: 640px) {
    .gradio-container {
        margin: 1rem !important; /* Adjust margin for mobile */
        padding: 1rem 1.5rem !important; /* Adjust padding for mobile */
        border-radius: 1rem !important;
    }
    .hero-section {
        padding: 2rem 1rem 1.5rem 1rem !important;
    }
    .hero-section h1 { font-size: 1.8rem !important; }
    .hero-section p { font-size: 1rem !important; }
    .chat-container { padding: 1rem !important; }
    .input-row {
        flex-direction: column !important;
        gap: 0.5rem !important;
        align-items: stretch !important;
    }
    .input-row .gr-button.primary {
        min-width: unset !important;
        width: 100% !important;
    }
    .chatbot-area .message { max-width: 90% !important; }
}
"""

def create_modern_demo() -> gr.Blocks:
    """
    Creates the Gradio Blocks UI for the chatbot.

    Returns:
        gr.Blocks: The configured Gradio Blocks instance.
    """

    # Use a clean, modern theme - REMOVED the faulty .set() call
    # Update theme to use Inter font primarily and adjust radius
    theme = gr.themes.Soft(
        primary_hue=gr.themes.colors.indigo, # Match --primary color better
        secondary_hue=gr.themes.colors.blue,
        neutral_hue=gr.themes.colors.slate,
        font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"],
        radius_size=gr.themes.sizes.radius_lg # Use larger radius globally from theme
    ).set(
        # Override specific theme elements if needed, but CSS is more direct
        # button_primary_background_fill='*primary', # Example
        # input_background_fill='*neutral_50' # Example
    )

    # --- UI Layout Definition ---
    with gr.Blocks(theme=theme, css=modern_css, title="Ari E-commerce Assistant", analytics_enabled=False) as demo:
        # State to store the unique session ID for the conversation
        session_state = gr.State(value=generate_new_session_id)

        # --- Hero Section --- (Displays title and subtitle)
        with gr.Column(elem_classes="hero-section"):
            gr.Markdown("# Ari E-commerce Assistant", elem_id="app-title") # Reusing ID for now, but styled by class
            gr.Markdown("Your friendly AI-powered helper for all things e-commerce.", elem_id="app-subtitle") # Added subtitle

        # --- Main Chat Interface Area --- (Contains Chatbot, Input, Clear Button)
        with gr.Column(elem_classes="chat-container"):
            # Chat History Display
            with gr.Column(elem_classes="chatbot-area"):
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

            # User Input Section
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

            # Conversation Control Buttons
            with gr.Row(elem_classes="clear-button-row"): # Added class for targeting
                clear_button = gr.Button("🗑️ Clear Conversation", variant="secondary", size="sm")

        # --- Event Handlers --- (Connects UI elements to backend functions)

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

def main() -> None:
    """
    Main function to initialize services and launch the Gradio app.
    """
    # Check if core services initialized correctly
    if conversation_manager is None:
         logger.warning("Conversation Manager failed to initialize. Chatbot UI will show an error message.")

    # Create the Gradio interface
    logger.info("Creating Gradio demo...")
    demo = create_modern_demo()

    # Enable Gradio queue for handling multiple simultaneous users/requests gracefully
    demo.queue()

    # Launch the Gradio app server
    logger.info("Launching Gradio demo server...")
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