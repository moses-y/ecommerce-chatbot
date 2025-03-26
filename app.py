# app.py: Modern Gradio interface for the E-commerce Support Chatbot
"""
Modern Gradio interface for the E-commerce Support Chatbot
This version applies a single theme with a very light blue glassmorphic background,
uses Calibri 12pt fonts for all text, and fixes container paddings/borders to 0.
All UI regions (sidepanel, main message area) share the same height.
"""

import os
import sys
from datetime import datetime
import time
import json
import bleach
import gradio as gr
from dotenv import load_dotenv
from typing import Dict, List, Tuple, Any
from logging import getLogger, basicConfig, StreamHandler
import logging

# Configure logging
basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = getLogger(__name__)
logger.addHandler(StreamHandler())

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
load_dotenv()

# Initialize credentials
def initialize_credentials():
    """Initialize Google credentials from environment variables."""
    try:
        credentials_json = os.getenv("GOOGLE_CREDENTIALS")
        if credentials_json:
            os.makedirs("/home/user/app", exist_ok=True)
            credentials_path = "/home/user/app/google_credentials.json"
            with open(credentials_path, "w") as f:
                f.write(credentials_json)
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_path
            logger.info("Successfully created credentials file at: %s", credentials_path)
            return True
    except Exception as e:
        logger.error("Error creating credentials file: %s", e)
        return False

if not initialize_credentials():
    logger.error("Failed to initialize credentials")
    sys.exit(1)

# Import required modules from our application.
try:
    from src.utils import load_order_data
    from src.vector_db import get_vector_db_instance
    from src.chatbot import chat_with_user
except ImportError as e:
    logger.error("Error importing required modules: %s", e)
    raise

from src.llm_service import LLMService
from src.state_management import ConversationMemory
from src.config import CONVERSATION_CONFIG, FAQ_CONFIG
from src.state_utils import reset_state
from src.credentials import verify_credentials

# Initialize services with credential verification.
def initialize_services():
    """Initialize required services with proper credential verification."""
    logger.info("Initializing services...")
    start = time.time()
    try:
        cred_results = verify_credentials(["GOOGLE_API_KEY", "GOOGLE_APPLICATION_CREDENTIALS", "HUGGINGFACE_TOKEN"])
        if not all(cred_results.values()):
            missing = [k for k, v in cred_results.items() if not v]
            raise EnvironmentError(f"Missing required credentials: {missing}")
        llm_service = LLMService()
        conversation_memory = ConversationMemory(max_history=CONVERSATION_CONFIG["max_history_length"])
        orders_df = load_order_data(use_cache=True)
        vector_collection = get_vector_db_instance(orders_df, use_subset=False)
        logger.info("Services initialized in %.2f seconds", time.time() - start)
        return llm_service, conversation_memory, vector_collection
    except Exception as e:
        logger.error("Failed to initialize services: %s", e)
        raise

llm_service, conversation_memory, vector_collection = initialize_services()

# ===== UI Configuration =====

# Define Gradio theme with a light blue glassmorphic background for the entire window.
# Global custom CSS enforces:
#    - Font family: Calibri
#    - Font size: 12px
#    - Zero container paddings and borders (ensuring consistent heights)
css = """
:root {
  font-size: 12px;
  font-family: "Calibri", sans-serif;
  --bg-glass: rgba(224, 247, 250, 0.4);
  --bg-glass-hover: rgba(224, 247, 250, 0.5);
  --border: 0;
}

/* Global reset for containers */
* {
  margin: 0;
  padding: 0;
  box-sizing: border-box;
}

/* Glassmorphism effect */
.glass {
  background: var(--bg-glass);
  backdrop-filter: blur(10px);
  border: 0;
}

/* Chat container & side panel set to same height */
.chat-container, .sidebar {
  height: 100%;
}

/* Chatbot display area */
#chatbot {
  height: 100%;
  overflow-y: auto;
}

/* Input area style */
.input-area {
  display: flex;
  align-items: center;
  height: 50px;
}
.input-area input {
  flex-grow: 1;
  font-size: 12px;
  padding: 0 8px;
  border: 0;
}
.input-area button {
  font-size: 12px;
  border: 0;
  cursor: pointer;
}

/* Sidebar style */
.sidebar {
  padding: 0;
}
"""

# Define a simple Gradio theme (all regions use a consistent light blue frosted look).
theme = gr.themes.Default(
    body_background_fill="rgba(224, 247, 250, 0.4)",
    block_background_fill="rgba(224, 247, 250, 0.4)",
    button_primary_background_fill="rgba(224, 247, 250, 0.8)",
    button_primary_text_color="white"
)

# ===== Helper Functions =====

def process_message(message: str, history: List, state: Dict[str, Any], order_id: str = None) -> Tuple[List, Dict[str, Any]]:
    """Process a user message and generate a response using chat_with_user."""
    try:
        cred_results = verify_credentials(["GOOGLE_API_KEY", "GOOGLE_APPLICATION_CREDENTIALS"])
        if not all(cred_results.values()):
            raise EnvironmentError("Missing required credentials")
        message = bleach.clean(message)
        if order_id and order_id.strip():
            message = f"What's the status of my order {order_id.strip()}?"
        if not state.get("messages"):
            state["messages"] = []
        state["messages"].append({"role": "user", "content": message})
        # Generate response using chat_with_user
        updated_state = chat_with_user(message, state)
        return updated_state["messages"], updated_state
    except Exception as e:
        logger.error("Error in process_message: %s", e)
        state["messages"].append({"role": "assistant", "content": "I encountered an error processing your message."})
        return state["messages"], state

def clear_conversation(state: Dict[str, Any]) -> Tuple[List, Dict[str, Any]]:
    """Clear current conversation state."""
    new_state = reset_state()
    new_state["session_id"] = datetime.now().strftime("%Y%m%d%H%M%S")
    new_state["messages"] = []
    new_state["chat_history"] = []
    return [], new_state

def submit_feedback(feedback: str, state: Dict[str, Any]) -> Dict[str, Any]:
    """Handle feedback submission."""
    if not feedback:
        return state
    new_state = state.copy()
    new_state["feedback"] = feedback
    try:
        os.makedirs("data/feedback", exist_ok=True)
        feedback_file = f"data/feedback/feedback_{new_state['session_id']}.json"
        feedback_data = {
            "session_id": new_state["session_id"],
            "timestamp": datetime.now().isoformat(),
            "feedback": feedback,
            "conversation": new_state.get("chat_history", [])
        }
        with open(feedback_file, "w") as f:
            json.dump(feedback_data, f, indent=2)
        logger.info("Feedback saved to %s", feedback_file)
    except Exception as e:
        logger.error("Error saving feedback: %s", e)
    return new_state

# ===== Gradio Interface Construction =====
with gr.Blocks(theme=theme, css=css) as demo:
    state = gr.State(value=reset_state())
    
    with gr.Row():
        with gr.Column(scale=1, elem_classes="sidebar glass"):
            gr.Markdown(
                """
                ### E-commerce Support Assistant
                """,
                elem_id="sidebar-header"
            )
            gr.Markdown("#### Quick Links")
            with gr.Row():
                faq_btn = gr.Button("FAQs")
                contact_btn = gr.Button("Contact")
            gr.Markdown("#### Track Your Order")
            order_input = gr.Textbox(placeholder="Enter order ID", show_label=False)
            track_btn = gr.Button("Track Order", variant="primary")
            gr.Markdown("#### FAQs")
            with gr.Accordion("Frequently Asked Questions", open=False):
                gr.Markdown("""
                **What is your return policy?**  
                Items can be returned within 30 days.

                **How long does shipping take?**  
                Standard shipping takes 5-7 business days.

                **What payment methods do you accept?**  
                We accept credit cards, PayPal, and Apple Pay.
                """)
            feedback = gr.Textbox(placeholder="Your feedback...", label="Feedback", lines=3)
            submit_fb = gr.Button("Submit Feedback", variant="secondary")
    
        with gr.Column(scale=3, elem_classes="chat-container glass"):
            chatbot = gr.Chatbot(elem_id="chatbot", height=500)
            with gr.Row(classes="input-area"):
                msg = gr.Textbox(placeholder="Type your message...", show_label=False)
                send_btn = gr.Button("Send", variant="primary")
    
    send_btn.click(fn=process_message, inputs=[msg, chatbot, state],
                   outputs=[chatbot, state]).then(lambda: "", None, msg)
    
    msg.submit(fn=process_message, inputs=[msg, chatbot, state],
               outputs=[chatbot, state]).then(lambda: "", None, msg)
    
    track_btn.click(fn=process_message, inputs=[order_input, chatbot, state],
                    outputs=[chatbot, state]).then(lambda: "", None, order_input)
    
    submit_fb.click(fn=submit_feedback, inputs=[feedback, state],
                    outputs=[state]).then(lambda: "", None, feedback)

# ===== Health Check and Server Config =====
def configure_logging():
    try:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[logging.StreamHandler()]
        )
        logger.info("Logging configured")
        return True
    except Exception as e:
        print(f"Failed to configure logging: {e}")
        return False

def get_server_config():
    return {
        "server_name": os.getenv("SERVER_HOST", "0.0.0.0"),
        "server_port": int(os.getenv("SERVER_PORT", 7860)),
        "share": os.getenv("ENABLE_SHARE", "false").lower() == "true",
        "auth": None if os.getenv("AUTH_REQUIRED", "false").lower() == "false" else (os.getenv("AUTH_USERNAME", "admin"), os.getenv("AUTH_PASSWORD", "admin")),
        "ssl_keyfile": os.getenv("SSL_KEYFILE", None),
        "ssl_certfile": os.getenv("SSL_CERTFILE", None),
        "ssl_keyfile_password": os.getenv("SSL_KEYFILE_PASSWORD", None)
    }

def health_check():
    try:
        cred_results = verify_credentials(["GOOGLE_API_KEY", "GOOGLE_APPLICATION_CREDENTIALS", "HUGGINGFACE_TOKEN"])
        if not all(cred_results.values()):
            missing = [k for k, v in cred_results.items() if not v]
            logger.error("Missing credentials: %s", missing)
            return {"status": "unhealthy", "message": f"Missing credentials: {missing}", "timestamp": datetime.utcnow().isoformat()}
        # Run a dummy LLM query
        test_state = reset_state()
        test_state["messages"] = [{"role": "user", "content": "test"}]
        llm_service.generate_response(test_state["messages"], conversation_memory)
        # Run a dummy vector query
        test_vector = [0.0] * 384
        vector_collection.query(query_embeddings=[test_vector], n_results=1)
        return {"status": "healthy", "message": "All systems operational", "timestamp": datetime.utcnow().isoformat()}
    except Exception as e:
        logger.error("Health check failed: %s", e)
        return {"status": "unhealthy", "message": str(e), "timestamp": datetime.utcnow().isoformat()}

def initialize_app():
    logger.info("Initializing application...")
    try:
        cred_results = verify_credentials(["GOOGLE_API_KEY", "GOOGLE_APPLICATION_CREDENTIALS", "HUGGINGFACE_TOKEN"])
        if not all(cred_results.values()):
            missing = [k for k, v in cred_results.items() if not v]
            raise EnvironmentError(f"Missing credentials: {missing}")
        if health_check()["status"] != "healthy":
            raise RuntimeError("Health check failed")
        logger.info("Application initialized successfully")
        return True
    except Exception as e:
        logger.error("Application initialization failed: %s", e, exc_info=True)
        raise

if __name__ == "__main__":
    try:
        if not configure_logging():
            print("Warning: Using default logging configuration")
        logger.info("Starting E-commerce Support Assistant...")
        initialize_app()
        server_conf = get_server_config()
        demo.launch(
            server_name=server_conf["server_name"],
            server_port=server_conf["server_port"],
            share=server_conf["share"],
            auth=server_conf["auth"],
            ssl_keyfile=server_conf["ssl_keyfile"],
            ssl_certfile=server_conf["ssl_certfile"],
            ssl_keyfile_password=server_conf["ssl_keyfile_password"],
            max_threads=40,
            show_error=True,
            root_path="",
            favicon_path="assets/favicon.ico",
            allowed_paths=["assets"],
            blocked_paths=["data", "logs", "config"]
        )
        logger.info("Application launched successfully")
    except Exception as e:
        print(f"Critical error: {e}")
        sys.exit(1)
    finally:
        logger.info("Shutting down application...")
