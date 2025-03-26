# app.py: Modern Gradio interface for the E-commerce Support Chatbot
import os
import sys
from datetime import datetime

print("===== Application Startup at", datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"), "=====")


# ==== Initialize_Credentials ====

def initialize_credentials():
    """Initialize Google credentials from environment variables"""
    try:
        credentials_json = os.getenv('GOOGLE_CREDENTIALS')
        if credentials_json:
            os.makedirs('/home/user/app', exist_ok=True)
            credentials_path = '/home/user/app/google_credentials.json'
            with open(credentials_path, 'w') as f:
                f.write(credentials_json)
            os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credentials_path
            print("Successfully created credentials file at:", credentials_path)
            return True
    except Exception as e:
        print(f"Error creating credentials file: {e}")
        return False

if not initialize_credentials():
    print("Failed to initialize credentials")
    sys.exit(1)

import time
import json
import bleach
import gradio as gr
from dotenv import load_dotenv
from typing import Dict, List, Tuple, Any
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
load_dotenv()

try:
    from src.utils import load_order_data
    from src.vector_db import get_vector_db_instance
    from src.chatbot import chat_with_user
except ImportError as e:
    print(f"Error importing required modules: {e}")
    raise

from src.llm_service import LLMService
from src.state_management import ConversationMemory
from src.config import CONVERSATION_CONFIG, FAQ_CONFIG
from src.state_utils import reset_state, update_state_from_result
from src.credentials import verify_credentials


# ===== Initialize services with credential verification =====

def initialize_services():
    """Initialize all required services with proper credential verification."""
    logger.info("Initializing services...")
    start_time = time.time()
    try:
        cred_results = verify_credentials([
            "GOOGLE_API_KEY",
            "GOOGLE_APPLICATION_CREDENTIALS",
            "HUGGINGFACE_TOKEN"
        ])
        if not all(cred_results.values()):
            missing_creds = [k for k, v in cred_results.items() if not v]
            raise EnvironmentError(f"Missing required credentials: {missing_creds}")
        llm_service = LLMService()
        conversation_memory = ConversationMemory(
            max_history=CONVERSATION_CONFIG["max_history_length"]
        )
        orders_df = load_order_data(use_cache=True)
        vector_collection = get_vector_db_instance(orders_df, use_subset=False)
        logger.info(f"Services initialized in {time.time() - start_time:.2f} seconds")
        return llm_service, conversation_memory, vector_collection
    except Exception as e:
        logger.error(f"Failed to initialize services: {e}")
        raise

try:
    llm_service, conversation_memory, vector_collection = initialize_services()
except Exception as e:
    logger.error(f"Critical error during service initialization: {e}")
    raise

# ===== UI Configuration =====

# Define a modern theme with light brown and light blue tones using glassmorphism effects
theme = gr.themes.Soft(
    primary_hue="#A1887F",          # light brown
    secondary_hue="#81D4FA",        # light blue
    neutral_hue="slate",
    font=["Inter", "SF Pro Display", "system-ui", "sans-serif"],
    radius_size=gr.themes.sizes.radius_sm,
).set(
    body_background_fill="linear-gradient(135deg, #F5F5DC 0%, #E0F7FA 100%)",  # beige to light blue
    background_fill_primary="rgba(245, 245, 220, 0.4)",   # light brown glass effect
    background_fill_secondary="rgba(224, 247, 250, 0.35)",  # light blue glass effect
    border_color_primary="rgba(255, 255, 255, 0.2)",
    button_primary_background_fill="rgba(161, 136, 127, 0.8)",
    button_primary_background_fill_hover="rgba(161, 136, 127, 1)",
    button_primary_text_color="white",
    button_secondary_background_fill="rgba(224, 247, 250, 0.4)",
    button_secondary_background_fill_hover="rgba(224, 247, 250, 0.8)",
    button_secondary_text_color="white",
    block_title_text_color="rgba(80, 80, 80, 0.9)",
    block_label_text_color="rgba(80, 80, 80, 0.7)",
    input_background_fill="rgba(255, 255, 255, 0.6)",
)

# Custom CSS with minimalist adjustments and glassmorphism touch
css = """
/* Global styles */
:root {
    --primary: #A1887F;
    --primary-light: rgba(161,136,127,0.3);
    --secondary: #81D4FA;
    --text-primary: #403E3D;
    --text-secondary: #737373;
    --bg-glass: rgba(245, 245, 220, 0.4);
    --bg-glass-hover: rgba(245, 245, 220, 0.5);
    --border-glass: rgba(255, 255, 255, 0.2);
    --shadow-glass: 0 4px 6px rgba(0, 0, 0, 0.1);
}

/* Glassmorphism effect for panels */
.glass-panel {
    background: var(--bg-glass);
    backdrop-filter: blur(10px);
    border: 1px solid var(--border-glass);
    border-radius: 12px;
    box-shadow: var(--shadow-glass);
}
.glass-panel:hover {
    background: var(--bg-glass-hover);
}

/* Layout: Chat container, sidebar and input area */
.chat-container {
    height: calc(100vh - 160px);
    display: flex;
    flex-direction: column;
    overflow: hidden;
}

/* Chat display area */
#chatbot {
    flex: 1;
    overflow-y: auto;
    padding: 1rem;
    border-radius: 12px;
    background: var(--bg-glass);
    backdrop-filter: blur(10px);
    border: 1px solid var(--border-glass);
    box-shadow: var(--shadow-glass);
}

/* Input container with send button */
.input-area {
    display: flex;
    gap: 8px;
    padding: 16px;
    background: var(--bg-glass);
    border-radius: 12px;
    border: 1px solid var(--border-glass);
    margin-top: 8px;
}
.input-area input {
    flex: 1;
    padding: 12px 16px;
    font-size: 16px;
    border: 1px solid var(--border-glass);
    border-radius: 8px;
    background: rgba(255, 255, 255, 0.6);
    color: var(--text-primary);
}
.input-area button {
    padding: 12px 20px;
    font-weight: 500;
    border: none;
    border-radius: 8px;
    background: var(--primary);
    color: white;
    cursor: pointer;
    transition: all 0.2s ease;
}
.input-area button:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
}

/* Sidebar with query history (minimalist) */
.sidebar {
    background: var(--bg-glass);
    backdrop-filter: blur(10px);
    border-right: 1px solid var(--border-glass);
    padding: 10px;
    display: flex;
    flex-direction: column;
    gap: 10px;
}
.sidebar-header {
    display: flex;
    align-items: center;
    gap: 10px;
}
.sidebar-header h2 {
    margin: 0;
    font-size: 16px;
    color: var(--text-primary);
}

/* Responsive design */
@media (max-width: 768px) {
    .sidebar {
        display: none;
    }
    .chat-container {
        height: calc(100vh - 120px);
    }
}
"""

# ===== Helper Functions =====

def process_message(message: str, history: List, state: Dict[str, Any], order_id: str = None) -> Tuple[List, Dict[str, Any], str]:
    """Process user message and generate response."""
    try:
        cred_results = verify_credentials(["GOOGLE_API_KEY", "GOOGLE_APPLICATION_CREDENTIALS"])
        if not all(cred_results.values()):
            raise EnvironmentError("Missing required credentials")
        if state["messages"] and state["messages"][-1]["role"] == "user":
            state["messages"] = state["messages"][:-1]
        if order_id and order_id.strip():
            message = f"What's the status of my order {order_id.strip()}?"
        message = bleach.clean(message)
        formatted_history = []
        if history:
            for h in history:
                if isinstance(h, tuple):
                    user_msg, assistant_msg = h
                    if user_msg:
                        formatted_history.append({"role": "user", "content": user_msg})
                    if assistant_msg:
                        formatted_history.append({"role": "assistant", "content": assistant_msg})
                elif isinstance(h, dict):
                    formatted_history.append(h)
        state["messages"] = formatted_history
        if not state["messages"] or state["messages"][-1]["role"] != "user":
            state["messages"].append({"role": "user", "content": message})
        yield formatted_history, state, "typing-indicator active"
        try:
            cred_results = verify_credentials(["GOOGLE_API_KEY", "GOOGLE_APPLICATION_CREDENTIALS"])
            if not all(cred_results.values()):
                raise EnvironmentError("Credentials became invalid during processing")
            updated_state = chat_with_user(message, state)
            if updated_state and "messages" in updated_state:
                new_messages = [msg for msg in updated_state["messages"] if msg not in state["messages"]]
                if new_messages:
                    state["messages"].extend(new_messages)
                for key in ["order_lookup_attempted", "current_order_id", "needs_human_agent", 
                            "contact_info_collected", "customer_name", "customer_email", 
                            "customer_phone", "contact_step"]:
                    if key in updated_state:
                        state[key] = updated_state[key]
        except EnvironmentError as e:
            logger.error(f"Credential error during chat processing: {e}")
            state["messages"].append({"role": "assistant", "content": "I'm sorry, but I'm having trouble accessing my services right now. Please try again later."})
        except Exception as e:
            logger.error(f"Error in chat processing: {e}")
            if not any(msg["role"] == "assistant" for msg in state["messages"]):
                state["messages"].append({"role": "assistant", "content": "I encountered an error processing your request."})
        state["chat_history"] = state["messages"]
        yield state["messages"], state, "typing-indicator"
    except EnvironmentError as e:
        logger.error(f"Credential error in process_message: {e}")
        error_message = {"role": "assistant", "content": "I'm sorry, but I'm having trouble accessing my services right now. Please try again later."}
        state["messages"] = state.get("messages", []) + [error_message]
        yield state["messages"], state, "typing-indicator"
    except Exception as e:
        logger.error(f"Error in process_message: {e}", exc_info=True)
        error_message = {"role": "assistant", "content": "I encountered an error processing your request."}
        state["messages"] = state.get("messages", []) + [error_message]
        yield state["messages"], state, "typing-indicator"

def clear_conversation(state: Dict[str, Any]) -> Tuple[List, Dict[str, Any]]:
    new_state = reset_state()
    new_state["session_id"] = datetime.now().strftime("%Y%m%d%H%M%S")
    new_state["type"] = "messages"
    new_state["messages"] = []
    new_state["chat_history"] = []
    return [], new_state

def submit_feedback(feedback: str, state: Dict[str, Any]) -> Dict[str, Any]:
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
            "conversation": new_state["chat_history"]
        }
        with open(feedback_file, "w") as f:
            json.dump(feedback_data, f, indent=2)
        print(f"Feedback saved to {feedback_file}")
    except Exception as e:
        print(f"Error saving feedback: {e}")
    return new_state

def use_suggestion(suggestion_text: str, state: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    return suggestion_text, state

def track_order(order_id: str, state: Dict[str, Any]) -> Tuple[List[Tuple[str, str]], Dict[str, Any], str]:
    if not order_id.strip():
        error_message = {"role": "assistant", "content": "Please provide a valid order ID."}
        return state["messages"] + [error_message], state, ""
    try:
        processed = process_message(f"What's the status of my order {order_id.strip()}?", state["chat_history"], state)
        return next(processed)
    except Exception as e:
        print(f"Error tracking order: {e}")
        error_message = {"role": "assistant", "content": "I couldn't find information for that order ID. Please verify and try again."}
        return state["messages"] + [error_message], state, ""

def get_faq_response(faq_key: str, state: Dict[str, Any]) -> Tuple[List, Dict[str, Any], str]:
    if faq_key not in FAQ_CONFIG["responses"]:
        return state.get("chat_history", []), state, ""
    response = FAQ_CONFIG["responses"][faq_key]
    query = f"Tell me about your {faq_key.replace('_', ' ')}"
    new_state = state.copy()
    if state.get("type") == "messages":
        history = state.get("chat_history", [])
        history.append({"role": "user", "content": query})
        history.append({"role": "assistant", "content": response})
    else:
        history = state.get("chat_history", []) + [(query, response)]
    new_state["messages"] = [{"role": "user" if i % 2 == 0 else "assistant", "content": msg}
                             for i, msg in enumerate([query, response])]
    new_state["chat_history"] = history
    return history, new_state, ""

# ===== Gradio Interface =====

with gr.Blocks(theme=theme, css=css) as demo:
    state = gr.State(value=reset_state())
    suggestion_text1 = gr.State("What's your return policy?")
    suggestion_text2 = gr.State("How do I track my order?")
    suggestion_text3 = gr.State("I need to speak to a human")
    suggestion_text4 = gr.State("What payment methods do you accept?")
    return_policy_key = gr.State("return_policy")
    shipping_policy_key = gr.State("shipping_policy")
    payment_methods_key = gr.State("payment_methods")
    contact_info_key = gr.State("contact_info")
    
    with gr.Row():
        with gr.Column(scale=1, elem_classes="sidebar"):
            gr.Markdown(
                """
                <div class="sidebar-header">
                    <img src="https://img.icons8.com/fluency/48/000000/shopping-cart.png" alt="Logo">
                    <h2>E-commerce Support</h2>
                </div>
                """, elem_id="sidebar-header"
            )
            gr.Markdown("### Quick Links", elem_classes="sidebar-section")
            with gr.Row():
                faq_btn = gr.Button("FAQs", elem_classes="sidebar-link")
                contact_btn = gr.Button("Contact Us", elem_classes="sidebar-link")
            with gr.Row():
                returns_btn = gr.Button("Returns", elem_classes="sidebar-link")
                shipping_btn = gr.Button("Shipping", elem_classes="sidebar-link")
            gr.Markdown("### Track Your Order", elem_classes="order-tracking")
            order_input = gr.Textbox(placeholder="Enter your order ID", label="Order ID", show_label=False)
            track_btn = gr.Button("Track Order", variant="primary")
            with gr.Accordion("Frequently Asked Questions", open=False, elem_classes="faq-accordion"):
                gr.Markdown("""
                **What is your return policy?**

                Items can be returned within 30 days of delivery for a full refund.

                **How long does shipping take?**

                Standard shipping takes 5-7 business days. Express shipping is 2-3 days.

                **What payment methods do you accept?**

                We accept credit cards, PayPal, and Apple Pay.

                **How do I track my order?**

                Enter your order ID in the tracking box on the left sidebar.
                """)
            feedback = gr.Textbox(placeholder="Share your feedback about this chat experience...", label="Feedback", lines=3)
            submit_btn = gr.Button("Submit Feedback", variant="secondary")
        with gr.Column(scale=3, elem_classes="chat-container"):
            with gr.Row(elem_classes="chat-header"):
                gr.Markdown("## E-commerce Support Assistant")
                clear_btn = gr.Button("Clear Chat", variant="secondary", elem_classes="chat-header-btn")
            chatbot = gr.Chatbot([], elem_id="chatbot", avatar_images=("👤", "🤖"), show_copy_button=True, height=500, type="messages")
            typing_indicator = gr.HTML('<div class="typing-indicator" id="typing-indicator" style="display:none;"><span></span><span></span><span></span> Bot is typing...</div>')
            with gr.Row(elem_classes="input-area"):
                with gr.Column(scale=4):
                    msg = gr.Textbox(placeholder="Type your message here...", label="Message", show_label=False, container=False)
                with gr.Column(scale=1):
                    send_btn = gr.Button("Send", variant="primary")
            gr.Markdown("### Suggested Questions", elem_classes="chat-suggestions")
            with gr.Row():
                suggestion1 = gr.Button("What's your return policy?", elem_classes="suggestion-chip")
                suggestion2 = gr.Button("How do I track my order?", elem_classes="suggestion-chip")
                suggestion3 = gr.Button("I need to speak to a human", elem_classes="suggestion-chip")
                suggestion4 = gr.Button("What payment methods do you accept?", elem_classes="suggestion-chip")
    
    send_btn.click(fn=process_message, inputs=[msg, chatbot, state], outputs=[chatbot, state, typing_indicator], api_name="send").then(fn=lambda: "", outputs=msg)
    msg.submit(fn=process_message, inputs=[msg, chatbot, state], outputs=[chatbot, state, typing_indicator], api_name="submit").then(fn=lambda: "", outputs=msg)
    clear_btn.click(fn=clear_conversation, inputs=[state], outputs=[chatbot, state], api_name="clear")
    submit_btn.click(fn=submit_feedback, inputs=[feedback, state], outputs=[state], api_name="feedback").then(fn=lambda: "", outputs=feedback)
    track_btn.click(fn=track_order, inputs=[order_input, state], outputs=[chatbot, state, typing_indicator], api_name="track").then(fn=lambda: "", outputs=order_input)
    suggestion1.click(fn=use_suggestion, inputs=[suggestion_text1, state], outputs=[msg, state]).then(fn=process_message, inputs=[msg, chatbot, state], outputs=[chatbot, state, typing_indicator]).then(fn=lambda: "", outputs=msg)
    suggestion2.click(fn=use_suggestion, inputs=[suggestion_text2, state], outputs=[msg, state]).then(fn=process_message, inputs=[msg, chatbot, state], outputs=[chatbot, state, typing_indicator]).then(fn=lambda: "", outputs=msg)
    suggestion3.click(fn=use_suggestion, inputs=[suggestion_text3, state], outputs=[msg, state]).then(fn=process_message, inputs=[msg, chatbot, state], outputs=[chatbot, state, typing_indicator]).then(fn=lambda: "", outputs=msg)
    suggestion4.click(fn=use_suggestion, inputs=[suggestion_text4, state], outputs=[msg, state]).then(fn=process_message, inputs=[msg, chatbot, state], outputs=[chatbot, state, typing_indicator]).then(fn=lambda: "", outputs=msg)
    faq_btn.click(fn=get_faq_response, inputs=[return_policy_key, state], outputs=[chatbot, state, typing_indicator])
    returns_btn.click(fn=get_faq_response, inputs=[return_policy_key, state], outputs=[chatbot, state, typing_indicator])
    shipping_btn.click(fn=get_faq_response, inputs=[shipping_policy_key, state], outputs=[chatbot, state, typing_indicator])
    contact_btn.click(fn=get_faq_response, inputs=[contact_info_key, state], outputs=[chatbot, state, typing_indicator])

# ===== Health Check and System Status =====

def health_check():
    """Verify system health and credential status."""
    try:
        cred_results = verify_credentials(["GOOGLE_API_KEY", "GOOGLE_APPLICATION_CREDENTIALS", "HUGGINGFACE_TOKEN"])
        if not all(cred_results.values()):
            missing_creds = [k for k, v in cred_results.items() if not v]
            logger.error(f"Health check failed: Missing credentials: {missing_creds}")
            return {"status": "unhealthy", "message": f"Missing credentials: {missing_creds}", "timestamp": datetime.utcnow().isoformat()}
        test_message = "test"
        test_state = reset_state()
        test_state["messages"] = [{"role": "user", "content": test_message}]
        llm_service.generate_response(test_state["messages"], conversation_memory)
        test_vector = [0.0] * 384
        vector_collection.query(query_embeddings=[test_vector], n_results=1)
        return {"status": "healthy", "message": "All systems operational", "timestamp": datetime.utcnow().isoformat(),
                "services": {"llm": "operational", "vector_db": "operational", "credentials": "valid"}}
    except Exception as e:
        error_msg = f"Health check failed: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return {"status": "unhealthy", "message": error_msg, "timestamp": datetime.utcnow().isoformat()}

def configure_logging():
    """Configure logging for production environment."""
    try:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[logging.StreamHandler()]
        )
        logging.getLogger('gradio').setLevel(logging.WARNING)
        logging.getLogger('urllib3').setLevel(logging.WARNING)
        logging.getLogger('matplotlib').setLevel(logging.WARNING)
        logger.info("Logging configured with console output")
        return True
    except Exception as e:
        print(f"Failed to configure logging: {e}")
        return False

def get_server_config():
    """Get server configuration from environment or use defaults."""
    return {
        "server_name": os.getenv("SERVER_HOST", "0.0.0.0"),
        "server_port": int(os.getenv("SERVER_PORT", 7860)),
        "share": os.getenv("ENABLE_SHARE", "false").lower() == "true",
        "auth": None if os.getenv("AUTH_REQUIRED", "false").lower() == "false" else (os.getenv("AUTH_USERNAME", "admin"), os.getenv("AUTH_PASSWORD", "admin")),
        "ssl_keyfile": os.getenv("SSL_KEYFILE", None),
        "ssl_certfile": os.getenv("SSL_CERTFILE", None),
        "ssl_keyfile_password": os.getenv("SSL_KEYFILE_PASSWORD", None)
    }

def initialize_app():
    """Initialize the application and verify all dependencies."""
    logger.info("Initializing application...")
    try:
        cred_results = verify_credentials(["GOOGLE_API_KEY", "GOOGLE_APPLICATION_CREDENTIALS", "HUGGINGFACE_TOKEN"])
        if not all(cred_results.values()):
            missing_creds = [k for k, v in cred_results.items() if not v]
            raise EnvironmentError(f"Missing required credentials: {missing_creds}")
        health_status = health_check()
        if health_status["status"] != "healthy":
            raise RuntimeError(f"Health check failed: {health_status['message']}")
        logger.info("Application initialized successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize application: {e}", exc_info=True)
        raise

# ===== Launch the app =====
if __name__ == "__main__":
    try:
        if not configure_logging():
            print("Warning: Could not configure logging, continuing with default configuration")
        logger.info("Starting E-commerce Support Assistant...")
        initialize_app()
        server_config = get_server_config()
        demo.launch(
            server_name=server_config["server_name"],
            server_port=server_config["server_port"],
            share=server_config["share"],
            auth=server_config["auth"],
            ssl_keyfile=server_config["ssl_keyfile"],
            ssl_certfile=server_config["ssl_certfile"],
            ssl_keyfile_password=server_config["ssl_keyfile_password"],
            max_threads=40,
            show_error=True,
            root_path="",
            favicon_path="assets/favicon.ico",
            allowed_paths=["assets"],
            blocked_paths=["data", "logs", "config"]
        )
        logger.info("Application launched successfully")
    except Exception as e:
        print(f"Critical error: Failed to launch application: {e}")
        sys.exit(1)
    finally:
        logger.info("Shutting down application...")
