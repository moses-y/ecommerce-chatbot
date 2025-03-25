# app.py: Modern Gradio interface for the E-commerce Support Chatbot
import os
import sys
import time
import json
import bleach
import gradio as gr
from datetime import datetime
from dotenv import load_dotenv
from typing import Dict, List, Optional, Tuple, Any

# Add logging
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add the project root to sys.path for module imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Load environment variables
load_dotenv()

# Import required modules
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

# ===== Initialize services with credential verification ====
def initialize_services():
    """Initialize all required services with proper credential verification."""
    logger.info("Initializing services...")
    start_time = time.time()
    
    try:
        # Verify required credentials
        cred_results = verify_credentials([
            "GOOGLE_API_KEY",
            "GOOGLE_APPLICATION_CREDENTIALS",
            "HUGGINGFACE_TOKEN"
        ])
        
        if not all(cred_results.values()):
            missing_creds = [k for k, v in cred_results.items() if not v]
            raise EnvironmentError(f"Missing required credentials: {missing_creds}")

        # Initialize LLM service
        llm_service = LLMService()
        
        # Initialize conversation memory
        conversation_memory = ConversationMemory(
            max_history=CONVERSATION_CONFIG["max_history_length"]
        )
        
        # Load order data and initialize vector database
        orders_df = load_order_data(use_cache=True)
        vector_collection = get_vector_db_instance(orders_df, use_subset=False)
        
        logger.info(f"Services initialized in {time.time() - start_time:.2f} seconds")
        return llm_service, conversation_memory, vector_collection
        
    except Exception as e:
        logger.error(f"Failed to initialize services: {e}")
        raise

# Initialize services
try:
    llm_service, conversation_memory, vector_collection = initialize_services()
except Exception as e:
    logger.error(f"Critical error during service initialization: {e}")
    raise

# ===== UI Configuration =====

# Define a modern theme with glassmorphism effects
theme = gr.themes.Soft(
    primary_hue="teal",
    secondary_hue="indigo",
    neutral_hue="slate",
    font=["Inter", "SF Pro Display", "system-ui", "sans-serif"],
    radius_size=gr.themes.sizes.radius_sm,
).set(
    body_background_fill="linear-gradient(135deg, #0F172A 0%, #1E293B 100%)",
    background_fill_primary="rgba(255, 255, 255, 0.05)",
    background_fill_secondary="rgba(255, 255, 255, 0.03)",
    border_color_primary="rgba(255, 255, 255, 0.1)",
    button_primary_background_fill="rgba(20, 184, 166, 0.8)",
    button_primary_background_fill_hover="rgba(20, 184, 166, 1)",
    button_primary_text_color="white",
    button_secondary_background_fill="rgba(255, 255, 255, 0.05)",
    button_secondary_background_fill_hover="rgba(255, 255, 255, 0.1)",
    button_secondary_text_color="white",
    block_title_text_color="rgba(255, 255, 255, 0.9)",
    block_label_text_color="rgba(255, 255, 255, 0.7)",
    input_background_fill="rgba(255, 255, 255, 0.05)",
)

# Custom CSS for enhanced UI elements
css = """
/* Global styles */
:root {
    --primary: rgba(20, 184, 166, 1);
    --primary-light: rgba(20, 184, 166, 0.1);
    --text-primary: rgba(255, 255, 255, 0.9);
    --text-secondary: rgba(255, 255, 255, 0.7);
    --bg-glass: rgba(255, 255, 255, 0.05);
    --bg-glass-hover: rgba(255, 255, 255, 0.1);
    --border-glass: rgba(255, 255, 255, 0.1);
    --shadow-glass: 0 4px 6px rgba(0, 0, 0, 0.1), 0 1px 3px rgba(0, 0, 0, 0.08);
}

/* Glassmorphism effects */
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

/* Chat container */
.chat-container {
    height: calc(120vh - 180px);
    display: flex;
    flex-direction: column;
    overflow: hidden;
}

/* Chatbot area */
#chatbot {
    flex: 1;
    overflow-y: auto;
    padding: 1rem;
    border-radius: 12px;
    background: var(--bg-glass);
    backdrop-filter: blur(10px);
    border: 1px solid var(--border-glass);
    box-shadow: var(--shadow-glass);
    scrollbar-width: thin;
    scrollbar-color: var(--primary) transparent;
}

#chatbot::-webkit-scrollbar {
    width: 6px;
}

#chatbot::-webkit-scrollbar-track {
    background: transparent;
}

#chatbot::-webkit-scrollbar-thumb {
    background-color: var(--primary);
    border-radius: 6px;
}

/* Message bubbles */
#chatbot .user-message {
    background: var(--primary);
    color: white;
    border-radius: 18px 18px 0 18px;
    padding: 12px 16px;
    margin: 8px 0;
    max-width: 80%;
    align-self: flex-end;
    box-shadow: 0 1px 2px rgba(0, 0, 0, 0.1);
    animation: slideInRight 0.3s ease;
}

#chatbot .bot-message {
    background: var(--bg-glass);
    color: var(--text-primary);
    border-radius: 18px 18px 18px 0;
    padding: 12px 16px;
    margin: 8px 0;
    max-width: 80%;
    align-self: flex-start;
    box-shadow: 0 1px 2px rgba(0, 0, 0, 0.1);
    animation: slideInLeft 0.3s ease;
}

/* Input area */
.input-area {
    display: flex;
    gap: 8px;
    padding: 16px;
    background: var(--bg-glass);
    border-radius: 12px;
    margin-top: 16px;
    border: 1px solid var(--border-glass);
}

.input-area input {
    flex: 1;
    background: rgba(255, 255, 255, 0.1);
    border: 1px solid var(--border-glass);
    border-radius: 8px;
    padding: 12px 16px;
    color: var(--text-primary);
    font-size: 16px;
}

.input-area button {
    background: var(--primary);
    color: white;
    border: none;
    border-radius: 8px;
    padding: 12px 20px;
    font-weight: 500;
    cursor: pointer;
    transition: all 0.2s ease;
}

.input-area button:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
}

.input-area button:active {
    transform: translateY(0);
}

/* Quick actions */
.quick-actions {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin-top: 12px;
}

.quick-action-btn {
    background: var(--bg-glass);
    color: var(--text-primary);
    border: 1px solid var(--border-glass);
    border-radius: 20px;
    padding: 8px 16px;
    font-size: 14px;
    cursor: pointer;
    transition: all 0.2s ease;
}

.quick-action-btn:hover {
    background: var(--primary-light);
    border-color: var(--primary);
}

/* Typing indicator */
.typing-indicator {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    padding: 8px 16px;
    background: var(--bg-glass);
    border-radius: 18px;
    margin: 8px 0;
}

.typing-indicator span {
    width: 8px;
    height: 8px;
    background: var(--primary);
    border-radius: 50%;
    display: inline-block;
    animation: bounce 0.8s infinite alternate;
}

.typing-indicator span:nth-child(2) {
    animation-delay: 0.2s;
}

.typing-indicator span:nth-child(3) {
    animation-delay: 0.4s;
}

/* Animations */
@keyframes slideInRight {
    from {
        opacity: 0;
        transform: translateX(20px);
    }
    to {
        opacity: 1;
        transform: translateX(0);
    }
}

@keyframes slideInLeft {
    from {
        opacity: 0;
        transform: translateX(-20px);
    }
    to {
        opacity: 1;
        transform: translateX(0);
    }
}

@keyframes bounce {
    from {
        transform: translateY(0);
    }
    to {
        transform: translateY(-4px);
    }
}

/* Sidebar */
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
    margin-bottom: 10px;
}

.sidebar-header img {
    width: 20px;
    height: 20px;
}

.sidebar-header h2 {
    color: var(--text-primary);
    font-size: 15px;
    font-weight: 600;
    margin: 0;
}

.sidebar-section {
    margin-bottom: 5px;
}

.sidebar-section h3 {
    color: var(--text-primary);
    font-size: 10px;
    font-weight: 600;
    margin-bottom: 5px;
    padding-bottom: 5px;
    border-bottom: 1px solid var(--border-glass);
}

.sidebar-link {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 5px 10px;
    color: var(--text-secondary);
    text-decoration: none;
    border-radius: 8px;
    transition: all 0.2s ease;
}

.sidebar-link:hover {
    background: var(--bg-glass-hover);
    color: var(--text-primary);
}

.sidebar-link.active {
    background: var(--primary-light);
    color: var(--primary);
}

/* Order tracking section */
.order-tracking {
    background: var(--bg-glass);
    border-radius: 12px;
    padding: 16px;
    margin-top: 24px;
    border: 1px solid var(--border-glass);
}

.order-tracking h3 {
    color: var(--text-primary);
    font-size: 16px;
    margin-top: 0;
    margin-bottom: 5px;
}

.order-tracking input {
    width: 100%;
    background: rgba(255, 255, 255, 0.1);
    border: 1px solid var(--border-glass);
    border-radius: 8px;
    padding: 10px 12px;
    color: var(--text-primary);
    margin-bottom: 5px;
}

.order-tracking button {
    width: 100%;
    background: var(--primary);
    color: white;
    border: none;
    border-radius: 5px;
    padding: 5px;
    font-weight: 500;
    cursor: pointer;
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

/* Accessibility */
button:focus, input:focus {
    outline: 2px solid var(--primary);
    outline-offset: 2px;
}

/* Dark mode toggle */
.theme-toggle {
    position: absolute;
    top: 20px;
    right: 20px;
    background: var(--bg-glass);
    border: 1px solid var(--border-glass);
    border-radius: 20px;
    padding: 8px 12px;
    display: flex;
    align-items: center;
    gap: 8px;
    cursor: pointer;
    transition: all 0.2s ease;
}

.theme-toggle:hover {
    background: var(--bg-glass-hover);
}

.theme-toggle span {
    color: var(--text-secondary);
    font-size: 14px;
}

/* Chat header */
.chat-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 16px;
    background: var(--bg-glass);
    border-radius: 12px 12px 0 0;
    border-bottom: 1px solid var(--border-glass);
}

.chat-header h2 {
    color: var(--text-primary);
    font-size: 18px;
    font-weight: 600;
    margin: 0;
}

.chat-header-actions {
    display: flex;
    gap: 8px;
}

.chat-header-btn {
    background: var(--bg-glass);
    color: var(--text-secondary);
    border: 1px solid var(--border-glass);
    border-radius: 8px;
    padding: 6px 12px;
    font-size: 14px;
    cursor: pointer;
    transition: all 0.2s ease;
}

.chat-header-btn:hover {
    background: var(--bg-glass-hover);
    color: var(--text-primary);
}

/* FAQ accordion */
.faq-accordion {
    margin-top: 16px;
}

.faq-item {
    border: 1px solid var(--border-glass);
    border-radius: 8px;
    margin-bottom: 8px;
    overflow: hidden;
}

.faq-question {
    padding: 12px 16px;
    background: var(--bg-glass);
    color: var(--text-primary);
    font-weight: 500;
    cursor: pointer;
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.faq-answer {
    padding: 0 16px;
    max-height: 0;
    overflow: hidden;
    transition: max-height 0.3s ease, padding 0.3s ease;
    background: var(--bg-glass-hover);
}

.faq-item.active .faq-answer {
    padding: 16px;
    max-height: 500px;
}

/* Chat suggestions */
.chat-suggestions {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin-top: 12px;
}

.suggestion-chip {
    background: var(--bg-glass);
    color: var(--text-secondary);
    border: 1px solid var(--border-glass);
    border-radius: 16px;
    padding: 6px 12px;
    font-size: 14px;
    cursor: pointer;
    transition: all 0.2s ease;
}

.suggestion-chip:hover {
    background: var(--primary-light);
    border-color: var(--primary);
    color: var(--primary);
}
"""

# ===== Helper Functions =====

# ===== Helper Functions =====

def process_message(
    message: str,
    history: List,
    state: Dict[str, Any],
    order_id: str = None
) -> Tuple[List, Dict[str, Any], str]:
    """Process user message and generate response."""
    try:
        # Verify credentials before processing
        cred_results = verify_credentials([
            "GOOGLE_API_KEY",
            "GOOGLE_APPLICATION_CREDENTIALS"
        ])
        if not all(cred_results.values()):
            raise EnvironmentError("Missing required credentials")

        # Clear only if duplicate exists
        if state["messages"] and state["messages"][-1]["role"] == "user":
            state["messages"] = state["messages"][:-1]

        # If order ID is provided, use it
        if order_id and order_id.strip():
            message = f"What's the status of my order {order_id.strip()}?"

        # Sanitize input
        message = bleach.clean(message)

        # Format history
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

        # Update state with formatted history
        state["messages"] = formatted_history
        if not state["messages"] or state["messages"][-1]["role"] != "user":
            state["messages"].append({"role": "user", "content": message})

        # Show typing indicator
        yield formatted_history, state, "typing-indicator active"

        # Process through chatbot with credential verification
        try:
            # Verify credentials again before chat processing
            cred_results = verify_credentials([
                "GOOGLE_API_KEY",
                "GOOGLE_APPLICATION_CREDENTIALS"
            ])
            if not all(cred_results.values()):
                raise EnvironmentError("Credentials became invalid during processing")

            updated_state = chat_with_user(message, state)
            
            # Add any new messages from the updated state
            if updated_state and "messages" in updated_state:
                new_messages = [msg for msg in updated_state["messages"] 
                              if msg not in state["messages"]]
                if new_messages:
                    state["messages"].extend(new_messages)
                    
                # Update other state properties
                for key in ["order_lookup_attempted", "current_order_id", "needs_human_agent", 
                          "contact_info_collected", "customer_name", "customer_email", 
                          "customer_phone", "contact_step"]:
                    if key in updated_state:
                        state[key] = updated_state[key]

        except EnvironmentError as e:
            logger.error(f"Credential error during chat processing: {e}")
            state["messages"].append({
                "role": "assistant",
                "content": "I'm sorry, but I'm having trouble accessing my services right now. Please try again later."
            })
        except Exception as e:
            logger.error(f"Error in chat processing: {e}")
            if not any(msg["role"] == "assistant" for msg in state["messages"]):
                state["messages"].append({
                    "role": "assistant",
                    "content": "I apologize, but I encountered an error processing your request."
                })

        # Save to chat history
        state["chat_history"] = state["messages"]
        
        # Hide typing indicator
        yield state["messages"], state, "typing-indicator"

    except EnvironmentError as e:
        logger.error(f"Credential error in process_message: {e}")
        error_message = {
            "role": "assistant",
            "content": "I'm sorry, but I'm having trouble accessing my services right now. Please try again later."
        }
        state["messages"] = state.get("messages", []) + [error_message]
        yield state["messages"], state, "typing-indicator"
    except Exception as e:
        logger.error(f"Error in process_message: {e}", exc_info=True)
        error_message = {
            "role": "assistant",
            "content": "I apologize, but I encountered an error processing your request."
        }
        state["messages"] = state.get("messages", []) + [error_message]
        yield state["messages"], state, "typing-indicator"

def clear_conversation(state: Dict[str, Any]) -> Tuple[List, Dict[str, Any]]:
    """Clear the conversation history."""
    new_state = reset_state()
    new_state["session_id"] = datetime.now().strftime("%Y%m%d%H%M%S")
    new_state["type"] = "messages"
    new_state["messages"] = []  # Ensure messages are empty
    new_state["chat_history"] = []  # Ensure chat history is empty
    return [], new_state

def submit_feedback(feedback: str, state: Dict[str, Any]) -> Dict[str, Any]:
    """Submit user feedback."""
    if not feedback:
        return state

    new_state = state.copy()
    new_state["feedback"] = feedback

    # Save feedback to file
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
    """Use a suggested query."""
    return suggestion_text, state

def track_order(order_id: str, state: Dict[str, Any]) -> Tuple[List[Tuple[str, str]], Dict[str, Any], str]:
    if not order_id.strip():
        error_message = {"role": "assistant", "content": "Please provide a valid order ID."}
        return state["messages"] + [error_message], state, ""
    
    try:
        # Process order lookup
        processed = process_message(
            f"What's the status of my order {order_id.strip()}?",
            state["chat_history"],
            state
        )
        return next(processed)
    except Exception as e:
        print(f"Error tracking order: {e}")
        error_message = {"role": "assistant", "content": "I'm sorry, I couldn't find information for that order ID. Please verify the order ID and try again."}
        return state["messages"] + [error_message], state, ""

def get_faq_response(faq_key: str, state: Dict[str, Any]) -> Tuple[List, Dict[str, Any], str]:
    """Get a response for a FAQ."""
    if faq_key not in FAQ_CONFIG["responses"]:
        return state.get("chat_history", []), state, ""

    # Get the FAQ response
    response = FAQ_CONFIG["responses"][faq_key]
    query = f"Tell me about your {faq_key.replace('_', ' ')}"

    # Update history and state
    new_state = state.copy()
    if state.get("type") == "messages":
        history = state.get("chat_history", [])
        history.append({"role": "user", "content": query})
        history.append({"role": "assistant", "content": response})
    else:
        history = state.get("chat_history", []) + [(query, response)]

    new_state["messages"] = [
        {"role": "user" if i % 2 == 0 else "assistant", "content": msg}
        for i, msg in enumerate([query, response])
    ]
    new_state["chat_history"] = history

    return history, new_state, ""
    
# ===== Gradio Interface =====

with gr.Blocks(theme=theme, css=css) as demo:
    # State
    state = gr.State(value=reset_state())

    # Create suggestion text variables
    suggestion_text1 = gr.State("What's your return policy?")
    suggestion_text2 = gr.State("How do I track my order?")
    suggestion_text3 = gr.State("I need to speak to a human")
    suggestion_text4 = gr.State("What payment methods do you accept?")

    # FAQ key states
    return_policy_key = gr.State("return_policy")
    shipping_policy_key = gr.State("shipping_policy")
    payment_methods_key = gr.State("payment_methods")
    contact_info_key = gr.State("contact_info")

    # Layout
    with gr.Row():
        # Sidebar
        with gr.Column(scale=1, elem_classes="sidebar"):
            gr.Markdown(
                """
                <div class="sidebar-header">
                    <img src="https://img.icons8.com/fluency/48/000000/shopping-cart.png" alt="Logo">
                    <h2>E-commerce Support</h2>
                </div>
                """,
                elem_id="sidebar-header"
            )

            # Quick Links Section
            gr.Markdown("### Quick Links", elem_classes="sidebar-section")
            with gr.Row():
                faq_btn = gr.Button("FAQs", elem_classes="sidebar-link")
                contact_btn = gr.Button("Contact Us", elem_classes="sidebar-link")
            with gr.Row():
                returns_btn = gr.Button("Returns", elem_classes="sidebar-link")
                shipping_btn = gr.Button("Shipping", elem_classes="sidebar-link")

            # Order Tracking Section
            gr.Markdown("### Track Your Order", elem_classes="order-tracking")
            order_input = gr.Textbox(
                placeholder="Enter your order ID",
                label="Order ID",
                show_label=False
            )
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

            feedback = gr.Textbox(
                placeholder="Share your feedback about this chat experience...",
                label="Feedback",
                lines=3
            )
            submit_btn = gr.Button("Submit Feedback", variant="secondary")

        # Main chat area
        with gr.Column(scale=3, elem_classes="chat-container"):
            with gr.Row(elem_classes="chat-header"):
                gr.Markdown("## E-commerce Support Assistant")
                clear_btn = gr.Button("Clear Chat", variant="secondary", elem_classes="chat-header-btn")

            chatbot = gr.Chatbot(
                [],
                elem_id="chatbot",
                avatar_images=("👤", "🤖"),
                show_copy_button=True,
                height=500,
                type="messages"
            )

            typing_indicator = gr.HTML(
                '<div class="typing-indicator" id="typing-indicator" style="display:none;"><span></span><span></span><span></span> Bot is typing...</div>'
            )

            with gr.Row(elem_classes="input-area"):
                with gr.Column(scale=4):
                    msg = gr.Textbox(
                        placeholder="Type your message here...",
                        label="Message",
                        show_label=False,
                        container=False
                    )
                with gr.Column(scale=1):
                    send_btn = gr.Button("Send", variant="primary")

            # Suggestions Section
            gr.Markdown("### Suggested Questions", elem_classes="chat-suggestions")
            with gr.Row():
                suggestion1 = gr.Button("What's your return policy?", elem_classes="suggestion-chip")
                suggestion2 = gr.Button("How do I track my order?", elem_classes="suggestion-chip")
                suggestion3 = gr.Button("I need to speak to a human", elem_classes="suggestion-chip")
                suggestion4 = gr.Button("What payment methods do you accept?", elem_classes="suggestion-chip")

    # Event handlers
    send_btn.click(
        fn=process_message,
        inputs=[msg, chatbot, state],
        outputs=[chatbot, state, typing_indicator],
        api_name="send"
    ).then(
        fn=lambda: "",
        outputs=msg
    )

    msg.submit(
        fn=process_message,
        inputs=[msg, chatbot, state],
        outputs=[chatbot, state, typing_indicator],
        api_name="submit"
    ).then(
        fn=lambda: "",
        outputs=msg
    )

    clear_btn.click(
        fn=clear_conversation,
        inputs=[state],
        outputs=[chatbot, state],
        api_name="clear"
    )

    submit_btn.click(
        fn=submit_feedback,
        inputs=[feedback, state],
        outputs=[state],
        api_name="feedback"
    ).then(
        fn=lambda: "",
        outputs=feedback
    )

    track_btn.click(
        fn=track_order,
        inputs=[order_input, state],
        outputs=[chatbot, state, typing_indicator],
        api_name="track"
    ).then(
        fn=lambda: "",
        outputs=order_input
    )

    # Suggestion buttons - fixed to use State objects instead of strings
    suggestion1.click(
        fn=use_suggestion,
        inputs=[suggestion_text1, state],
        outputs=[msg, state]
    ).then(
        fn=process_message,
        inputs=[msg, chatbot, state],
        outputs=[chatbot, state, typing_indicator]
    ).then(
        fn=lambda: "",
        outputs=msg
    )

    suggestion2.click(
        fn=use_suggestion,
        inputs=[suggestion_text2, state],
        outputs=[msg, state]
    ).then(
        fn=process_message,
        inputs=[msg, chatbot, state],
        outputs=[chatbot, state, typing_indicator]
    ).then(
        fn=lambda: "",
        outputs=msg
    )

    suggestion3.click(
        fn=use_suggestion,
        inputs=[suggestion_text3, state],
        outputs=[msg, state]
    ).then(
        fn=process_message,
        inputs=[msg, chatbot, state],
        outputs=[chatbot, state, typing_indicator]
    ).then(
        fn=lambda: "",
        outputs=msg
    )

    suggestion4.click(
        fn=use_suggestion,
        inputs=[suggestion_text4, state],
        outputs=[msg, state]
    ).then(
        fn=process_message,
        inputs=[msg, chatbot, state],
        outputs=[chatbot, state, typing_indicator]
    ).then(
        fn=lambda: "",
        outputs=msg
    )

    # FAQ buttons
    faq_btn.click(
        fn=get_faq_response,
        inputs=[return_policy_key, state],
        outputs=[chatbot, state, typing_indicator]
    )

    returns_btn.click(
        fn=get_faq_response,
        inputs=[return_policy_key, state],
        outputs=[chatbot, state, typing_indicator]
    )

    shipping_btn.click(
        fn=get_faq_response,
        inputs=[shipping_policy_key, state],
        outputs=[chatbot, state, typing_indicator]
    )

    contact_btn.click(
        fn=get_faq_response,
        inputs=[contact_info_key, state],
        outputs=[chatbot, state, typing_indicator]
    )

# ===== Health Check and System Status =====

def health_check():
    """Verify system health and credential status."""
    try:
        # Verify credentials
        cred_results = verify_credentials([
            "GOOGLE_API_KEY",
            "GOOGLE_APPLICATION_CREDENTIALS",
            "HUGGINGFACE_TOKEN"
        ])
        
        if not all(cred_results.values()):
            missing_creds = [k for k, v in cred_results.items() if not v]
            logger.error(f"Health check failed: Missing credentials: {missing_creds}")
            return {
                "status": "unhealthy",
                "message": f"Missing credentials: {missing_creds}",
                "timestamp": datetime.utcnow().isoformat()
            }
            
        # Verify services
        test_message = "test"
        test_state = reset_state()
        test_state["messages"] = [{"role": "user", "content": test_message}]
        
        # Test LLM service
        llm_service.generate_response(test_state["messages"], conversation_memory)
        
        # Test vector database
        vector_collection.query("test")
        
        return {
            "status": "healthy",
            "message": "All systems operational",
            "timestamp": datetime.utcnow().isoformat(),
            "services": {
                "llm": "operational",
                "vector_db": "operational",
                "credentials": "valid"
            }
        }
    except Exception as e:
        error_msg = f"Health check failed: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return {
            "status": "unhealthy",
            "message": error_msg,
            "timestamp": datetime.utcnow().isoformat()
        }

# ===== Launch Configuration =====

def configure_logging():
    """Configure logging for production environment."""
    try:
        log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
        os.makedirs(log_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d')
        log_file = os.path.join(log_dir, f"app_{timestamp}.log")
        
        # Configure root logger
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),  # Console output
                logging.handlers.RotatingFileHandler(
                    log_file,
                    maxBytes=10485760,  # 10MB
                    backupCount=5,
                    encoding='utf-8'
                ),
                logging.handlers.TimedRotatingFileHandler(
                    log_file,
                    when='midnight',
                    interval=1,
                    backupCount=30,
                    encoding='utf-8'
                )
            ]
        )
        
        # Set specific levels for third-party loggers
        logging.getLogger('gradio').setLevel(logging.WARNING)
        logging.getLogger('urllib3').setLevel(logging.WARNING)
        logging.getLogger('matplotlib').setLevel(logging.WARNING)
        
        logger.info(f"Logging configured. Log file: {log_file}")
        return True
    except Exception as e:
        print(f"Failed to configure logging: {e}")
        raise

def get_server_config():
    """Get server configuration from environment or use defaults."""
    return {
        "server_name": os.getenv("SERVER_HOST", "0.0.0.0"),
        "server_port": int(os.getenv("SERVER_PORT", 7860)),
        "share": os.getenv("ENABLE_SHARE", "false").lower() == "true",
        "auth": None if os.getenv("AUTH_REQUIRED", "false").lower() == "false" else (
            os.getenv("AUTH_USERNAME", "admin"),
            os.getenv("AUTH_PASSWORD", "admin")
        ),
        "ssl_keyfile": os.getenv("SSL_KEYFILE", None),
        "ssl_certfile": os.getenv("SSL_CERTFILE", None),
        "ssl_keyfile_password": os.getenv("SSL_KEYFILE_PASSWORD", None)
    }

def initialize_app():
    """Initialize the application and verify all dependencies."""
    logger.info("Initializing application...")
    
    try:
        # Verify credentials
        cred_results = verify_credentials([
            "GOOGLE_API_KEY",
            "GOOGLE_APPLICATION_CREDENTIALS",
            "HUGGINGFACE_TOKEN"
        ])
        
        if not all(cred_results.values()):
            missing_creds = [k for k, v in cred_results.items() if not v]
            raise EnvironmentError(f"Missing required credentials: {missing_creds}")
            
        # Verify services health
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
        # Configure logging first
        configure_logging()
        logger.info("Starting E-commerce Support Assistant...")
        
        # Initialize application
        initialize_app()
        
        # Add health check endpoint
        demo.add_api_route("/health", health_check, methods=["GET"])
        
        # Get server configuration
        server_config = get_server_config()
        
        # Launch the app with production configuration
        demo.launch(
            server_name=server_config["server_name"],
            server_port=server_config["server_port"],
            share=server_config["share"],
            auth=server_config["auth"],
            ssl_keyfile=server_config["ssl_keyfile"],
            ssl_certfile=server_config["ssl_certfile"],
            ssl_keyfile_password=server_config["ssl_keyfile_password"],
            enable_queue=True,
            max_threads=40,
            show_error=True,
            cache_examples=True,
            max_queue_size=100,
            root_path="",
            concurrency_count=4,
            startup_timeout=60,
            favicon_path="assets/favicon.ico",
            allowed_paths=["assets"],
            blocked_paths=["data", "logs", "config"],
            queue_callbacks=True,
            show_api=False,
            analytics_enabled=False
        )
        
        logger.info("Application launched successfully")
        
    except Exception as e:
        logger.critical(f"Failed to launch application: {e}", exc_info=True)
        sys.exit(1)
    finally:
        logger.info("Shutting down application...")
