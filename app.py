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

# Initialize vector database with error handling - ONLY ONCE using singleton pattern
print("Initializing vector database...")
start_time = time.time()
try:
    # Load order data
    orders_df = load_order_data(use_cache=True)

    # Initialize vector database using singleton pattern
    vector_collection = get_vector_db_instance(orders_df, use_subset=False)

    print(f"Vector database initialized in {time.time() - start_time:.2f} seconds")
except Exception as e:
    print(f"Error initializing vector DB: {e}")
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
/* Chat container */
.chat-container {
    height: calc(100vh - 240px); /* Reduced from 120vh to 100vh */
    min-height: 400px; /* Minimum height to prevent too small containers */
    max-height: 800px; /* Maximum height to prevent excessive stretching */
    display: flex;
    flex-direction: column;
    overflow: hidden;
    position: relative; /* For proper child positioning */
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
    scrollbar-color: var(--primary) transparent);
    height: calc(100% - 120px) !important; /* Adjust for input area */
    min-height: 300px; /* Minimum height */
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
    position: sticky;
    bottom: 0;
    background: var(--bg-glass);
    backdrop-filter: blur(10px);
    border-top: 1px solid var(--border-glass);
    padding: 1rem;
    margin-top: auto;
    z-index: 10;
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
    .chat-container {
        height: calc(100vh - 180px);
        min-height: 300px;
    }

    #chatbot {
        height: calc(100% - 100px) !important;
        min-height: 200px;
    }

    .input-area {
        padding: 0.75rem;
    }
}

/* For very small screens */
@media (max-width: 480px) {
    .chat-container {
        height: calc(100vh - 160px);
        min-height: 250px;
    }

    #chatbot {
        height: calc(100% - 80px) !important;
        min-height: 150px;
    }

    .input-area {
        padding: 0.5rem;
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

def reset_state() -> Dict:
    """Initialize or reset the chatbot's state."""
    return {
        "messages": [],
        "order_lookup_attempted": False,
        "current_order_id": None,
        "needs_human_agent": False,
        "contact_info_collected": False,
        "customer_name": None,
        "customer_email": None,
        "customer_phone": None,
        "contact_step": 0,
        "chat_history": [],
        "session_id": datetime.now().strftime("%Y%m%d%H%M%S"),
        "feedback": None,
        "type": "messages"  # Add this to track the chatbot type
    }

def process_message(
    message: str,
    history: List,
    state: Dict[str, Any],
    order_id: str = None
) -> Tuple[List, Dict[str, Any], str]:
    """Process user message and generate response."""
    try:
        # Clear only if duplicate exists - preserve previous conversation
        if state["messages"] and state["messages"][-1]["role"] == "user":
            state["messages"] = state["messages"][:-1]  # Remove last user message

        # If order ID is provided, use it
        if order_id and order_id.strip():
            message = f"What's the status of my order {order_id.strip()}?"

        # Sanitize input
        message = bleach.clean(message)

        # Convert history to the correct format
        formatted_history = []
        if history:
            for h in history:
                if isinstance(h, tuple):
                    user_msg, assistant_msg = h
                    formatted_history.append({"role": "user", "content": user_msg})
                    if assistant_msg:
                        formatted_history.append({"role": "assistant", "content": assistant_msg})
                elif isinstance(h, dict):
                    formatted_history.append(h)

        # Update state with formatted history
        state["messages"] = formatted_history

        # Add the new user message to state
        state["messages"].append({"role": "user", "content": message})

        # Show typing indicator
        yield formatted_history, state, "typing-indicator active"

        # Process message
        updated_state = chat_with_user(message, state)

        # Get assistant response
        if updated_state and "messages" in updated_state and updated_state["messages"]:
            assistant_messages = [msg for msg in updated_state["messages"] if msg["role"] == "assistant"]
            if assistant_messages:
                assistant_message = assistant_messages[-1]["content"]
                updated_state["messages"] = state["messages"] + [{"role": "assistant", "content": assistant_message}]
            else:
                updated_state["messages"] = state["messages"] + [{"role": "assistant", "content": "I apologize, but I encountered an error processing your request."}]

        # Save to chat history
        updated_state["chat_history"] = updated_state["messages"]

        # Hide typing indicator
        yield updated_state["messages"], updated_state, "typing-indicator"

    except Exception as e:
        print(f"Error processing message: {e}")
        # Ensure we always return something even if there's an error
        if isinstance(history, list) and len(history) > 0:
            if isinstance(history[-1], dict) and history[-1]["role"] == "assistant":
                history[-1]["content"] = "I apologize, but I encountered an error processing your request."
            elif isinstance(history[-1], tuple) and history[-1][1] is None:
                history[-1] = (message, "I apologize, but I encountered an error processing your request.")
        yield history, state, "typing-indicator"

def clear_conversation(state: Dict[str, Any]) -> Tuple[List, Dict[str, Any]]:
    """Clear the conversation history."""
    new_state = reset_state()
    new_state["session_id"] = state.get("session_id", datetime.now().strftime("%Y%m%d%H%M%S"))
    new_state["type"] = "messages"  # Force correct type
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
    """Track an order by ID."""
    if not order_id.strip():
        return [], state, ""

    # Process and unpack the generator
    processed = process_message(
        f"What's the status of my order {order_id.strip()}?",
        state["chat_history"],
        state
    )
    return next(processed)  # Get the first yield value from generator

def get_faq_response(faq_key: str, state: Dict[str, Any]) -> Tuple[List, Dict[str, Any], str]:
    """Get a response for a FAQ."""
    # Define FAQ responses
    FAQ_RESPONSES = {
        "return_policy": """Our return policy is as follows:

1. Items can be returned within 30 days of delivery for a full refund.
2. Products must be in original packaging and unused condition.
3. For electronics, returns are accepted within 15 days and must include all accessories.
4. Shipping costs for returns are covered by the customer unless the item was defective.
5. Refunds are processed within 5-7 business days after we receive the returned item.

Would you like more information about a specific aspect of our return policy?""",

        "shipping_policy": """Our shipping policy:

1. Standard shipping (5-7 business days): Free for orders over $35, otherwise $4.99
2. Express shipping (2-3 business days): $9.99
3. Next-day delivery (where available): $19.99
4. International shipping available to select countries

Delivery times may vary based on your location and product availability. You can track your shipment using the order ID provided in your confirmation email.

Do you have any other questions about shipping?""",

        "payment_methods": """We accept the following payment methods:

1. Credit cards (Visa, Mastercard, American Express, Discover)
2. Debit cards
3. PayPal
4. Store credit/gift cards
5. Apple Pay and Google Pay (on mobile)

All payment information is securely processed and encrypted. We do not store your full credit card details on our servers.

Is there anything specific about our payment options you'd like to know?""",

        "contact_info": """Our contact information:

Customer Service Hours:
- Monday to Friday: 8:00 AM - 8:00 PM EST
- Saturday: 9:00 AM - 6:00 PM EST
- Sunday: 10:00 AM - 5:00 PM EST

Phone: +1-800-123-4567
Email: support@ecommerce-example.com
Live Chat: Available on our website during business hours

For the fastest response, please have your order number ready when contacting us.

Would you like me to connect you with a customer service representative?"""
    }

    if faq_key not in FAQ_RESPONSES:
        return state.get("chat_history", []), state, ""

    # Get the FAQ response
    response = FAQ_RESPONSES[faq_key]
    query = f"Tell me about your {faq_key.replace('_', ' ')}"

    # Update history and state based on the chatbot type
    if state.get("type") == "messages":
        # For messages format
        history = state.get("chat_history", [])
        history.append({"role": "user", "content": query})
        history.append({"role": "assistant", "content": response})
    else:
        # For tuples format
        history = state.get("chat_history", []) + [(query, response)]

    new_state = state.copy()
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
                height="100%",
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

# ===== Launch the app =====
if __name__ == "__main__":
    try:
        demo.launch(share=True)
    except Exception as e:
        print(f"Error launching Gradio app: {e}")
        raise
