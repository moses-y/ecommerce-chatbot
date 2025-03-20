import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import gradio as gr
from dotenv import load_dotenv
from src.chatbot import chat_with_user

load_dotenv()

# Dark minimalist theme using Gradio's 2025 theming capabilities
custom_theme = gr.themes.Base(
    primary_hue="gray",       # Dark gray for primary elements
    secondary_hue="blue",     # Blue for accents (buttons)
    neutral_hue="gray",       # Dark gray for backgrounds
    font=["Inter", "ui-sans-serif", "system-ui", "sans-serif"],  # Clean, readable font
).set(
    body_background_fill="#1C2526",  # Dark gray background
    background_fill_primary="#2D3748",  # Slightly lighter dark gray for main areas
    button_primary_background_fill="#4B5EAA",  # Muted blue for buttons
    button_primary_background_fill_hover="#3B4A8A",  # Darker blue on hover
    text_color="#FFFFFF",  # White text for all elements
)

# Simplified CSS for a dark minimalist design with white fonts
css = """
/* Center the entire app with padding for spaciousness */
body {
    display: flex;
    justify-content: center;
    align-items: center;
    padding: 1.5rem;
    background-color: #1C2526;  /* Dark gray background */
}

/* App container with max width for better readability */
.gradio-container {
    max-width: 700px;
    width: 100%;
    background-color: #2D3748;  /* Slightly lighter dark gray */
    padding: 1.5rem;
    border-radius: 8px;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
}

/* Chatbot area with smooth scrolling */
#chatbot {
    max-height: 60vh;  /* Flexible height */
    overflow-y: auto;
    scroll-behavior: smooth;  /* Smooth scrolling */
    padding: 1rem;
    border-radius: 8px;
    background-color: #2D3748;  /* Dark gray background */
}

/* Message styling for user and assistant with white fonts */
#chatbot .message.user {
    background-color: #4A5568;  /* Slightly lighter gray for user messages */
    border-radius: 8px;
    padding: 0.75rem;
    margin-bottom: 0.5rem;
    text-align: left;
    color: #FFFFFF !important;  /* White text for readability */
}

#chatbot .message.assistant {
    background-color: #3B4A8A;  /* Muted blue-gray for assistant messages */
    border-radius: 8px;
    padding: 0.75rem;
    margin-bottom: 0.5rem;
    text-align: left;
    color: #FFFFFF !important;  /* White text for readability */
}

/* Smooth transitions for input and buttons */
input, button {
    transition: all 0.2s ease-in-out;  /* Fluid interactions */
}

/* Ensure input text is white for readability */
input {
    color: #FFFFFF !important;  /* White text in input fields */
    background-color: #4A5568;  /* Dark gray background for input */
    border: 1px solid #718096;  /* Subtle border */
}

/* Title and subtitle styling with white fonts */
.gradio-container .title {
    text-align: center;
    font-weight: 600;
    font-size: 1.5rem;
    margin-bottom: 0.5rem;
    color: #FFFFFF !important;  /* White for readability */
}

.gradio-container .subtitle {
    text-align: center;
    font-size: 0.875rem;
    margin-bottom: 1.5rem;
    color: #E2E8F0 !important;  /* Slightly lighter white for subtitle */
}
"""

# Initialize or reset chat state
def reset_state():
    return {
        "messages": [],
        "order_lookup_attempted": False,
        "current_order_id": None,
        "needs_human_agent": False,
        "contact_info_collected": False,
        "customer_name": None,
        "customer_email": None,
        "customer_phone": None,
        "contact_step": 0
    }

# Process user message and return assistant response
def respond(message, history):
    global chat_state
    # Reset state for new conversation if history is empty
    if not history:
        chat_state = reset_state()
    
    # Process the message
    chat_state = chat_with_user(message, chat_state)
    return chat_state["messages"][-1]["content"]

# Build the Gradio app with updated UI/UX
with gr.Blocks(theme=custom_theme, css=css) as demo:
    gr.Markdown("# E-commerce Support Chatbot", elem_classes="title")
    gr.Markdown("Ask about order status, return policies, or speak with a human.", elem_classes="subtitle")

    chatbot = gr.ChatInterface(
        respond,
        chatbot=gr.Chatbot(elem_id="chatbot", show_copy_button=True),
        examples=[
            "What's the status of my order e481f51cbdc54678b7cc49136f2d6af7?",
            "What is the status of my order e69bfb5eb88e0ed6a785585b27e16dbf?",
            "What is your return policy for electronics?",
            "I need to speak to a human representative."
        ],
        title="",
        stream=True,  # Enable streaming outputs for real-time rendering
        concurrency_limit=10,  # Allow queuing for up to 10 concurrent users
    )

if __name__ == "__main__":
    chat_state = reset_state()  # Initial state
    demo.launch(share=True)