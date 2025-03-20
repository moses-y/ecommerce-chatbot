import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import gradio as gr
from dotenv import load_dotenv
from src.chatbot import chat_with_user

load_dotenv()

# Define a dark minimalist theme using Gradio's theming capabilities
custom_theme = gr.themes.Base(
    primary_hue="gray",       # Dark gray for primary elements
    secondary_hue="blue",     # Blue accents for buttons
    neutral_hue="gray",       # Dark gray for backgrounds
    font=["Inter", "ui-sans-serif", "system-ui", "sans-serif"],  # Clean, readable font
).set(
    body_background_fill="#1C2526",         # Dark gray background for the body
    background_fill_primary="#2D3748",      # Slightly lighter gray for main areas
    button_primary_background_fill="#4B5EAA",  # Muted blue for buttons
    button_primary_background_fill_hover="#3B4A8A"  # Darker blue on hover
)

# Custom CSS to ensure white text and refine the UI
css = """
/* Center the app with padding for a spacious feel */
body {
    display: flex;
    justify-content: center;
    align-items: center;
    padding: 1.5rem;
    background-color: #1C2526;  /* Dark gray background */
    color: #FFFFFF !important;  /* White text for body */
}

/* Style the app container */
.gradio-container {
    max-width: 700px;
    width: 100%;
    background-color: #2D3748;  /* Lighter dark gray */
    padding: 1.5rem;
    border-radius: 8px;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
    color: #FFFFFF !important;  /* White text for container */
}

/* Chatbot area with smooth scrolling */
#chatbot {
    max-height: 60vh;
    overflow-y: auto;
    scroll-behavior: smooth;
    padding: 1rem;
    border-radius: 8px;
    background-color: #2D3748;  /* Dark gray background */
    color: #FFFFFF !important;  /* White text for chatbot */
}

/* User message styling */
#chatbot .message.user {
    background-color: #4A5568;  /* Light gray for user messages */
    border-radius: 8px;
    padding: 0.75rem;
    margin-bottom: 0.5rem;
    text-align: left;
    color: #FFFFFF !important;  /* White text */
}

/* Assistant message styling */
#chatbot .message.assistant {
    background-color: #3B4A8A;  /* Muted blue-gray for assistant messages */
    border-radius: 8px;
    padding: 0.75rem;
    margin-bottom: 0.5rem;
    text-align: left;
    color: #FFFFFF !important;  /* White text */
}

/* Smooth transitions for inputs and buttons */
input, button {
    transition: all 0.2s ease-in-out;  /* Fluid interactions */
}

/* Style input fields */
input {
    color: #FFFFFF !important;  /* White text in inputs */
    background-color: #4A5568;  /* Dark gray background */
    border: 1px solid #718096;  /* Subtle border */
}

/* Title styling */
.gradio-container .title {
    text-align: center;
    font-weight: 600;
    font-size: 1.5rem;
    margin-bottom: 0.5rem;
    color: #FFFFFF !important;  /* White text */
}

/* Subtitle styling */
.gradio-container .subtitle {
    text-align: center;
    font-size: 0.875rem;
    margin-bottom: 1.5rem;
    color: #E2E8F0 !important;  /* Slightly lighter white */
}

/* Ensure all text is white */
* {
    color: #FFFFFF !important;
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

# Process user message and generate assistant response
def respond(message, history):
    global chat_state
    # Reset state if starting a new conversation
    if not history:
        chat_state = reset_state()
    
    # Process the message with the chatbot logic
    chat_state = chat_with_user(message, chat_state)
    return chat_state["messages"][-1]["content"]

# Build the Gradio interface
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
        stream=True,  # Enable real-time streaming of responses
        concurrency_limit=10,  # Support up to 10 concurrent users
    )

if __name__ == "__main__":
    chat_state = reset_state()  # Initialize state
    demo.launch(share=True)