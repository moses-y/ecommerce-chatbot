import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import gradio as gr
from dotenv import load_dotenv
from src.chatbot import chat_with_user

load_dotenv()

# Minimalist theme with neutral colors
custom_theme = gr.themes.Base(
    primary_hue="slate",      # Neutral primary color
    secondary_hue="blue",     # Secondary color for accents
    neutral_hue="slate",      # Neutral hues for backgrounds
    font=["Inter", "ui-sans-serif", "system-ui", "sans-serif"],  # Simple, readable font
).set(
    body_background_fill="#f8fafc",  # Light background for spacious feel
    background_fill_primary="#ffffff",  # White for main content areas
    button_primary_background_fill="#6366f1",  # Blue for buttons
    button_primary_background_fill_hover="#4f46e5",  # Darker blue on hover
)

# Updated CSS for smooth, minimalist design with improved font colors
css = """
/* Center the entire app with padding for spaciousness */
body {
    display: flex;
    justify-content: center;
    align-items: center;
    padding: 2rem;
    background-color: #f8fafc;
}

/* App container with max width for better readability */
.gradio-container {
    max-width: 700px;
    width: 100%;
    background-color: #ffffff;
    padding: 2rem;
    border-radius: 8px;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
}

/* Chatbot area with smooth scrolling */
#chatbot {
    max-height: 60vh;  /* Flexible height */
    overflow-y: auto;
    scroll-behavior: smooth;  /* Smooth scrolling */
    padding: 1rem;
    border-radius: 8px;
    background-color: #f8fafc;
}

/* Message styling for user and assistant with updated font colors */
.message.user {
    background-color: #e2e8f0;  /* Light gray for user messages */
    border-radius: 5px;
    padding: 0.75rem;
    margin-bottom: 0.5rem;
    text-align: left;
    color: #333333;  /* Dark gray for better readability */
}

.message.assistant {
    background-color: #ffffff;  /* White for assistant messages */
    border-radius: 5px;
    padding: 0.75rem;
    margin-bottom: 0.5rem;
    text-align: left;
    color: #83C5BE;  /* Light teal for a modern touch */
}

/* Smooth transitions for input and buttons */
input, button {
    transition: all 0.2s ease-in-out;  /* Fluid interactions */
}

/* Title and subtitle styling with improved font colors */
.title {
    text-align: center;
    font-weight: 600;
    font-size: 1.5rem;  /* Smaller, minimalist font */
    margin-bottom: 0.5rem;
    color: #000000;  /* Black for readability */
}

.subtitle {
    text-align: center;
    font-size: 0.875rem;
    margin-bottom: 1.5rem;
    color: #333333;  /* Dark gray for readability */
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
    )

if __name__ == "__main__":
    chat_state = reset_state()  # Initial state
    demo.launch(share=True)