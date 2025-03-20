import os
import sys
# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import gradio as gr
from dotenv import load_dotenv
from src.chatbot import chat_with_user

# Load environment variables
load_dotenv()

# Initialize chat state
chat_state = {
    "messages": [],
    "order_lookup_attempted": False,
    "current_order_id": None,
    "needs_human_agent": False,
    "contact_info_collected": False,
    "customer_name": None,
    "customer_email": None,
    "customer_phone": None,
    "contact_collection_step": 0  # Add a step counter for contact collection
}

# Create a custom theme
custom_theme = gr.themes.Base(
    primary_hue="indigo",
    secondary_hue="purple",
    neutral_hue="slate",
    font=["Inter", "ui-sans-serif", "system-ui", "sans-serif"],
).set(
    body_background_fill="#f8fafc",
    background_fill_primary="#ffffff",
    button_primary_background_fill="#6366f1",
    button_primary_background_fill_hover="#4f46e5",
)

def respond(message, history):
    """Process user message and return assistant response."""
    global chat_state

    # Process the message
    chat_state = chat_with_user(message, chat_state)

    # Get the latest assistant message
    return chat_state["messages"][-1]["content"]

# Custom CSS for responsive design
css = """
#chatbot {
    height: calc(100vh - 280px) !important;
    min-height: 300px;
}
.title {
    text-align: center !important;
    font-weight: 600 !important;
    font-size: 1.75rem !important;
    margin-bottom: 0.5rem !important;
    color: #4f46e5 !important;
}
.subtitle {
    text-align: center !important;
    font-size: 1rem !important;
    margin-bottom: 1.5rem !important;
    color: #6b7280 !important;
}
"""

# Create Gradio interface
with gr.Blocks(theme=custom_theme, css=css) as demo:
    gr.Markdown("# E-commerce Customer Support Chatbot", elem_classes="title")
    gr.Markdown("Ask about order status, return policies, or request to speak with a human representative.", elem_classes="subtitle")

    chatbot = gr.ChatInterface(
        respond,
        chatbot=gr.Chatbot(elem_id="chatbot", show_copy_button=True),
        examples=[
            "What's the status of my order e481f51cbdc54678b7cc49136f2d6af7?",
            "What is your return policy for electronics?",
            "I need to speak to a human representative."
        ],
        title="",
    )

# Launch the app
if __name__ == "__main__":
    demo.launch(share=True)