# Add to app.py before creating the Gradio interface
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import gradio as gr
from dotenv import load_dotenv
from src.chatbot import chat_with_user
from src.utils import initialize_vector_db
import time
print("Initializing vector database...")
start_time = time.time()

print(f"Vector database initialized in {time.time() - start_time:.2f} seconds")

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

# Updated CSS for a professional and user-friendly interface
css = """
/* Center the app with padding for a spacious feel */
body {
    display: flex;
    justify-content: center;
    align-items: center;
    padding: 0.5rem;
    background-color: #1C2526;  /* Dark gray background */
    color: #FFFFFF !important;  /* White text for body */
}

/* Style the app container with subtle shadow for depth */
.gradio-container {
    max-width: 700px;
    width: 100%;
    background-color: #2D3748;  /* Lighter dark gray */
    padding: 0.5rem;
    border-radius: 5px;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);  /* Subtle shadow */
    color: #FFFFFF !important;  /* White text for container */
}

/* Chatbot area with increased height and smooth scrolling */
#chatbot {
    max-height: 70vh;  /* Taller chat area for more visibility */
    overflow-y: auto;
    scroll-behavior: smooth;
    padding: 0.5rem;
    border-radius: 8px;
    background-color: #2D3748;  /* Dark gray background */
    color: #FFFFFF !important;  /* White text for chatbot */
}

/* User message styling with right alignment and speech bubble effect */
#chatbot .message.user {
    background-color: #4A5568;  /* Light gray for user messages */
    border-radius: 12px 12px 0 12px;  /* Flat bottom-right corner */
    padding: 0.75rem;
    margin-bottom: 0.5rem;  /* Increased spacing */
    margin-left: auto;  /* Align to right */
    max-width: 80%;  /* Limit width */
    text-align: left;
    color: #FFFFFF !important;  /* White text */
}

/* Assistant message styling with left alignment and speech bubble effect */
#chatbot .message.assistant {
    background-color: #3B4A8A;  /* Muted blue-gray for assistant messages */
    border-radius: 12px 12px 12px 0;  /* Flat bottom-left corner */
    padding: 0.75rem;
    margin-bottom: 1rem;  /* Increased spacing */
    margin-right: auto;  /* Align to left */
    max-width: 80%;  /* Limit width */
    text-align: left;
    color: #FFFFFF !important;  /* White text */
}

/* Smooth transitions for inputs and buttons */
input, button {
    transition: all 0.2s ease-in-out;  /* Fluid interactions */
}

/* Style input fields with more padding */
input {
    color: #FFFFFF !important;  /* White text in inputs */
    background-color: #4A5568;  /* Dark gray background */
    border: 1px solid #718096;  /* Subtle border */
    padding: 0.5rem;  /* Increased padding for comfort */
}

/* Focus styles for accessibility */
input:focus, button:focus {
    outline: 2px solid #4B5EAA;  /* Blue outline on focus */
    outline-offset: 2px;
}

/* Title styling */
.gradio-container .title {
    text-align: center;
    font-weight: 600;
    font-size: 1rem;
    margin-bottom: 0.2rem;
    color: #FFFFFF !important;  /* White text */
}

/* Subtitle styling */
.gradio-container .subtitle {
    text-align: center;
    font-size: 0.5rem;
    margin-bottom: 0.2rem;
    color: #E2E8F0 !important;  /* Slightly lighter white */
}

/* Ensure all text is white */
* {
    color: #FFFFFF !important;
}

/* Responsive design for smaller screens */
@media (max-width: 600px) {
    body {
        padding: 0.5rem;  /* Reduced padding */
    }
    .gradio-container {
        padding: 1rem;
    }
    #chatbot {
        max-height: 50vh;  /* Shorter chat area on mobile */
    }
    .title {
        font-size: 1.25rem;  /* Smaller title */
    }
    .subtitle {
        font-size: 0.75rem;  /* Smaller subtitle */
    }
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
        type='messages',
        concurrency_limit=10,
    )

if __name__ == "__main__":
    chat_state = reset_state()  # Initialize state
    demo.launch(share=True)