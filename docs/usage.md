# User Guide

Interacting with the Ari chatbot is simple and intuitive.

## Accessing the Chatbot

You can use the live demo hosted on Hugging Face Spaces:

**[➡️ Launch Live Demo](https://huggingface.co/spaces/MoeTensors/E-commerce_chatbot)**

Alternatively, if running locally, access it via the URL provided when you run `python app.py` (usually `http://127.0.0.1:7860`).

## Common Interactions

*   **Checking Order Status:**
    *   You can say: "Check my order status", "Where is my order?", "Status for order [your 32-char order ID]"
    *   If you don't provide the ID initially, the bot will ask for it. Please provide the full 32-character alphanumeric ID.
    *   *Example ID format:* `abc123def456ghi789jkl012mno345p0`

*   **Asking about Returns:**
    *   You can say: "What's your return policy?", "How do I return an item?"
    *   The bot will provide the relevant policy information stored in its configuration.

*   **Requesting Human Help:**
    *   You can say: "Talk to a person", "Connect me to support", "I need human help"
    *   The bot will guide you through providing contact information to facilitate a callback or email from a support representative.

## Tips

*   Be clear and concise in your requests.
*   Provide the full 32-character order ID when asked.
*   Use the "Clear Conversation" button if you want to start fresh.