# Welcome to Ari - The E-commerce Chatbot

Ari is an intelligent chatbot designed to assist customers with common e-commerce inquiries, such as checking order status and understanding return policies. It leverages Google Gemini for natural language understanding and provides a user-friendly interface built with Gradio.

This documentation provides information for both users and developers.

## ✨ Key Features

*   **Order Status Checking:** Retrieve real-time status updates for orders using a 32-character alphanumeric order ID.
*   **Return Policy Information:** Get clear explanations of the store's return policy from `data/policies.json`.
*   **Human Handoff:** Seamlessly guide users through the process of requesting contact with a human representative.
*   **Natural Language Understanding:** Powered by Google Gemini to understand user intent.
*   **Modular Agent Architecture:** Easily extendable with new capabilities (`src/agents/`).
*   **Database Integration:** Uses SQLAlchemy to interact with a database for order information (`src/db/`, `src/services/order_service.py`).

## 🚀 Try the Chatbot!

You can interact with a live version of Ari hosted on Hugging Face Spaces:

**[➡️ Launch Live Demo](https://huggingface.co/spaces/MoeTensors/E-commerce_chatbot)**

## 📖 Getting Started

*   **Users:** See the [User Guide](usage.md) for instructions on how to interact with the chatbot.
*   **Developers:** Refer to the [Development Setup](development/setup.md) guide to get the project running locally. Explore the [Architecture](development/architecture.md) to understand how it works.