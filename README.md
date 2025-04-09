---
title: E-commerce Chatbot
emoji: 🛒
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 5.22.0
app_file: app.py
pinned: false
---

# Ari - E-commerce Customer Support Chatbot 🛒

[![CI/CD Status](https://github.com/moses-y/ecommerce-chatbot/actions/workflows/ci.yml/badge.svg)](https://github.com/moses-y/ecommerce-chatbot/actions/workflows/ci.yml)

An intelligent chatbot designed to assist customers with common e-commerce inquiries using Google Gemini, SQLAlchemy for database interaction, and Gradio for the user interface.

## ✨ Features

*   **Order Status Checking:** Retrieve order status using 32-character alphanumeric IDs via database lookup.
*   **Return Policy Information:** Provide details based on the content in `data/policies.json`.
*   **Human Handoff:** Guide users through providing contact details for human support follow-up.
*   **Natural Language Understanding:** Powered by Google Gemini for intent detection.
*   **Modular Architecture:** Built with distinct agents (`src/agents/`), services (`src/services/`), and core logic (`src/core/`) for maintainability.
*   **Web Interface:** User-friendly chat interface provided by Gradio (`src/ui/gradio_app.py`).

## 🚀 Live Demo

Try Ari live on Hugging Face Spaces:

**[➡️ Launch Chatbot Demo](https://huggingface.co/spaces/MoeTensors/E-commerce_chatbot)**

## 📚 Documentation

For full details on usage, setup, architecture, testing, and deployment, please refer to the documentation site built with MkDocs:

**[➡️ View Documentation](https://moses-y.github.io/ecommerce-chatbot/)**

## 🏗️ Project Structure

```ecommerce-chatbot/ 
├── .github/ 
│ └── workflows/ 
│ └── ci.yml # GitHub Actions CI/CD workflow 
├── assets/ # Static assets (icons) for UI │ ├── bot-icon.png 
│ └── user-icon.png 
├── data/ 
│ ├── policies.json # Return policy information 
│ └── chatbot_data.db # Default SQLite database file 
├── docs/ # Documentation files (Markdown) │ ├── index.md 
│ ├── usage.md 
│ └── development/ 
│ ├── setup.md 
│ ├── architecture.md 
│ ├── testing.md 
│ └── deployment.md 
├── src/ 
│ ├── agents/ # Intent-specific processing logic │ 
│ ├── init.py │
│ ├── base_agent.py │ 
│ ├── order_status_agent.py │ 
│ ├── return_policy_agent.py │ 
│ └── human_rep_agent.py │ 
├── core/ # Core conversation management │ │ ├── init.py │ 
│ ├── conversation.py # ConversationManager class │ 
│ └── state.py # ConversationState class 
│ ├── db/ # Database interaction (SQLAlchemy) │ 
│ ├── init.py │ 
│ ├── database.py │ 
│ ├── models.py │ 
│ └── setup_db.py │ 
├── llm/ # Language model interaction │ 
│ ├── init.py │ 
│ └── gemini_service.py │ 
├── services/ # Business logic (data access) │ 
│ ├── init.py │ 
│ ├── order_service.py │ 
│ └── policy_service.py 
│ ├── ui/ # User interface (Gradio) │ 
│ ├── init.py │ 
│ └── gradio_app.py 
│ └── utils/ # Utility functions 
│ ├── init.py 
│ └── helpers.py 
├── tests/ # Automated tests (pytest) 
│ ├── init.py 
│ ├── conftest.py # Pytest fixtures and mocks 
│ └── test_main_flows.py # Integration and utility tests 
├── .env.example # Example environment variables 
├── .gitignore # Files ignored by Git 
├── app.py # Main application entry point ├── Dockerfile # Docker configuration for deployment 
├── docker-compose.yml # Docker Compose for local setup 
├── LICENSE # Project License (MIT) 
├── mkdocs.yml # MkDocs configuration 
└── requirements.txt # Python dependencies
```


## ⚙️ Setup Instructions (Developers)

### Prerequisites

*   Python 3.11+
*   Git
*   Google Gemini API Key

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/moses-y/ecommerce-chatbot.git
    cd ecommerce-chatbot
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    # Windows
    python -m venv venv
    .\venv\Scripts\activate

    # macOS/Linux
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install `uv` (fast package installer):**
    ```bash
    pip install uv
    ```

4.  **Install dependencies using `uv`:**
    ```bash
    uv pip install -r requirements.txt
    ```

5.  **Set up environment variables:**
    *   Copy the example environment file:
        *   Windows: `copy .env.example .env`
        *   macOS/Linux: `cp .env.example .env`
    *   Edit the newly created `.env` file and add your `GOOGLE_API_KEY`.
    *   *(Optional)* Modify `DATABASE_URL` if you want to use a different database (default is SQLite in `data/chatbot_data.db`).

6.  **Initialize the database (if using default SQLite):**
    *   Ensure the `data/` directory exists: `mkdir data` (if needed).
    *   Run the setup script:
        ```bash
        python src/db/setup_db.py
        ```

## ▶️ Running the Web Interface

To start the chatbot with the Gradio web interface:

```bash
python app.py
```

This will launch a local web server. Access the interface via the URL provided in the console (e.g., http://127.0.0.1:7860).

✅ Running Tests

Run the automated test suite using pytest:
```
pytest -v
```
🐳 Running with Docker (Local)
Build the image:
```
docker build -t ecommerce-chatbot:latest .
```
Run the container (using your local .env file):
```
docker run -p 7860:7860 -d --name ari-chatbot-local --env-file .env ecommerce-chatbot:latest
```

Access at http://localhost:7860.
🐳 Running with Docker Compose (Local)
Ensure your .env file is configured.

Run:
```
docker-compose up -d
```
Access at http://localhost:7860.
📄 License

This project is licensed under the MIT License - see the LICENSE file for details.