# Development Setup Guide

This guide explains how to set up the project for local development and contribution.

## Prerequisites

*   Python 3.11+ (as used in testing/deployment)
*   Git
*   Access to a Google Gemini API Key

## Installation

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
    *(This reads the `requirements.txt` file in the project root)*

5.  **Set up environment variables:**
    *   Copy the example environment file:
        *   Windows: `copy .env.example .env`
        *   macOS/Linux: `cp .env.example .env`
    *   Edit the newly created `.env` file and add your `GOOGLE_API_KEY`.
    *   Configure `DATABASE_URL` if you are not using the default SQLite DB (`sqlite:///./data/chatbot_data.db`). The default path assumes the `data` directory exists in the root.

6.  **Initialize the database (if using the default SQLite and it doesn't exist):**
    *   Ensure the `data/` directory exists: `mkdir data` (if needed)
    *   Run the setup script:
        ```bash
        python src/db/setup_db.py
        ```
    *(Note: If you implement Alembic migrations later, update this step accordingly)*

## Running the Application

```bash
python app.py
```
The Gradio interface should launch. Access it via the URL provided in the console (typically http://127.0.0.1:7860).

Running Tests
Ensure all tests pass before committing changes.

```bash
pytest -v
```