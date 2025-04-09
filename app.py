# app.py (in project root)
import logging
import os
from src.ui.gradio_app import main

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Ensure required directories exist
    os.makedirs("data", exist_ok=True)
    os.makedirs("assets", exist_ok=True)

    # Launch the Gradio app
    main()