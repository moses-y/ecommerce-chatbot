# app.py is the main entry point for the Gradio application.
import logging
import os
from pathlib import Path
from sqlalchemy import text, select, func
from src.ui.gradio_app import main
from src.db.setup_db import create_tables, load_orders_from_csv
from src.db.database import SessionLocal
from src.db.models import Order  # Import the Order model

def setup_logging():
    """Configure logging for the application"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def ensure_database():
    """Ensure database is set up properly"""
    logger = logging.getLogger(__name__)

    try:
        create_tables()
        db = SessionLocal()
        try:
            # Use SQLAlchemy's ORM query instead of raw SQL
            order_count = db.query(func.count(Order.order_id)).scalar()

            if order_count == 0:
                logger.info("No orders found in database, loading initial data...")
                load_orders_from_csv(db)
            else:
                logger.info(f"Database already contains {order_count} orders")
        finally:
            db.close()
    except Exception as e:
        logger.error(f"Database setup error: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    setup_logging()
    logger = logging.getLogger(__name__)

    try:
        # Ensure data directory exists
        os.makedirs("data", exist_ok=True)
        os.makedirs("assets", exist_ok=True)

        # Always ensure database is set up
        ensure_database()

        # Launch the Gradio app
        logger.info("Starting Gradio application...")
        main()
    except Exception as e:
        logger.error(f"Application startup error: {e}", exc_info=True)
        raise