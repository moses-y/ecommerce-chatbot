# src/db/setup_db.py
import pandas as pd
import logging
import os
from sqlalchemy.orm import Session
from src.db.database import engine, Base, SessionLocal
from src.db.models import Order, ContactRequest # Import all models
from src.core.config import ORDERS_CSV_PATH, ORDERS_TABLE_NAME, DATA_DIR
import numpy as np # Import numpy

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_tables():
    """Creates database tables based on models."""
    logger.info("Attempting to create database tables...")
    try:
        # Ensure the directory for the database exists
        db_dir = os.path.dirname(os.path.abspath(engine.url.database))
        os.makedirs(db_dir, exist_ok=True)
        logger.info(f"Ensured database directory exists: {db_dir}")

        Base.metadata.create_all(bind=engine)
        logger.info("Database tables created successfully (if they didn't exist).")
        logger.info(f"Tables created: {Base.metadata.tables.keys()}")
    except Exception as e:
        logger.error(f"Error creating database tables: {e}", exc_info=True)
        raise

def load_orders_from_csv(db: Session):
    """Loads order data from CSV into the database, checking for existing data."""
    logger.info(f"Checking if data needs to be loaded from {ORDERS_CSV_PATH} into table '{ORDERS_TABLE_NAME}'...")

    try:
        # Check if the table already has data
        order_count = db.query(Order).count()
        if order_count > 0:
            logger.info(f"Table '{ORDERS_TABLE_NAME}' already contains {order_count} records. Skipping CSV load.")
            return

        logger.info(f"Table '{ORDERS_TABLE_NAME}' is empty. Attempting to load data from CSV...")

        if not os.path.exists(ORDERS_CSV_PATH):
            logger.error(f"CSV file not found at {ORDERS_CSV_PATH}. Cannot load orders.")
            return

        # Define date columns for parsing
        date_columns = [
            'order_purchase_timestamp',
            'order_approved_at',
            'order_delivered_carrier_date',
            'order_delivered_customer_date',
            'order_estimated_delivery_date'
        ]

        # --- Robust CSV Reading and Cleaning ---
        try:
            # Step 1: Read all columns as strings initially to prevent incorrect type inference.
            # Use keep_default_na=False and specify na_values to control NaN interpretation.
            df = pd.read_csv(
                ORDERS_CSV_PATH,
                dtype=str, # Read everything as string first
                keep_default_na=False, # Don't use default NaN interpretation
                na_values=['', '#N/A', '#N/A N/A', '#NA', '-1.#IND', '-1.#QNAN', '-NaN', '-nan', '1.#IND', '1.#QNAN', '<NA>', 'N/A', 'NA', 'NULL', 'NaN', 'n/a', 'nan', 'null'] # Define strings to treat as NaN
            )
            logger.info(f"Successfully read {len(df)} rows from {ORDERS_CSV_PATH} (initially as strings).")

            # Step 2: Convert specific columns to datetime, coercing errors to NaT.
            for col in date_columns:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col], errors='coerce', dayfirst=True) # Keep dayfirst=True
                    logger.debug(f"Converted column '{col}' to datetime (errors coerced to NaT).")
                else:
                    logger.warning(f"Date column '{col}' not found in CSV.")

            # Step 3: Replace all forms of null/NaN (including pandas NaT) with Python None.
            # This is crucial for SQLAlchemy compatibility.
            # Replace numpy NaN (just in case, though dtype=str should prevent float columns)
            df = df.replace({np.nan: None})
            # Replace pandas NaT (for datetimes)
            df = df.replace({pd.NaT: None})
            # Replace any remaining specified na_values (which were read as strings) with None
            for na_val in ['', '#N/A', '#N/A N/A', '#NA', '-1.#IND', '-1.#QNAN', '-NaN', '-nan', '1.#IND', '1.#QNAN', '<NA>', 'N/A', 'NA', 'NULL', 'NaN', 'n/a', 'nan', 'null']:
                 df = df.replace({na_val: None})

            logger.info("Replaced NaT and various null/NaN representations with Python None.")

        except Exception as e:
            logger.error(f"Error reading or processing CSV file {ORDERS_CSV_PATH}: {e}", exc_info=True)
            return
        # --- End Robust CSV Reading ---


        # Select only columns that exist in the Order model to avoid inserting extra columns
        model_columns = [c.name for c in Order.__table__.columns]
        df_columns_to_load = [col for col in df.columns if col in model_columns]
        if len(df_columns_to_load) < len(model_columns):
             missing_cols = set(model_columns) - set(df_columns_to_load)
             logger.warning(f"CSV is missing columns defined in the Order model: {missing_cols}")
        if len(df.columns) > len(df_columns_to_load):
             extra_cols = set(df.columns) - set(df_columns_to_load)
             logger.warning(f"CSV has extra columns not in the Order model (will be ignored): {extra_cols}")

        df_filtered = df[df_columns_to_load]
        logger.info(f"Filtered DataFrame to include only model columns: {df_columns_to_load}")


        # Convert DataFrame to list of dictionaries
        orders_data = df_filtered.to_dict(orient='records')

        # Bulk insert using SQLAlchemy Core
        if orders_data:
             # Debug: Log the first record and its types just before insertion
            if orders_data:
                sample_record = orders_data[0]
                logger.debug(f"Sample record data before insert: {sample_record}")
                logger.debug(f"Types in sample record: {{k: type(v) for k, v in sample_record.items()}}")

            logger.info(f"Attempting bulk insert of {len(orders_data)} records...")
            db.bulk_insert_mappings(Order, orders_data)
            db.commit()
            logger.info(f"Successfully loaded {len(orders_data)} orders into the database.")
        else:
            logger.info("No order data found in the CSV to load.")

    except Exception as e:
        logger.error(f"Error during database operation: {e}", exc_info=True) # Changed logging message slightly
        db.rollback() # Rollback in case of error during commit
        raise
    finally:
        # The session is managed outside this function when called from main
        pass

# --- Main execution block ---
if __name__ == "__main__":
    # Configure logging specifically for standalone execution
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger.info("--- Database Setup Script Started (Standalone Execution) ---")

    # Ensure data directory exists (redundant if config does it, but safe)
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        logger.info(f"Created data directory: {DATA_DIR}")

    create_tables() # Create tables first

    # Get a new session specifically for the setup script
    db_session = SessionLocal()
    try:
        load_orders_from_csv(db=db_session) # Load data using the session
    except Exception as e:
        logger.error(f"An error occurred during the setup process: {e}")
    finally:
        db_session.close() # Ensure the session is closed
        logger.info("--- Database Setup Script Finished (Standalone Execution) ---")