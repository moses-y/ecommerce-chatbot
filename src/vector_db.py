# src/vector_db.py
import os
import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from typing import List, Dict, Any, Optional
from functools import lru_cache

# Load environment variables
load_dotenv()

# Singleton instance for vector database
_VECTOR_DB_INSTANCE = None

class LocalEmbeddingFunction:
    """Custom embedding function using sentence-transformers"""
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')  # Lightweight model

    def __call__(self, input):
        """Generate embeddings for a list of texts"""
        if not input:
            return []
        cleaned_texts = [text.strip() for text in input if text and text.strip()]
        if not cleaned_texts:
            return []
        return self.model.encode(cleaned_texts).tolist()

def get_vector_db_instance(orders_df: Optional[pd.DataFrame] = None,
                          db_path: str = "./data/vector_db",
                          use_subset: bool = False,
                          force_refresh: bool = False) -> chromadb.Collection:
    """
    Get or create the vector database instance (Singleton pattern).

    Args:
        orders_df: DataFrame containing order data (optional)
        db_path: Path to store the vector database
        use_subset: Whether to use a subset of the data for development
        force_refresh: Whether to force a refresh of the database

    Returns:
        ChromaDB collection with order data
    """
    global _VECTOR_DB_INSTANCE

    # Return existing instance if available and not forcing refresh
    if _VECTOR_DB_INSTANCE is not None and not force_refresh:
        return _VECTOR_DB_INSTANCE

    # Create new instance
    _VECTOR_DB_INSTANCE = create_order_vector_db(orders_df, db_path, use_subset)
    return _VECTOR_DB_INSTANCE

def create_order_vector_db(orders_df: Optional[pd.DataFrame],
                          db_path: str = "./data/vector_db",
                          use_subset: bool = False) -> chromadb.Collection:
    """
    Create a vector database from order data using local embeddings.

    Args:
        orders_df: DataFrame containing order data (optional)
        db_path: Path to store the vector database
        use_subset: Whether to use a subset of the data for development

    Returns:
        ChromaDB collection with order data
    """
    # Ensure directory exists
    os.makedirs(db_path, exist_ok=True)

    # Initialize ChromaDB client with settings
    client = chromadb.PersistentClient(
        path=db_path,
        settings=chromadb.Settings(
            anonymized_telemetry=False,
            allow_reset=True
        )
    )

    # Try to get existing collection first
    try:
        collection = client.get_collection(
            name="orders",
            embedding_function=LocalEmbeddingFunction()
        )
        print("Using existing vector database")
        return collection
    except Exception as e:
        print(f"Could not load existing collection: {e}")
        pass

    # If we don't have order data and couldn't load existing collection, we can't proceed
    if orders_df is None:
        raise ValueError("Orders DataFrame is required to create a new vector database")

    # Use a subset of the data if specified
    if use_subset:
        orders_df = orders_df.head(1000)
        print(f"Using subset of {len(orders_df)} orders for development")
    else:
        print(f"Processing full dataset of {len(orders_df)} orders")

    # Try to delete if it exists but is corrupted
    try:
        client.delete_collection("orders")
        print("Deleted existing collection to create fresh one")
    except Exception:
        pass

    # Create a new collection with optimized parameters
    try:
        collection = client.create_collection(
            name="orders",
            embedding_function=LocalEmbeddingFunction(),
            metadata={"hnsw:M": 16, "hnsw:construction_ef": 100}
        )
        print("Creating new vector database")

        # Prepare data for insertion
        documents = []
        metadatas = []
        ids = []
        batch_size = 5000  # Batch size for ChromaDB

        total_orders = len(orders_df)
        for idx, row in orders_df.iterrows():
            # Create a text representation of the order for embedding
            order_text = f"""
            Order ID: {row['order_id']}
            Customer ID: {row.get('customer_id', 'Unknown')}
            Status: {row['order_status']}
            Purchase Date: {row['order_purchase_timestamp']}
            """

            # Add optional fields if available
            if pd.notna(row.get('order_approved_at')):
                order_text += f"Approved Date: {row['order_approved_at']}\n"
            if pd.notna(row.get('order_delivered_carrier_date')):
                order_text += f"Carrier Delivery Date: {row['order_delivered_carrier_date']}\n"
            if pd.notna(row.get('order_delivered_customer_date')):
                order_text += f"Customer Delivery Date: {row['order_delivered_customer_date']}\n"
            if pd.notna(row.get('order_estimated_delivery_date')):
                order_text += f"Estimated Delivery Date: {row['order_estimated_delivery_date']}\n"

            # Store the document, metadata, and ID
            documents.append(order_text)
            metadatas.append({
                "order_id": row['order_id'],
                "customer_id": str(row.get('customer_id', '')),
                "status": row['order_status'],
                "purchase_date": row['order_purchase_timestamp']
            })
            ids.append(str(idx))

            # Add in larger batches to reduce indexing overhead
            if len(documents) >= batch_size:
                collection.add(
                    documents=documents,
                    metadatas=metadatas,
                    ids=ids
                )
                print(f"Indexed {min(idx + 1, total_orders)} of {total_orders} orders")
                documents = []
                metadatas = []
                ids = []

        # Add any remaining documents
        if documents:
            collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            print(f"Indexed {total_orders} of {total_orders} orders")

        return collection
    except Exception as e:
        print(f"Error creating or populating vector database: {e}")
        # ADD proper logging
        import logging
        logger = logging.getLogger(__name__)
        logger.error(f"Failed to create/populate vector DB: {e}", exc_info=True)
        # REMOVE or comment out the fallback empty collection:
        # collection = client.create_collection(name="orders")
        # print("Created empty collection as fallback")
        # return collection
        raise # Re-raise the exception

def query_vector_db(query: str, collection: Optional[chromadb.Collection] = None, top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Query the vector database for relevant orders.

    Args:
        query: User query text
        collection: ChromaDB collection to query (optional, uses singleton if None)
        top_k: Number of results to return

    Returns:
        List of relevant order data
    """
    if collection is None:
        collection = get_vector_db_instance()

    try:
        # Query the collection
        results = collection.query(
            query_texts=[query],
            n_results=top_k
        )

        # Format results
        formatted_results = []
        for i in range(len(results["ids"][0])):
            formatted_results.append({
                "id": results["ids"][0][i],
                "metadata": results["metadatas"][0][i],
                "document": results["documents"][0][i],
                "distance": results["distances"][0][i] if "distances" in results else None
            })

        return formatted_results
    except Exception as e:
        print(f"Error querying vector database: {e}")
        return []

def get_order_by_id(order_id: str, collection: Optional[chromadb.Collection] = None) -> Dict[str, Any]:
    """
    Get order details by order ID.

    Args:
        order_id: Order ID to look up
        collection: ChromaDB collection to query (optional, uses singleton if None)

    Returns:
        Order details or None if not found
    """
    if collection is None:
        collection = get_vector_db_instance()

    try:
        # Query by metadata
        results = collection.query(
            query_texts=[""],
            where={"order_id": order_id},
            n_results=1
        )

        if results["ids"][0]:
            return {
                "id": results["ids"][0][0],
                "metadata": results["metadatas"][0][0],
                "document": results["documents"][0][0]
            }
        return None
    except Exception as e:
        print(f"Error getting order by ID: {e}")
        return None

def get_orders_by_customer_id(customer_id: str, collection: Optional[chromadb.Collection] = None) -> List[Dict[str, Any]]:
    """
    Get all orders for a customer.

    Args:
        customer_id: Customer ID to look up
        collection: ChromaDB collection to query (optional, uses singleton if None)

    Returns:
        List of order details
    """
    if collection is None:
        collection = get_vector_db_instance()

    try:
        # Query by metadata
        results = collection.query(
            query_texts=[""],
            where={"customer_id": customer_id},
            n_results=100  # Set a high limit to get all orders
        )

        formatted_results = []
        for i in range(len(results["ids"][0])):
            formatted_results.append({
                "id": results["ids"][0][i],
                "metadata": results["metadatas"][0][i],
                "document": results["documents"][0][i]
            })

        return formatted_results
    except Exception as e:
        print(f"Error getting orders by customer ID: {e}")
        return []

@lru_cache(maxsize=1000)
def cached_get_order_by_id(order_id, collection=None):
    """Cached version of get_order_by_id using lru_cache."""
    # Directly call the function; lru_cache handles the rest.
    result = get_order_by_id(order_id, collection)
    return result

@lru_cache(maxsize=1000)
def cached_get_orders_by_customer_id(customer_id, collection=None):
    """Cached version of get_orders_by_customer_id using lru_cache."""
    # Directly call the function; lru_cache handles the rest.
    result = get_orders_by_customer_id(customer_id, collection)
    return result
