import os
import pandas as pd
import numpy as np
import chromadb
import google.generativeai as genai
from dotenv import load_dotenv
from typing import List, Dict, Any
from functools import lru_cache
# Load environment variables
load_dotenv()

# Configure Gemini API
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

class GeminiEmbeddingFunction:
    """Custom embedding function using Gemini API"""
    def __call__(self, input):
        """Generate embeddings for a list of texts"""
        if not input:
            return []

        # Clean and prepare texts
        cleaned_texts = [text.strip() for text in input if text and text.strip()]
        if not cleaned_texts:
            return []

        try:
            # Use Gemini embedding model
            model = "models/embedding-001"
            embeddings = []

            # Process texts in batches to avoid API limits
            batch_size = 20
            for i in range(0, len(cleaned_texts), batch_size):
                batch = cleaned_texts[i:i+batch_size]
                batch_embeddings = [
                    genai.embed_content(model=model, content=text)["embedding"]
                    for text in batch
                ]
                embeddings.extend(batch_embeddings)

            return embeddings
        except Exception as e:
            print(f"Error generating embeddings: {e}")
            # Return zero embeddings as fallback
            return [np.zeros(768) for _ in cleaned_texts]

def create_order_vector_db(orders_df: pd.DataFrame, db_path: str = "./data/vector_db") -> chromadb.Collection:
    """
    Create a vector database from order data.

    Args:
        orders_df: DataFrame containing order data
        db_path: Path to store the vector database

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

    # Try to get existing collection or create a new one
    try:
        # First try to delete if it exists but is corrupted
        try:
            client.delete_collection("orders")
            print("Deleted existing collection to create fresh one")
        except Exception as e:
            pass

        # Create a new collection
        collection = client.create_collection(
            name="orders",
            embedding_function=GeminiEmbeddingFunction()
        )
        print("Creating new vector database")

        # Prepare data for insertion
        documents = []
        metadatas = []
        ids = []

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

            # Add in batches to avoid memory issues
            if len(documents) >= 1000:
                collection.add(
                    documents=documents,
                    metadatas=metadatas,
                    ids=ids
                )
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

        return collection
    except Exception as e:
        print(f"Error creating vector database: {e}")
        # Try to get the collection without embedding function as fallback
        try:
            collection = client.get_collection(name="orders")
            print("Using existing collection without custom embedding function")
            return collection
        except Exception as e:
            # Create an empty collection as last resort
            collection = client.create_collection(name="orders")
            print("Created empty collection as fallback")
            return collection

def query_vector_db(query: str, collection: chromadb.Collection, top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Query the vector database for relevant orders.

    Args:
        query: User query text
        collection: ChromaDB collection to query
        top_k: Number of results to return

    Returns:
        List of relevant order data
    """
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

def get_order_by_id(order_id: str, collection: chromadb.Collection) -> Dict[str, Any]:
    """
    Get order details by order ID.

    Args:
        order_id: Order ID to look up
        collection: ChromaDB collection to query

    Returns:
        Order details or None if not found
    """
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

def get_orders_by_customer_id(customer_id: str, collection: chromadb.Collection) -> List[Dict[str, Any]]:
    """
    Get all orders for a customer.

    Args:
        customer_id: Customer ID to look up
        collection: ChromaDB collection to query

    Returns:
        List of order details
    """
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
    
# Add to vector_db.py

# In-memory cache for faster lookups
_MEMORY_CACHE = {}

@lru_cache(maxsize=1000)
def cached_get_order_by_id(order_id, collection):
    """Cached version of get_order_by_id for faster lookups"""
    # Check memory cache first
    cache_key = f"order_{order_id}"
    if cache_key in _MEMORY_CACHE:
        return _MEMORY_CACHE[cache_key]

    # Then check vector DB
    result = get_order_by_id(order_id, collection)

    # Cache the result
    if result:
        _MEMORY_CACHE[cache_key] = result

    return result

@lru_cache(maxsize=1000)
def cached_get_orders_by_customer_id(customer_id, collection):
    """Cached version of get_orders_by_customer_id for faster lookups"""
    cache_key = f"customer_{customer_id}"
    if cache_key in _MEMORY_CACHE:
        return _MEMORY_CACHE[cache_key]

    result = get_orders_by_customer_id(customer_id, collection)

    if result:
        _MEMORY_CACHE[cache_key] = result

    return result