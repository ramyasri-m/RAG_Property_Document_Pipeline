import weaviate
import os
from typing import List, Dict

WEAVIATE_URL = os.getenv("WEAVIATE_URL", "http://localhost:8080")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "PropertyDocuments")


def get_client() -> weaviate.Client:
    """Connect to Weaviate."""
    return weaviate.Client(url=WEAVIATE_URL)


def create_schema(client: weaviate.Client):
    """Create Weaviate schema for property documents."""
    schema = {
        "class": COLLECTION_NAME,
        "description": "Property document chunks with embeddings",
        "vectorizer": "none",
        "properties": [
            {"name": "chunk_id", "dataType": ["text"]},
            {"name": "filename", "dataType": ["text"]},
            {"name": "page_number", "dataType": ["int"]},
            {"name": "chunk_index", "dataType": ["int"]},
            {"name": "text", "dataType": ["text"]},
            {"name": "char_count", "dataType": ["int"]},
        ]
    }
    existing = [c["class"] for c in client.schema.get()["classes"]]
    if COLLECTION_NAME not in existing:
        client.schema.create_class(schema)
        print(f"Created collection: {COLLECTION_NAME}")
    else:
        print(f"Collection already exists: {COLLECTION_NAME}")


def index_chunks(client: weaviate.Client, chunks: List[Dict]):
    """Batch index chunks into Weaviate."""
    print(f"Indexing {len(chunks)} chunks into Weaviate...")
    with client.batch as batch:
        batch.batch_size = 50
        for chunk in chunks:
            batch.add_data_object(
                data_object={
                    "chunk_id": chunk["chunk_id"],
                    "filename": chunk["filename"],
                    "page_number": chunk["page_number"],
                    "chunk_index": chunk["chunk_index"],
                    "text": chunk["text"],
                    "char_count": chunk["char_count"],
                },
                class_name=COLLECTION_NAME,
                vector=chunk["embedding"]
            )
    print(f"Indexed {len(chunks)} chunks successfully.")


def delete_collection(client: weaviate.Client):
    """Delete the collection (for reset)."""
    client.schema.delete_class(COLLECTION_NAME)
    print(f"Deleted collection: {COLLECTION_NAME}")


if __name__ == "__main__":
    client = get_client()
    print(f"Weaviate ready: {client.is_ready()}")
    create_schema(client)
