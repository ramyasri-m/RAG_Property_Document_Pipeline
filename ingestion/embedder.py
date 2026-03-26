from sentence_transformers import SentenceTransformer
from typing import List, Dict
import os

MODEL_NAME = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# Load model once at module level
print(f"Loading embedding model: {MODEL_NAME}")
model = SentenceTransformer(MODEL_NAME)


def embed_chunks(chunks: List[Dict], batch_size: int = 32) -> List[Dict]:
    """Generate embeddings for each chunk."""
    texts = [chunk["text"] for chunk in chunks]
    print(f"Embedding {len(texts)} chunks...")
    embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=True)
    for i, chunk in enumerate(chunks):
        chunk["embedding"] = embeddings[i].tolist()
    print(f"Done embedding {len(chunks)} chunks.")
    return chunks


def embed_query(query: str) -> List[float]:
    """Embed a single query string."""
    return model.encode(query).tolist()


if __name__ == "__main__":
    sample_chunks = [
        {"text": "This property is located at 123 Main St.", "chunk_id": "test_1"},
        {"text": "The listing price is $450,000.", "chunk_id": "test_2"},
    ]
    result = embed_chunks(sample_chunks)
    print(f"Embedding dimension: {len(result[0]['embedding'])}")
