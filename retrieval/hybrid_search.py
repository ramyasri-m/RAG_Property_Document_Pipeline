import weaviate
import os
from typing import List, Dict
from ingestion.embedder import embed_query

COLLECTION_NAME = os.getenv("COLLECTION_NAME", "PropertyDocuments")


def hybrid_search(client: weaviate.Client, query: str, top_k: int = 5, alpha: float = 0.5) -> List[Dict]:
    """
    Hybrid search combining semantic (nearVector) and keyword (BM25).
    alpha=0.0 -> pure keyword, alpha=1.0 -> pure semantic
    """
    query_embedding = embed_query(query)
    result = (
        client.query
        .get(COLLECTION_NAME, ["chunk_id", "filename", "page_number", "text"])
        .with_hybrid(query=query, vector=query_embedding, alpha=alpha)
        .with_limit(top_k)
        .with_additional(["score"])
        .do()
    )
    hits = result.get("data", {}).get("Get", {}).get(COLLECTION_NAME, [])
    return [
        {
            "chunk_id": h["chunk_id"],
            "filename": h["filename"],
            "page_number": h["page_number"],
            "text": h["text"],
            "score": h["_additional"]["score"],
            "search_type": "hybrid"
        }
        for h in hits
    ]


if __name__ == "__main__":
    from vectorstore.weaviate_client import get_client
    client = get_client()
    results = hybrid_search(client, "3 bedroom house with garage")
    for r in results:
        print(f"[{r['score']}] {r['filename']} p{r['page_number']}: {r['text'][:100]}")
