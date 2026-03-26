import weaviate
import os
from typing import List, Dict
from ingestion.embedder import embed_query

COLLECTION_NAME = os.getenv("COLLECTION_NAME", "PropertyDocuments")


def semantic_search(client: weaviate.Client, query: str, top_k: int = 5) -> List[Dict]:
    """Search using vector similarity (nearVector)."""
    query_embedding = embed_query(query)
    result = (
        client.query
        .get(COLLECTION_NAME, ["chunk_id", "filename", "page_number", "text"])
        .with_near_vector({"vector": query_embedding})
        .with_limit(top_k)
        .with_additional(["certainty", "distance"])
        .do()
    )
    hits = result.get("data", {}).get("Get", {}).get(COLLECTION_NAME, [])
    return [
        {
            "chunk_id": h["chunk_id"],
            "filename": h["filename"],
            "page_number": h["page_number"],
            "text": h["text"],
            "certainty": h["_additional"]["certainty"],
            "search_type": "semantic"
        }
        for h in hits
    ]


if __name__ == "__main__":
    from vectorstore.weaviate_client import get_client
    client = get_client()
    results = semantic_search(client, "property price listing")
    for r in results:
        print(f"[{r['certainty']:.2f}] {r['filename']} p{r['page_number']}: {r['text'][:100]}")
