import weaviate
import os
from typing import List, Dict

COLLECTION_NAME = os.getenv("COLLECTION_NAME", "PropertyDocuments")


def keyword_search(client: weaviate.Client, query: str, top_k: int = 5) -> List[Dict]:
    """Search using BM25 keyword search."""
    result = (
        client.query
        .get(COLLECTION_NAME, ["chunk_id", "filename", "page_number", "text"])
        .with_bm25(query=query, properties=["text"])
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
            "search_type": "keyword"
        }
        for h in hits
    ]


if __name__ == "__main__":
    from vectorstore.weaviate_client import get_client
    client = get_client()
    results = keyword_search(client, "bedroom bathroom garage")
    for r in results:
        print(f"[{r['score']}] {r['filename']} p{r['page_number']}: {r['text'][:100]}")
