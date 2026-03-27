import ollama
import os
from typing import List, Dict

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:1b")


def format_context(chunks: List[Dict]) -> str:
    """Format retrieved chunks into context string."""
    context_parts = []
    for i, chunk in enumerate(chunks):
        context_parts.append(
            f"[Source {i+1}: {chunk['filename']}, Page {chunk['page_number']}]\n{chunk['text']}"
        )
    return "\n\n".join(context_parts)


def answer_question(query: str, chunks: List[Dict], chat_history: List[Dict] = None) -> str:
    """Generate answer using Ollama LLM with retrieved context."""
    context = format_context(chunks)
    system_prompt = """You are a helpful real estate assistant that answers questions about property documents.
Use only the provided context to answer questions.
If the answer is not in the context, say "I couldn't find that information in the documents."
Always cite the source document and page number when answering.
Be concise and factual."""

    user_message = f"""Context from property documents:
{context}

Question: {query}

Answer based on the context above:"""

    messages = [{"role": "system", "content": system_prompt}]
    if chat_history:
        messages.extend(chat_history[-4:])  # keep last 2 turns
    messages.append({"role": "user", "content": user_message})
    
    ollama_client = ollama.Client(host=OLLAMA_URL)
    response = ollama_client.chat(
    model=OLLAMA_MODEL,
    messages=messages,
    options={"temperature": 0.1}
    )
    
    return response["message"]["content"]


def rag_pipeline(
    client,
    query: str,
    search_type: str = "hybrid",
    top_k: int = 5,
    chat_history: List[Dict] = None
) -> Dict:
    """Full RAG pipeline: retrieve + generate."""
    from retrieval.semantic_search import semantic_search
    from retrieval.hybrid_search import hybrid_search
    from retrieval.keyword_search import keyword_search

    if search_type == "semantic":
        chunks = semantic_search(client, query, top_k)
    elif search_type == "keyword":
        chunks = keyword_search(client, query, top_k)
    else:
        chunks = hybrid_search(client, query, top_k)

    if not chunks:
        return {
            "query": query,
            "answer": "No relevant documents found.",
            "sources": [],
            "search_type": search_type
        }

    answer = answer_question(query, chunks, chat_history)
    return {
        "query": query,
        "answer": answer,
        "sources": chunks,
        "search_type": search_type
    }


if __name__ == "__main__":
    from vectorstore.weaviate_client import get_client
    client = get_client()
    result = rag_pipeline(client, "What is the property address?")
    print(f"Answer: {result['answer']}")
    print(f"Sources: {len(result['sources'])} chunks")
