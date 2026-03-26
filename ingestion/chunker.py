from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List, Dict


def chunk_document(doc_data: dict, chunk_size: int = 512, chunk_overlap: int = 50) -> List[Dict]:
    """Split document pages into overlapping chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    chunks = []
    for page in doc_data["pages"]:
        if not page["text"].strip():
            continue
        page_chunks = splitter.split_text(page["text"])
        for i, chunk in enumerate(page_chunks):
            chunks.append({
                "chunk_id": f"{doc_data['filename']}_p{page['page_number']}_c{i}",
                "filename": doc_data["filename"],
                "page_number": page["page_number"],
                "chunk_index": i,
                "text": chunk,
                "char_count": len(chunk)
            })
    print(f"Created {len(chunks)} chunks from {doc_data['filename']}")
    return chunks


if __name__ == "__main__":
    sample = {
        "filename": "test.pdf",
        "pages": [{"page_number": 1, "text": "This is a sample property document. " * 50}]
    }
    chunks = chunk_document(sample)
    print(f"Sample chunk: {chunks[0]['text'][:100]}")
