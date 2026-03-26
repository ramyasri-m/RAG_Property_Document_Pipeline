# RAG Property Document Pipeline

A Retrieval-Augmented Generation (RAG) pipeline for querying property documents using natural language. Built with Weaviate, sentence-transformers, and Ollama for fully local inference with no API keys required.

## Architecture
```
PDF Upload → Text Extraction (PyMuPDF) → Chunking (LangChain)
    → Embedding (sentence-transformers) → Weaviate Vector Store
    → Query → Hybrid/Semantic/Keyword Search → Ollama LLM → Answer
```

## Tech Stack

| Component | Tool |
|---|---|
| Document Extraction | PyMuPDF + pdfplumber |
| Text Chunking | LangChain RecursiveCharacterTextSplitter |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| Vector Database | Weaviate |
| LLM | Ollama (llama3.2:1b) |
| UI | Streamlit |

## Features

- PDF ingestion with text and table extraction
- Three search modes: semantic (nearVector), keyword (BM25), and hybrid
- Conversational Q&A with source citations
- Fully local — no API keys or cloud services required

## Setup

### Prerequisites

- Docker
- Python 3.10+
- Ollama with llama3.2:1b installed

### Installation
```bash
git clone https://github.com/ramyasri-m/RAG_Property_Document_Pipeline.git
cd RAG_Property_Document_Pipeline

pip install -r requirements.txt

cp .env.example .env

docker compose up -d

ollama pull llama3.2:1b

streamlit run app.py
```

Open http://localhost:8501 in your browser.

## Usage

1. Upload property PDF documents using the sidebar
2. Click Ingest Documents to process and index them
3. Ask questions in the chat interface
4. Switch between hybrid, semantic, or keyword search modes

## Project Structure
```
RAG_Property_Document_Pipeline/
├── ingestion/
│   ├── pdf_extractor.py      # Text and table extraction using PyMuPDF
│   ├── chunker.py            # Semantic chunking using LangChain
│   └── embedder.py           # Embedding generation using sentence-transformers
├── vectorstore/
│   └── weaviate_client.py    # Weaviate schema, indexing, and connection
├── retrieval/
│   ├── semantic_search.py    # Vector similarity search
│   ├── hybrid_search.py      # Combined BM25 and vector search
│   └── keyword_search.py     # BM25 keyword search
├── chat/
│   └── qa_chain.py           # LLM integration and RAG pipeline
├── app.py                    # Streamlit UI
├── docker-compose.yml        # Weaviate container setup
└── requirements.txt
```

## Background

This pipeline is inspired by a RAG system built during an AI Engineering internship at AriesView, where property documents were processed and queried using similar techniques. This repository recreates that architecture using open-source, locally-run tools.

## License

MIT
