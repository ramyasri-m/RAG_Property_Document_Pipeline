import streamlit as st
import os
import tempfile
from dotenv import load_dotenv

load_dotenv()

from vectorstore.weaviate_client import get_client, create_schema, index_chunks, delete_collection
from ingestion.pdf_extractor import extract_document
from ingestion.chunker import chunk_document
from ingestion.embedder import embed_chunks
from chat.qa_chain import rag_pipeline

st.set_page_config(
    page_title="RAG Property Document Pipeline",
    page_icon="🏠",
    layout="wide"
)

# Initialize Weaviate client
@st.cache_resource
def init_client():
    client = get_client()
    create_schema(client)
    return client

client = init_client()

# Sidebar
with st.sidebar:
    st.title("🏠 Property RAG")
    st.markdown("---")

    # Search type
    search_type = st.selectbox(
        "Search Type",
        ["hybrid", "semantic", "keyword"],
        help="hybrid=BM25+vector, semantic=vector only, keyword=BM25 only"
    )

    top_k = st.slider("Number of results", 1, 10, 5)

    st.markdown("---")

    # Upload section
    st.subheader("📄 Upload Documents")
    uploaded_files = st.file_uploader(
        "Upload property PDFs",
        type=["pdf"],
        accept_multiple_files=True
    )

    if uploaded_files:
        if st.button("🔄 Ingest Documents", type="primary"):
            progress = st.progress(0)
            for i, uploaded_file in enumerate(uploaded_files):
                with st.spinner(f"Processing {uploaded_file.name}..."):
                    # Save to temp file
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                        tmp.write(uploaded_file.read())
                        tmp_path = tmp.name

                    # Pipeline
                    doc_data = extract_document(tmp_path)
                    doc_data["filename"] = uploaded_file.name
                    chunks = chunk_document(doc_data)
                    chunks = embed_chunks(chunks)
                    index_chunks(client, chunks)
                    os.unlink(tmp_path)

                progress.progress((i + 1) / len(uploaded_files))
            st.success(f"✅ Ingested {len(uploaded_files)} document(s)!")

    st.markdown("---")
    if st.button("🗑️ Clear Knowledge Base", type="secondary"):
        delete_collection(client)
        create_schema(client)
        st.success("Knowledge base cleared!")
        st.rerun()

# Main chat interface
st.title("🏠 Property Document Q&A")
st.markdown("Ask questions about your uploaded property documents.")

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander(f"📚 Sources ({len(message['sources'])} chunks)"):
                for src in message["sources"]:
                    st.markdown(f"**{src['filename']}** — Page {src['page_number']}")
                    st.caption(src["text"][:200] + "...")
                    st.markdown("---")

# Chat input
if query := st.chat_input("Ask about your property documents..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Searching documents and generating answer..."):
            chat_history = [
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages[:-1]
            ]
            result = rag_pipeline(
                client, query,
                search_type=search_type,
                top_k=top_k,
                chat_history=chat_history
            )

        st.markdown(result["answer"])
        if result["sources"]:
            with st.expander(f"📚 Sources ({len(result['sources'])} chunks)"):
                for src in result["sources"]:
                    st.markdown(f"**{src['filename']}** — Page {src['page_number']}")
                    st.caption(src["text"][:200] + "...")
                    st.markdown("---")

    st.session_state.messages.append({
        "role": "assistant",
        "content": result["answer"],
        "sources": result["sources"]
    })
