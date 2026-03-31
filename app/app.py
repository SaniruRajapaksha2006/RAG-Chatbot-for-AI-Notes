"""
Streamlit UI for RAG Chatbot
Ask questions about your AI & Data Science notes
"""

import streamlit as st
import pickle
import os
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.rag_chain import RAGChain

# Page configuration
st.set_page_config(
    page_title="AI Notes Chatbot",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    .user-message {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
    }
    .assistant-message {
        background-color: #f5f5f5;
        border-left: 4px solid #4caf50;
    }
    .source-card {
        background-color: #fff3e0;
        padding: 0.5rem;
        border-radius: 5px;
        margin: 0.5rem 0;
        font-size: 0.8rem;
    }
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "rag" not in st.session_state:
    st.session_state.rag = None
if "documents_loaded" not in st.session_state:
    st.session_state.documents_loaded = False

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/color/96/artificial-intelligence.png", width=80)
    st.title("📚 AI Notes Assistant")
    st.markdown("---")

    st.markdown("### 📖 About")
    st.markdown("""
    This chatbot uses **RAG (Retrieval-Augmented Generation)** to answer questions about your AI and Data Science lecture notes.

    **How it works:**
    1. Your PDFs are split into chunks
    2. Each chunk is converted to embeddings
    3. When you ask a question, relevant chunks are retrieved
    4. Llama2 generates an answer with source citations
    """)

    st.markdown("---")

    # Model info
    st.markdown("### 🤖 Model Info")
    st.markdown("""
    - **LLM:** Llama2 (local via Ollama)
    - **Embeddings:** Sentence Transformers (all-MiniLM-L6-v2)
    - **Vector DB:** ChromaDB
    """)

    st.markdown("---")

    # Status indicator
    st.markdown("### 📊 Status")
    if st.session_state.documents_loaded:
        st.success("✅ Documents loaded")
    else:
        st.warning("⚠️ Loading documents...")

    # Clear chat button
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

# Main header
st.markdown("""
<div class="main-header">
    <h1 style="color: white; margin: 0;">📚 AI & Data Science Notes Assistant</h1>
    <p style="color: white; margin: 0.5rem 0 0 0;">Ask questions about your lecture notes and get answers with sources</p>
</div>
""", unsafe_allow_html=True)


# Load RAG system
@st.cache_resource
def load_rag_system():
    """Load the RAG system with cached documents"""
    chunks_path = Path("data/chunks/chunks.pkl")

    if not chunks_path.exists():
        st.error("❌ No documents found!")
        st.info("Please add PDF files to `data/pdfs/` and run `python test_loader.py` first.")
        return None

    try:
        # Load chunks
        with open(chunks_path, "rb") as f:
            chunks = pickle.load(f)

        st.info(f"📄 Loaded {len(chunks)} document chunks")

        # Initialize RAG chain
        rag = RAGChain(model_type="local", embedding_type="local")

        # Add documents to vector store
        with st.spinner("🔄 Building vector database (this may take a minute)..."):
            rag.add_documents(chunks)

        st.success(f"✅ Ready! {rag.vector_store.collection.count()} chunks indexed")
        return rag

    except Exception as e:
        st.error(f"Error loading RAG system: {e}")
        return None


# Load RAG system if not already loaded
if st.session_state.rag is None:
    with st.spinner("🚀 Initializing RAG system..."):
        st.session_state.rag = load_rag_system()
        if st.session_state.rag is not None:
            st.session_state.documents_loaded = True

# Display chat history
for message in st.session_state.messages:
    role = message["role"]
    content = message["content"]

    with st.chat_message(role):
        st.markdown(content)

        # Display sources for assistant messages
        if role == "assistant" and "sources" in message and message["sources"]:
            with st.expander("📚 View Sources"):
                for i, source in enumerate(message["sources"][:3]):
                    st.markdown(f"""
                    <div class="source-card">
                        <strong>📄 Source {i + 1}:</strong> {source['source']} (page {source['page']})<br>
                        <strong>🎯 Similarity:</strong> {source['similarity']:.3f}<br>
                        <strong>📝 Excerpt:</strong> {source['content'][:150]}...
                    </div>
                    """, unsafe_allow_html=True)

# Chat input
if prompt := st.chat_input("Ask a question about your AI/Data Science notes..."):
    if not st.session_state.documents_loaded:
        st.error("Please wait for documents to load before asking questions.")
    else:
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get response
        with st.chat_message("assistant"):
            with st.spinner("🤔 Thinking..."):
                try:
                    result = st.session_state.rag.answer(prompt)

                    # Display answer
                    st.markdown(result["answer"])

                    # Display sources
                    if result["sources"]:
                        with st.expander(f"📚 Sources ({len(result['sources'])} found)"):
                            for i, source in enumerate(result["sources"][:3]):
                                st.markdown(f"""
                                **Source {i + 1}:** `{source['source']}` (page {source['page']})  
                                **Similarity:** {source['similarity']:.3f}  
                                **Excerpt:** {source['content'][:200]}...
                                """)

                    # Display retrieval info
                    st.caption(f"🔍 Retrieved {result['retrieved_count']} relevant documents")

                    # Add to session state
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": result["answer"],
                        "sources": result["sources"]
                    })

                except Exception as e:
                    st.error(f"Error: {e}")
                    st.info("Make sure Ollama is running: `ollama serve` in another terminal")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.8rem;">
    <p>Built with ❤️ using LangChain, Llama2, and Streamlit</p>
    <p>© 2025 R. S. P. S. Uthsara | BSc (Hons) AI & Data Science</p>
</div>
""", unsafe_allow_html=True)