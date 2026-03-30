# 📚 RAG Chatbot for AI & Data Science Notes

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![LangChain](https://img.shields.io/badge/LangChain-0.3.28-green.svg)](https://www.langchain.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.50.0-red.svg)](https://streamlit.io/)

A Retrieval-Augmented Generation (RAG) chatbot that answers questions about your AI and Data Science lecture notes, textbooks, and course materials. Get accurate answers with source citations from your own documents.

---

## 🎯 What is RAG?

RAG (Retrieval-Augmented Generation) combines:

- **Retrieval**: Finds relevant information from your documents
- **Generation**: Uses an LLM to formulate answers based on retrieved context
- **Citations**: Tells you exactly which document and page the information came from

---

## ✨ Features

- 📄 **Upload PDF documents** — Lecture notes, textbooks, research papers
- 🔍 **Semantic Search** — Finds relevant content based on meaning, not just keywords
- 💬 **Natural Language Q&A** — Ask questions like you'd ask a tutor
- 📖 **Source Citations** — Know exactly which document and page the answer came from
- 🚀 **Local or Cloud LLM** — Use free local models (Llama2) or OpenAI
- 💾 **Persistent Storage** — Documents are indexed once, reused for all questions

---

## 🏗️ Architecture

```
PDF Documents → Text Chunks → Embeddings → Vector Database
                                                  ↓
User Question → Embeddings → Similarity Search → Relevant Chunks
                                                  ↓
                              LLM + Context → Answer with Sources
```

---

## 📊 Project Structure

```
RAG-Chatbot-AI-Notes/
├── src/
│   ├── document_loader.py  # Load and chunk PDF documents
│   ├── embeddings.py       # Generate embeddings (OpenAI or local)
│   ├── vector_store.py     # ChromaDB vector database operations
│   └── rag_chain.py        # Complete RAG pipeline
├── app/
│   └── app.py              # Streamlit web interface (coming soon)
├── data/
│   ├── pdfs/               # Place your PDFs here
│   ├── chunks/             # Processed text chunks
│   └── chroma_db/          # Vector database storage
├── test_loader.py          # Test document loading
├── test_rag.py             # Interactive RAG testing
├── requirements.txt        # Dependencies
└── README.md               # This file
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.9 or higher
- [Ollama](https://ollama.ai/) for local LLM (free) OR OpenAI API key (paid)
- Your PDF lecture notes

### Installation

```bash
# Clone the repository
git clone https://github.com/SaniruRajapaksha2006/RAG-Chatbot-for-AI-Notes.git
cd RAG-Chatbot-AI-Notes

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install Ollama (for free local LLM)
curl -fsSL https://ollama.com/install.sh | sh
ollama pull llama2
```

### Add Your Documents

```bash
# Copy your lecture PDFs to the data/pdfs folder
cp /path/to/your/lecture-notes.pdf data/pdfs/
cp /path/to/your/textbook.pdf data/pdfs/
```

### Process Documents

```bash
# Test document loading and chunking
python test_loader.py

# This will:
# - Read all PDFs from data/pdfs/
# - Split into chunks (1000 chars, 200 overlap)
# - Save chunks to data/chunks/chunks.pkl
```

### Run the RAG Chatbot

```bash
# Interactive Q&A with your documents
python test_rag.py

# Ask questions like:
# - "What is machine learning?"
# - "Explain supervised vs unsupervised learning"
# - "What is backpropagation in neural networks?"
```

---

## 📋 Day-by-Day Progress

### Day 1: Foundation
- Project structure created
- Virtual environment setup
- Dependencies installed (LangChain, ChromaDB, PyPDF)
- Document loader module with PDF processing
- Text chunking with overlapping windows
- Git repository initialized and pushed

### Day 2: RAG Pipeline
- Embeddings module (Sentence Transformers for local, OpenAI as alternative)
- Vector store with ChromaDB
- RAG chain integrating retrieval and generation
- Llama2 integration via Ollama
- Source citations with similarity scores
- Interactive Q&A testing

### Day 3: Deployment *(Coming Soon)*
- Streamlit web interface
- Live deployment to Streamlit Cloud
- GitHub release with pre-trained models
- Project documentation and portfolio polish

---

## 🛠️ Technology Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Framework | LangChain | Orchestrate RAG pipeline |
| LLM | Llama2 (local) / GPT-3.5 (OpenAI) | Answer generation |
| Embeddings | Sentence Transformers (`all-MiniLM-L6-v2`) | Text to vectors |
| Vector DB | ChromaDB | Store and search embeddings |
| Document Processing | PyPDF | Extract text from PDFs |
| Web Interface | Streamlit | User interface *(coming soon)* |

---

## 💡 Example Usage

```python
import pickle
from src.rag_chain import RAGChain

# Load chunks
with open("data/chunks/chunks.pkl", "rb") as f:
    chunks = pickle.load(f)

# Initialize RAG
rag = RAGChain(model_type="local", embedding_type="local")
rag.add_documents(chunks)

# Ask a question
result = rag.answer("What is artificial intelligence?")

print(result["answer"])
# Output: Artificial Intelligence (AI) is the science of making machines
#         that can think and act like humans...

print(result["sources"])
# Output: [{'source': 'Lecture_01.pdf', 'page': 16, 'similarity': 0.427}, ...]
```

---

## ⚙️ Configuration

### Using Local Models (Free, Recommended)

- LLM: Llama2 via Ollama
- Embeddings: Sentence Transformers (`all-MiniLM-L6-v2`)
- No API key required
- Runs entirely on your machine

### Using OpenAI (Paid, Faster)

Create a `.env` file:

```env
OPENAI_API_KEY=your-api-key-here
```

Then modify initialization:

```python
rag = RAGChain(model_type="openai", embedding_type="openai")
```

---

## 🚨 Troubleshooting

**Ollama not responding**
```bash
# Check if Ollama is running
ollama list

# Restart Ollama
ollama serve
```

**Out of memory**
- Reduce chunk size in `document_loader.py`
- Process fewer PDFs at once
- Use a smaller embedding model

**Slow response**
- The first query is slower (model loading)
- Subsequent queries are faster
- Consider using OpenAI for production use

---

## 📝 Future Improvements

- [ ] Streamlit web interface
- [ ] Support for more document types (DOCX, TXT, URLs)
- [ ] Chat history and conversation memory
- [ ] Document management UI
- [ ] Export Q&A sessions
- [ ] Multiple vector collections for different courses

---

## 👨‍💻 Author

**R. S. P. S. Uthsara**
- BSc (Hons) Artificial Intelligence and Data Science
- IIT Sri Lanka / Robert Gordon University

---

## 🙏 Acknowledgments

- **IIT Sri Lanka**: For providing the learning environment and resources
- **LangChain**: For the amazing RAG framework
- **Ollama**: For free local LLM inference

---

## 📄 License

---

⭐ **Star the Project**

If you found this project helpful, please consider giving it a star on GitHub!