# 📚 RAG Chatbot for AI & Data Science Notes

A Retrieval-Augmented Generation (RAG) chatbot that answers questions about your course materials using AI.

## 🚀 Features
- Upload PDF lecture notes
- Ask questions in natural language
- Get answers with source citations
- Powered by LangChain and OpenAI

## 📋 Day 1 Progress
- [x] Project structure created
- [x] Dependencies installed
- [x] Document loader module ready
- [x] PDF processing tested

## 🔧 Setup
```bash
# Clone
git clone [your-repo-url]
cd RAG-Chatbot-AI-Notes

# Setup environment
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Add your PDFs to data/pdfs/
# Set OpenAI API key in .env

# Test document loading
python test_loader.py