"""
Test script for RAG pipeline
"""

import os
import pickle
from pathlib import Path

from src.rag_chain import RAGChain


def main():
    """Test the RAG pipeline"""

    print("=" * 60)
    print("RAG CHATBOT - TESTING")
    print("=" * 60)

    # Check if chunks exist
    chunks_path = Path("data/chunks/chunks.pkl")

    if not chunks_path.exists():
        print("\n❌ No chunks found!")
        print("Please process PDFs first.")
        print("Place PDFs in data/pdfs/ and run test_loader.py")
        return

    # Load chunks
    print("\nLoading chunks...")
    with open(chunks_path, 'rb') as f:
        chunks = pickle.load(f)
    print(f"✅ Loaded {len(chunks)} chunks")

    # Initialize RAG chain
    print("\nInitializing RAG chain with local models...")

    # Try OpenAI first
    try:
        rag = RAGChain(model_type="local", embedding_type="local")
        print("✅ Using local models (Ollama + Sentence Transformers)")
    except Exception as e:
        print(f"Ollama failed: {e}")
        print("Falling back to local embeddings...")
        rag = RAGChain(model_type="openai", embedding_type="local")
        print("✅ Using OpenAI with local embeddings")

    # Add documents to vector store
    print("\nAdding documents to vector store...")
    rag.add_documents(chunks)
    print(f"✅ Added {rag.vector_store.collection.count()} documents")

    # Interactive Q&A
    print("\n" + "=" * 60)
    print("READY! Ask questions about your documents.")
    print("Type 'quit' to exit")
    print("=" * 60)

    while True:
        query = input("\nYour question: ").strip()

        if query.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break

        if not query:
            continue

        print("\nThinking...")

        try:
            result = rag.answer(query)

            print(f"\nANSWER:\n{result['answer']}")

            if result['sources']:
                print(f"\nSOURCES:")
                for i, source in enumerate(result['sources'][:3]):
                    print(
                        f"   [{i + 1}] {source['source']} (page {source['page']}) - Similarity: {source['similarity']}")

            print(f"\nRetrieved {result['retrieved_count']} documents")

        except Exception as e:
            print(f"\n❌ Error: {e}")


if __name__ == "__main__":
    main()