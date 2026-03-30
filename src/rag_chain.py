"""
RAG Chain Module
Orchestrates retrieval and generation for question answering
"""

import os
import logging
from typing import List, Tuple
from dotenv import load_dotenv

from langchain.schema import Document
from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatOllama

from src.embeddings import EmbeddingGenerator
from src.vector_store import VectorStore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()


class RAGChain:
    """Retrieval-Augmented Generation pipeline"""

    def __init__(self, model_type="local", embedding_type="local"):
        """
        Initialize RAG chain

        Args:
            model_type: "openai" or "local"
            embedding_type: "openai" or "local"
        """
        self.model_type = model_type
        self.embedding_type = embedding_type

        # Initialize embedding generator
        self.embedder = EmbeddingGenerator(model_type=embedding_type)

        # Initialize vector store
        self.vector_store = VectorStore()

        # Initialize LLM
        if model_type == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY not found in .env file")

            self.llm = ChatOpenAI(
                model="gpt-3.5-turbo",
                temperature=0.7,
                openai_api_key=api_key
            )
            logger.info("Initialized OpenAI LLM")

        elif model_type == "local":
            # For free local LLM (requires Ollama installed)
            self.llm = ChatOllama(
                model="llama2",
                temperature=0.7
            )
            logger.info("Initialized local Ollama LLM with llama2")

        else:
            raise ValueError(f"Unknown model_type: {model_type}")

    def add_documents(self, documents: List[Document]):
        """
        Add documents to the vector store

        Args:
            documents: List of Document objects (chunks)
        """
        logger.info(f"Adding {len(documents)} documents to vector store...")

        # Generate embeddings for all documents
        texts = [doc.page_content for doc in documents]
        embeddings = self.embedder.embed_documents(texts)

        # Add to vector store
        self.vector_store.add_documents(documents, embeddings)

        logger.info("Documents added successfully")

    def retrieve(self, query: str, top_k: int = 5) -> List[Tuple[Document, float]]:
        """
        Retrieve relevant documents for a query

        Args:
            query: User question
            top_k: Number of documents to retrieve

        Returns:
            List of (Document, similarity_score) tuples
        """
        # Generate query embedding
        query_embedding = self.embedder.embed_text(query)

        # Search vector store
        results = self.vector_store.search(query_embedding, top_k=top_k)

        return results

    def generate_prompt(self, query: str, documents: List[Tuple[Document, float]]) -> str:
        """
        Create prompt with context and question

        Args:
            query: User question
            documents: Retrieved documents with scores

        Returns:
            Formatted prompt string
        """
        # Build context from retrieved documents
        context_parts = []
        sources = []

        for i, (doc, score) in enumerate(documents):
            source = doc.metadata.get('source', 'unknown')
            page = doc.metadata.get('page', 'unknown')
            context_parts.append(f"[{i + 1}] {doc.page_content}")
            sources.append(f"[{i + 1}] Source: {source}, Page: {page}")

        context = "\n\n".join(context_parts)
        sources_text = "\n".join(sources)

        prompt = f"""You are a helpful AI assistant answering questions based on the provided documents.

CONTEXT:
{context}

SOURCES:
{sources_text}

QUESTION: {query}

INSTRUCTIONS:
1. Answer based ONLY on the context provided above
2. If the answer is not in the context, say "I don't have enough information to answer this question"
3. Cite your sources using the numbers [1], [2], etc.
4. Be concise but informative

ANSWER:"""

        return prompt

    def answer(self, query: str, top_k: int = 5) -> dict:
        """
        Answer a question using RAG

        Args:
            query: User question
            top_k: Number of documents to retrieve

        Returns:
            Dictionary with answer and sources
        """
        logger.info(f"Processing query: {query}")

        # Retrieve relevant documents
        retrieved_docs = self.retrieve(query, top_k=top_k)

        if not retrieved_docs:
            return {
                "answer": "No relevant documents found. Please add some PDFs first.",
                "sources": []
            }

        # Generate prompt
        prompt = self.generate_prompt(query, retrieved_docs)

        # Get response from LLM
        response = self.llm.invoke(prompt)
        answer = response.content

        # Extract sources
        sources = []
        for doc, score in retrieved_docs:
            sources.append({
                "source": doc.metadata.get('source', 'unknown'),
                "page": doc.metadata.get('page', 'unknown'),
                "content": doc.page_content[:200] + "...",
                "similarity": round(score, 3)
            })

        return {
            "answer": answer,
            "sources": sources,
            "retrieved_count": len(retrieved_docs)
        }


if __name__ == "__main__":
    # Test RAG chain
    print("Testing RAG Chain...")

    # Initialize
    rag = RAGChain(model_type="openai", embedding_type="openai")

    # Load chunks
    chunks_path = "data/chunks/chunks.pkl"

    if os.path.exists(chunks_path):
        import pickle

        with open(chunks_path, 'rb') as f:
            chunks = pickle.load(f)

        print(f"Loaded {len(chunks)} chunks from Day 1")

        # Add to vector store
        rag.add_documents(chunks)

        # Test question
        query = "What is machine learning?"
        result = rag.answer(query)

        print(f"\nQuestion: {query}")
        print(f"Answer: {result['answer']}")
        print(f"\nSources: {len(result['sources'])}")

    else:
        print("No chunks found. Please process PDFs First.")