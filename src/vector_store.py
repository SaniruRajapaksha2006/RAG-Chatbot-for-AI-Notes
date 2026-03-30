"""
Vector Store Module
Manages ChromaDB for storing and retrieving embeddings
"""

import os
import pickle
from pathlib import Path
from typing import List, Tuple
import logging

import chromadb
from chromadb.config import Settings
from langchain.schema import Document

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VectorStore:
    """Manage vector database for RAG"""

    def __init__(self, persist_directory: str = "data/chroma_db"):
        """
        Initialize ChromaDB

        Args:
            persist_directory: Directory to store the database
        """
        self.persist_directory = persist_directory

        # Create directory if it doesn't exist
        Path(persist_directory).mkdir(parents=True, exist_ok=True)

        # Initialize Chroma client
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False)
        )

        # Collection name
        self.collection_name = "rag_documents"

        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}  # Use cosine similarity
        )

        logger.info(f"Vector store initialized at {persist_directory}")
        logger.info(f"Collection has {self.collection.count()} documents")

    def add_documents(self, documents: List[Document], embeddings: List[List[float]]):
        """
        Add documents and their embeddings to the vector store

        Args:
            documents: List of Document objects
            embeddings: List of embedding vectors
        """
        if len(documents) != len(embeddings):
            raise ValueError("Number of documents and embeddings must match")

        # Prepare data for ChromaDB
        ids = []
        texts = []
        metadatas = []

        for i, doc in enumerate(documents):
            doc_id = f"doc_{i}_{doc.metadata.get('source', 'unknown')}"
            ids.append(doc_id)
            texts.append(doc.page_content)
            metadatas.append({
                "source": doc.metadata.get("source", "unknown"),
                "page": doc.metadata.get("page", 0),
                "chunk_id": doc.metadata.get("chunk_id", i)
            })

        # Add to ChromaDB
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas,
            documents=texts
        )

        logger.info(f"Added {len(documents)} documents to vector store")

    def search(self, query_embedding: List[float], top_k: int = 5) -> List[Tuple[Document, float]]:
        """
        Search for similar documents

        Args:
            query_embedding: Embedding of the query
            top_k: Number of results to return

        Returns:
            List of (Document, similarity_score) tuples
        """
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )

        documents = []
        if results['documents'] and results['documents'][0]:
            for i, doc_text in enumerate(results['documents'][0]):
                metadata = results['metadatas'][0][i] if results['metadatas'] else {}
                similarity = 1 - results['distances'][0][i]  # Convert distance to similarity

                doc = Document(
                    page_content=doc_text,
                    metadata=metadata
                )
                documents.append((doc, similarity))

        logger.info(f"Found {len(documents)} similar documents")
        return documents

    def get_all_documents(self) -> List[Document]:
        """
        Retrieve all documents from the vector store

        Returns:
            List of all Document objects
        """
        all_data = self.collection.get()

        documents = []
        for i, doc_text in enumerate(all_data['documents']):
            metadata = all_data['metadatas'][i] if all_data['metadatas'] else {}
            doc = Document(page_content=doc_text, metadata=metadata)
            documents.append(doc)

        return documents

    def delete_collection(self):
        """Delete the entire collection"""
        self.client.delete_collection(self.collection_name)
        logger.info(f"Deleted collection: {self.collection_name}")


if __name__ == "__main__":
    # Test vector store
    store = VectorStore()

    # Create a test document
    from langchain.schema import Document

    test_doc = Document(
        page_content="This is a test document about machine learning.",
        metadata={"source": "test.pdf", "page": 1}
    )

    # Create a dummy embedding (768 dimensions for local model)
    import numpy as np

    test_embedding = np.random.randn(768).tolist()

    # Add test document
    store.add_documents([test_doc], [test_embedding])
    print(f"✅ Added test document. Total: {store.collection.count()}")

    # Search with dummy query
    results = store.search(test_embedding, top_k=1)
    print(f"✅ Search returned {len(results)} results")