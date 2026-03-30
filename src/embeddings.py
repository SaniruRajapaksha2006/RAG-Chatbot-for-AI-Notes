"""
Embeddings Module
Converts text chunks into vector embeddings for semantic search
"""

import os
from typing import List
import logging
from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


class EmbeddingGenerator:
    """Generate embeddings for text chunks"""

    def __init__(self, model_type="openai"):
        """
        Initialize embedding model

        Args:
            model_type: "openai" or "local"
        """
        self.model_type = model_type

        if model_type == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY not found in .env file")

            self.embeddings = OpenAIEmbeddings(
                model="text-embedding-ada-002",
                openai_api_key=api_key
            )
            logger.info("Initialized OpenAI embeddings")

        elif model_type == "local":
            # local embeddings
            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
            logger.info("Initialized local HuggingFace embeddings")

        else:
            raise ValueError(f"Unknown model_type: {model_type}")

    def embed_text(self, text: str) -> List[float]:
        """
        Generate embedding for a single text

        Args:
            text: Input text

        Returns:
            List of floats (embedding vector)
        """
        return self.embeddings.embed_query(text)

    def embed_documents(self, documents: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple documents

        Args:
            documents: List of text documents

        Returns:
            List of embedding vectors
        """
        return self.embeddings.embed_documents(documents)

    def get_embedding_dimension(self) -> int:
        """Get the dimension of the embedding vectors"""
        # Test with a sample text
        sample_embedding = self.embed_text("test")
        return len(sample_embedding)


if __name__ == "__main__":
    # Test the embedding generator
    try:
        # Try OpenAI first
        generator = EmbeddingGenerator(model_type="openai")
        test_embedding = generator.embed_text("What is machine learning?")
        print(f"✅ OpenAI embedding generated (dimension: {len(test_embedding)})")

    except Exception as e:
        print(f"OpenAI failed: {e}")
        print("Falling back to local embeddings...")

        # Fallback to local
        generator = EmbeddingGenerator(model_type="local")
        test_embedding = generator.embed_text("What is machine learning?")
        print(f"✅ Local embedding generated (dimension: {len(test_embedding)})")