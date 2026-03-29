"""
Document Loader Module
Handles loading and chunking PDF documents
"""

import os
from pathlib import Path
from typing import List, Dict
import logging

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DocumentLoader:
    """Load and chunk PDF documents for RAG"""

    def __init__(self, chunk_size=1000, chunk_overlap=200):
        """
        Initialize document loader

        Args:
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""],
            length_function=len
        )

    def load_pdf(self, pdf_path: str) -> List[Document]:
        """
        Load a single PDF file

        Args:
            pdf_path: Path to PDF file

        Returns:
            List of Document objects
        """
        logger.info(f"Loading PDF: {pdf_path}")
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        logger.info(f"Loaded {len(documents)} pages")
        return documents

    def load_all_pdfs(self, pdf_folder: str) -> List[Document]:
        """
        Load all PDFs from a folder

        Args:
            pdf_folder: Path to folder containing PDFs

        Returns:
            List of all documents
        """
        all_documents = []
        pdf_files = list(Path(pdf_folder).glob("*.pdf"))

        logger.info(f"Found {len(pdf_files)} PDF files")

        for pdf_file in pdf_files:
            try:
                docs = self.load_pdf(str(pdf_file))
                all_documents.extend(docs)
                logger.info(f"Successfully loaded: {pdf_file.name}")
            except Exception as e:
                logger.error(f"Error loading {pdf_file.name}: {e}")

        logger.info(f"Total documents loaded: {len(all_documents)}")
        return all_documents

    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into smaller chunks

        Args:
            documents: List of Document objects

        Returns:
            List of chunked documents
        """
        logger.info(f"Chunking {len(documents)} documents...")
        chunks = self.text_splitter.split_documents(documents)
        logger.info(f"Created {len(chunks)} chunks")

        # Add metadata to each chunk
        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_id"] = i

        return chunks

    def process_folder(self, pdf_folder: str, save_path: str = None) -> List[Document]:
        """
        Complete pipeline: load PDFs and create chunks

        Args:
            pdf_folder: Path to folder with PDFs
            save_path: Optional path to save chunks

        Returns:
            List of chunked documents
        """
        # Load all PDFs
        documents = self.load_all_pdfs(pdf_folder)

        if not documents:
            logger.warning("No documents loaded")
            return []

        # Create chunks
        chunks = self.chunk_documents(documents)

        # Save chunks if path provided
        if save_path:
            import pickle
            with open(save_path, 'wb') as f:
                pickle.dump(chunks, f)
            logger.info(f"Saved chunks to {save_path}")

        return chunks


if __name__ == "__main__":
    # Test the loader
    loader = DocumentLoader()

    # Example: Process PDFs from data/pdfs folder
    pdf_folder = "data/pdfs"

    if os.path.exists(pdf_folder):
        chunks = loader.process_folder(
            pdf_folder,
            save_path="data/chunks/chunks.pkl"
        )
        print(f"\n✅ Processed {len(chunks)} chunks")
        print(f"Sample chunk: {chunks[0].page_content[:200]}...")
    else:
        print(f"📁 Please add PDF files to: {pdf_folder}")