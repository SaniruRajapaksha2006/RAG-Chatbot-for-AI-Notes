"""Test document loader"""

from src.document_loader import DocumentLoader

# Initialize loader
loader = DocumentLoader(chunk_size=500, chunk_overlap=50)

# Process PDFs
chunks = loader.process_folder(
    "data/pdfs",
    save_path="data/chunks/chunks.pkl"
)

# Verify
print(f"\n✅ Successfully processed {len(chunks)} chunks")
if chunks:
    print(f"\nSample chunk preview:")
    print(f"Content: {chunks[0].page_content[:300]}...")
    print(f"Metadata: {chunks[0].metadata}")
