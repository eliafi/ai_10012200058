# Student: Eli Afi Ayekpley | Index: 10012200058
# CS4241 - Introduction to Artificial Intelligence | ACity 2026
"""
Download datasets, run ingestion/chunking, and build the FAISS index.
Run this once before starting the backend.
"""
import sys
import os
from pathlib import Path

# Ensure backend directory is on path
sys.path.insert(0, str(Path(__file__).parent))

from data_ingestion import run_ingestion
from embeddings import build_index

if __name__ == "__main__":
    print("=" * 60)
    print("ACity RAG Setup")
    print("=" * 60)

    print("\nStep 1/2 — Data ingestion")
    chunks = run_ingestion()
    print(f"  Total chunks: {len(chunks)}")

    print("\nStep 2/2 — Building FAISS index")
    pipeline, store = build_index()
    print(f"  Index size: {store.index.ntotal} vectors")

    print("\nSetup complete! You can now start the backend with:")
    print("  python app.py")
