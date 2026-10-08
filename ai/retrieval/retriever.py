"""
Vector Store + Retrieval Layer (PRD Section 5.4 - "Unified Query and Retrieval Layer")

We use ChromaDB - a simple, free, local vector database. It stores:
  - the embedding vector (for similarity search)
  - the original text (so we can show it back to the user)
  - metadata: modality, source_id, timestamp, topic (for filtering + evidence display)

Flow:
  1. index_segments()         -> embed + store all preprocessed sample segments
  2. index_uploaded_chunks()  -> embed + store chunks from user-uploaded files
  3. retrieve()               -> run every time a user asks a question
"""

import sys
import os
import uuid

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

import chromadb
from ai.embeddings.embedder import get_embedding, get_embeddings_batch
from data.schemas.asset_schema import Segment

CHROMA_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "chroma_db")

_client = chromadb.PersistentClient(path=CHROMA_PATH)
_collection = _client.get_or_create_collection(name="multimodal_segments")


def index_segments(segments: list[Segment], clear_existing: bool = True) -> None:
    """
    Embed every segment and store it in ChromaDB.
    Run this once after preprocessing (or whenever new assets are added).

    clear_existing=True (default) wipes the collection first, so re-running
    this script doesn't create duplicate entries. Set to False only if you're
    intentionally adding NEW segments on top of an already-indexed set.
    """
    if len(segments) == 0:
        print("No segments to index.")
        return

    if clear_existing and _collection.count() > 0:
        existing_ids = _collection.get()["ids"]
        _collection.delete(ids=existing_ids)
        print(f"Cleared {len(existing_ids)} old segments before re-indexing.")

    texts = [s.text for s in segments]
    print(f"Generating embeddings for {len(texts)} segments... (this calls the Gemini API)")
    embeddings = get_embeddings_batch(texts, task_type="RETRIEVAL_DOCUMENT")

    _collection.add(
        ids=[s.segment_id for s in segments],
        embeddings=embeddings,
        documents=texts,
        metadatas=[
            {
                "source_id": s.source_id,
                "modality": s.modality,
                "topic": s.topic,
                "timestamp": s.timestamp or "",
                "source_title": s.source_title,
            }
            for s in segments
        ],
    )
    print(f"Indexed {len(segments)} segments into ChromaDB at '{CHROMA_PATH}'.")


def index_uploaded_chunks(chunks: list[dict]) -> int:
    """
    Index user-uploaded chunks directly (no Segment object needed).
    Each chunk: {"text", "source_id", "modality", "source_title"}
    Returns number of chunks indexed.
    """
    if not chunks:
        return 0

    texts = [c["text"] for c in chunks]
    embeddings = get_embeddings_batch(texts, task_type="RETRIEVAL_DOCUMENT")

    _collection.add(
        ids=[f"upload_{uuid.uuid4().hex[:12]}" for _ in chunks],
        embeddings=embeddings,
        documents=texts,
        metadatas=[
            {
                "source_id": c["source_id"],
                "modality": c["modality"],
                "topic": c.get("topic", "uploaded"),
                "timestamp": c.get("timestamp", ""),
                "source_title": c["source_title"],
            }
            for c in chunks
        ],
    )
    return len(chunks)


def retrieve(query: str, top_k: int = 5, modality_filter: str | None = None) -> list[dict]:
    """
    Given a user's question, return the top_k most relevant segments
    across ALL modalities (unless modality_filter is set).

    This is the "unified query" from the PRD - one search call that
    can surface video, slide, quiz, AND discussion evidence together.
    """
    query_embedding = get_embedding(query, task_type="RETRIEVAL_QUERY")

    where_clause = {"modality": modality_filter} if modality_filter else None

    results = _collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        where=where_clause,
    )

    # Reshape Chroma's raw output into a clean, readable list of evidence dicts
    evidence = []
    for i in range(len(results["ids"][0])):
        evidence.append({
            "segment_id": results["ids"][0][i],
            "text": results["documents"][0][i],
            "distance": results["distances"][0][i],  # lower = more similar
            **results["metadatas"][0][i],
        })
    return evidence


def collection_count() -> int:
    return _collection.count()


if __name__ == "__main__":
    from ai.preprocessing.chunker import preprocess_all_assets

    segments = preprocess_all_assets()
    index_segments(segments)

    print("\n--- Test query ---")
    results = retrieve("Why are students confused about learning rate?", top_k=5)
    for r in results:
        print(f"[{r['modality']}] (dist={r['distance']:.3f}) {r['text'][:80]}")