# Student: Eli Afi Ayekpley | Index: 10012200058
# CS4241 - Introduction to Artificial Intelligence | ACity 2026
import os
import json
import logging
import numpy as np
import faiss
from pathlib import Path
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from groq import Groq

logger = logging.getLogger(__name__)

DATA_DIR = Path(os.getenv("DATA_DIR", str(Path(__file__).parent.parent / "data")))
INDEX_PATH = DATA_DIR / "faiss.index"
CHUNKS_PATH = DATA_DIR / "chunks.json"
MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384
LOW_CONFIDENCE_THRESHOLD = 0.25


class EmbeddingPipeline:
    def __init__(self):
        self.model = SentenceTransformer(MODEL_NAME)

    def encode(self, texts: list[str]) -> np.ndarray:
        vecs = self.model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
        # L2 normalise so IndexFlatIP computes cosine similarity
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        return (vecs / norms).astype("float32")


class VectorStore:
    def __init__(self):
        self.index = faiss.IndexFlatIP(EMBEDDING_DIM)
        self.chunks: list[dict] = []

    def add(self, embeddings: np.ndarray, chunks: list[dict]) -> None:
        self.index.add(embeddings)
        self.chunks.extend(chunks)

    def search(self, query_vec: np.ndarray, top_k: int = 5) -> list[dict]:
        scores, indices = self.index.search(query_vec, top_k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            results.append({**self.chunks[idx], "score": float(score)})
        return results

    def save(self, path: Path = INDEX_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(path))
        meta_path = path.with_suffix(".chunks.json")
        meta_path.write_text(json.dumps(self.chunks, indent=2))

    def load(self, path: Path = INDEX_PATH) -> None:
        self.index = faiss.read_index(str(path))
        meta_path = path.with_suffix(".chunks.json")
        self.chunks = json.loads(meta_path.read_text())


def expand_query(query: str, groq_client: Groq) -> list[str]:
    """Generate 2 rephrased versions of the query using Groq."""
    try:
        resp = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{
                "role": "user",
                "content": (
                    f"Rephrase the following question in 2 different ways to improve retrieval. "
                    f"Return only the 2 rephrased questions, one per line, no numbering.\n\nQuestion: {query}"
                ),
            }],
            max_tokens=150,
            temperature=0.5,
        )
        lines = resp.choices[0].message.content.strip().splitlines()
        rephrased = [l.strip() for l in lines if l.strip()][:2]
        return [query] + rephrased
    except Exception as e:
        logger.warning(f"Query expansion failed: {e}")
        return [query]


def retrieve_with_expansion(
    query: str,
    pipeline: EmbeddingPipeline,
    store: VectorStore,
    groq_client: Groq,
    top_k: int = 5,
) -> list[dict]:
    """Expand query, retrieve for all variants, deduplicate and re-rank by score."""
    queries = expand_query(query, groq_client)
    seen_ids = {}
    for q in queries:
        vec = pipeline.encode([q])
        results = store.search(vec, top_k=top_k)
        for r in results:
            cid = r["id"]
            if cid not in seen_ids or r["score"] > seen_ids[cid]["score"]:
                seen_ids[cid] = r

    ranked = sorted(seen_ids.values(), key=lambda x: x["score"], reverse=True)[:top_k]

    if ranked and ranked[0]["score"] < LOW_CONFIDENCE_THRESHOLD:
        logger.warning(
            f"Low confidence retrieval: top score={ranked[0]['score']:.3f} < {LOW_CONFIDENCE_THRESHOLD}"
        )

    return ranked


def build_index() -> tuple[EmbeddingPipeline, VectorStore]:
    chunks = json.loads(CHUNKS_PATH.read_text())
    pipeline = EmbeddingPipeline()
    store = VectorStore()

    texts = [c["text"] for c in chunks]
    batch_size = 128
    all_vecs = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding"):
        batch = texts[i: i + batch_size]
        all_vecs.append(pipeline.encode(batch))

    embeddings = np.vstack(all_vecs)
    store.add(embeddings, chunks)
    store.save()
    print(f"Index built: {store.index.ntotal} vectors saved to {INDEX_PATH}")
    return pipeline, store
