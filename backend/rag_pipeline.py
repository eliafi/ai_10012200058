# Student: Eli Afi Ayekpley | Index: 10012200058
# CS4241 - Introduction to Artificial Intelligence | ACity 2026
import os
import json
import time
import logging
import uuid
from datetime import datetime
from pathlib import Path
from groq import Groq
from embeddings import EmbeddingPipeline, VectorStore, retrieve_with_expansion

logger = logging.getLogger(__name__)

GROQ_MODEL = "llama-3.3-70b-versatile"
LOGS_DIR = Path(os.getenv("LOGS_DIR", str(Path(__file__).parent.parent / "logs")))
FEEDBACK_PATH = LOGS_DIR / "feedback.json"
TOKEN_BUDGET = 3000

SYSTEM_PROMPT_V3 = """You are a precise research assistant for Ghana election and budget data. Answer ONLY using the provided context.

Rules:
1. If the context contains the answer, provide it clearly and cite the source document.
2. For election questions: the context contains per-region vote rows. Identify the candidate(s) with the most votes across the regions shown and state who appears to have won. If only partial data is shown, note that.
3. If the context is truly irrelevant or empty, say: "I don't have enough context to answer this question."
4. Never fabricate facts, statistics, names, or dates not present in the context.
5. Keep answers concise and factual.
6. When citing, use the format: [Source: <filename>]
"""


def build_prompt_v3(question: str, retrieved: list[dict]) -> str:
    """Build prompt with context window management: rank by score, truncate at TOKEN_BUDGET chars."""
    sorted_docs = sorted(retrieved, key=lambda x: x["score"], reverse=True)

    context_parts = []
    total_chars = 0
    # Approximate: 1 token ≈ 4 chars
    char_budget = TOKEN_BUDGET * 4

    for doc in sorted_docs:
        entry = f"[Source: {doc['source']}]\n{doc['text']}\n"
        if total_chars + len(entry) > char_budget:
            break
        context_parts.append(entry)
        total_chars += len(entry)

    context = "\n---\n".join(context_parts) if context_parts else "No context available."
    return f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"


class RAGPipeline:
    def __init__(self, groq_api_key: str):
        self.client = Groq(api_key=groq_api_key)
        self.embedding_pipeline = EmbeddingPipeline()
        self.vector_store = VectorStore()
        LOGS_DIR.mkdir(parents=True, exist_ok=True)

    def load_index(self, index_path=None) -> None:
        self.vector_store.load()

    def query(self, question: str, top_k: int = 10, use_expansion: bool = False,
              retrieval_query: str = None) -> dict:
        # retrieval_query: clean question for vector search (strips conversation history)
        # question: full contextual question for the LLM prompt
        rq = retrieval_query or question
        t0 = time.time()

        if use_expansion:
            retrieved = retrieve_with_expansion(
                rq, self.embedding_pipeline, self.vector_store, self.client, top_k
            )
        else:
            vec = self.embedding_pipeline.encode([rq])
            retrieved = self.vector_store.search(vec, top_k)

        retrieval_time = time.time() - t0
        logger.info(f"Retrieval time: {retrieval_time:.3f}s, docs: {len(retrieved)}")

        prompt = build_prompt_v3(question, retrieved)
        logger.info(f"Prompt chars: {len(prompt)}")

        t1 = time.time()
        resp = self.client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT_V3},
                {"role": "user", "content": prompt},
            ],
            max_tokens=512,
            temperature=0.2,
        )
        llm_latency = time.time() - t1

        answer = resp.choices[0].message.content.strip()
        usage = resp.usage

        logger.info(
            f"LLM latency: {llm_latency:.3f}s | "
            f"prompt_tokens={usage.prompt_tokens} completion_tokens={usage.completion_tokens}"
        )

        log_entry = {
            "id": str(uuid.uuid4()),
            "timestamp": datetime.utcnow().isoformat(),
            "question": question,
            "answer": answer,
            "retrieval_time_s": round(retrieval_time, 3),
            "llm_latency_s": round(llm_latency, 3),
            "prompt_chars": len(prompt),
            "prompt_tokens": usage.prompt_tokens,
            "completion_tokens": usage.completion_tokens,
            "top_k": top_k,
            "use_expansion": use_expansion,
            "sources": [{"id": d["id"], "source": d["source"], "score": d["score"]} for d in retrieved],
        }
        log_path = LOGS_DIR / f"query_{log_entry['id'][:8]}.json"
        log_path.write_text(json.dumps(log_entry, indent=2))

        return {
            "answer": answer,
            "sources": retrieved,
            "prompt": prompt,
            "log": log_entry,
        }

    def compare_rag_vs_llm(self, question: str) -> dict:
        """Run the same question through RAG and through pure LLM with no context."""
        rag_result = self.query(question, top_k=5, use_expansion=True)

        t0 = time.time()
        raw_resp = self.client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": question},
            ],
            max_tokens=512,
            temperature=0.2,
        )
        llm_latency = time.time() - t0
        pure_answer = raw_resp.choices[0].message.content.strip()

        return {
            "question": question,
            "rag_answer": rag_result["answer"],
            "rag_sources": rag_result["sources"],
            "llm_answer": pure_answer,
            "llm_latency_s": round(llm_latency, 3),
        }


def record_feedback(query: str, answer: str, rating: int, comment: str = "") -> dict:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    entry = {
        "id": str(uuid.uuid4()),
        "timestamp": datetime.utcnow().isoformat(),
        "query": query,
        "answer": answer,
        "rating": rating,
        "comment": comment,
    }
    feedback = []
    if FEEDBACK_PATH.exists():
        try:
            feedback = json.loads(FEEDBACK_PATH.read_text())
        except Exception:
            feedback = []
    feedback.append(entry)
    FEEDBACK_PATH.write_text(json.dumps(feedback, indent=2))
    return entry


def load_feedback_stats() -> dict:
    if not FEEDBACK_PATH.exists():
        return {"total": 0, "positive": 0, "negative": 0, "average_rating": None}
    try:
        feedback = json.loads(FEEDBACK_PATH.read_text())
    except Exception:
        return {"total": 0, "positive": 0, "negative": 0, "average_rating": None}

    total = len(feedback)
    positive = sum(1 for f in feedback if f.get("rating", 0) >= 4)
    negative = sum(1 for f in feedback if f.get("rating", 0) <= 2)
    avg = sum(f.get("rating", 0) for f in feedback) / total if total else None
    return {
        "total": total,
        "positive": positive,
        "negative": negative,
        "average_rating": round(avg, 2) if avg is not None else None,
    }
