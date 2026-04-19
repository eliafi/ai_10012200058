# Student: Eli Afi Ayekpley | Index: 10012200058
# CS4241 - Introduction to Artificial Intelligence | ACity 2026
import os
import json
import logging
from pathlib import Path
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

app = Flask(__name__)
_allowed = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    r"https://.*\.vercel\.app",
]
if os.getenv("FRONTEND_URL"):
    _allowed.append(os.getenv("FRONTEND_URL"))
CORS(app, origins=_allowed)

_pipeline = None
LOGS_DIR = Path(os.path.join(os.path.dirname(__file__), "..", "logs"))


def get_pipeline():
    global _pipeline
    if _pipeline is None:
        from rag_pipeline import RAGPipeline
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY environment variable not set.")
        logger.info("Initialising RAG pipeline (lazy load)...")
        _pipeline = RAGPipeline(groq_api_key=api_key)
        _pipeline.load_index()
        logger.info("RAG pipeline ready.")
    return _pipeline


@app.get("/health")
def health():
    return jsonify({"status": "ok"}), 200


@app.post("/api/chat")
def chat():
    body = request.get_json(force=True)
    question = body.get("question", "").strip()
    if not question:
        return jsonify({"error": "question is required"}), 400
    retrieval_query = body.get("retrieval_query", question)
    top_k = int(body.get("top_k", 10))
    use_expansion = bool(body.get("use_expansion", False))
    try:
        pipeline = get_pipeline()
        result = pipeline.query(question, top_k=top_k, use_expansion=use_expansion,
                                retrieval_query=retrieval_query)
        return jsonify({
            "answer": result["answer"],
            "sources": result["sources"],
            "prompt": result["prompt"],
        }), 200
    except Exception as e:
        logger.exception("Chat error")
        return jsonify({"error": str(e)}), 500


@app.post("/api/compare")
def compare():
    body = request.get_json(force=True)
    question = body.get("question", "").strip()
    if not question:
        return jsonify({"error": "question is required"}), 400
    try:
        pipeline = get_pipeline()
        result = pipeline.compare_rag_vs_llm(question)
        return jsonify(result), 200
    except Exception as e:
        logger.exception("Compare error")
        return jsonify({"error": str(e)}), 500


@app.post("/api/feedback")
def feedback():
    from rag_pipeline import record_feedback
    body = request.get_json(force=True)
    query = body.get("query", "")
    answer = body.get("answer", "")
    rating = int(body.get("rating", 3))
    comment = body.get("comment", "")
    entry = record_feedback(query, answer, rating, comment)
    return jsonify(entry), 201


@app.get("/api/feedback/stats")
def feedback_stats():
    from rag_pipeline import load_feedback_stats
    return jsonify(load_feedback_stats()), 200


@app.get("/api/logs")
def logs():
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_files = sorted(LOGS_DIR.glob("query_*.json"), reverse=True)[:20]
    entries = []
    for f in log_files:
        try:
            entries.append(json.loads(f.read_text()))
        except Exception:
            pass
    return jsonify(entries), 200


if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
