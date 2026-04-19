// Student: Eli Afi Ayekpley | Index: 10012200058
// CS4241 - Introduction to Artificial Intelligence | ACity 2026
const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:5000";

export interface Source {
  id: string;
  source: string;
  text: string;
  score: number;
}

export interface ChatResponse {
  answer: string;
  sources: Source[];
  prompt?: string;
}

export interface CompareResponse {
  question: string;
  rag_answer: string;
  rag_sources: Source[];
  llm_answer: string;
  llm_latency_s: number;
}

export interface FeedbackStats {
  total: number;
  positive: number;
  negative: number;
  average_rating: number | null;
}

export async function chatQuery(
  question: string,
  retrieval_query: string,
  top_k = 10,
  use_expansion = false
): Promise<ChatResponse> {
  const res = await fetch(`${API_URL}/api/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question, retrieval_query, top_k, use_expansion }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ error: res.statusText }));
    throw new Error(err.error || "Chat request failed");
  }
  return res.json();
}

export async function compareQuery(question: string): Promise<CompareResponse> {
  const res = await fetch(`${API_URL}/api/compare`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ error: res.statusText }));
    throw new Error(err.error || "Compare request failed");
  }
  return res.json();
}

export async function sendFeedback(
  query: string,
  answer: string,
  rating: number,
  comment = ""
): Promise<void> {
  await fetch(`${API_URL}/api/feedback`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query, answer, rating, comment }),
  });
}

export async function getFeedbackStats(): Promise<FeedbackStats> {
  const res = await fetch(`${API_URL}/api/feedback/stats`);
  return res.json();
}
