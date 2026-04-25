// Student: Eli Afi Ayekpley | Index: 10012200058
// CS4241 - Introduction to Artificial Intelligence | ACity 2026
"use client";

import { useState, useRef, useEffect } from "react";
import { chatQuery, compareQuery, sendFeedback, healthCheck, type Source, type CompareResponse } from "@/lib/api";

const SUGGESTIONS = [
  "Who won the 2020 Ghana election?",
  "2025 Ghana Budget priorities",
  "Which party won the most seats?",
  "Ghana GDP growth target 2025",
  "2012 Ghana election highlights",
  "Ghana revenue targets 2025",
];

interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  sources?: Source[];
  prompt?: string;
  compareData?: CompareResponse;
  feedbackSent?: boolean;
}

function Dots() {
  return (
    <div style={{ display: "flex", gap: 5, padding: "14px 18px" }}>
      <span className="dot" /><span className="dot" /><span className="dot" />
    </div>
  );
}

function ScoreBar({ score }: { score: number }) {
  const pct = Math.round(score * 100);
  const color = pct > 60 ? "#7c3aed" : pct > 35 ? "#f59e0b" : "#f87171";
  return (
    <div>
      <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 4 }}>
        <span style={{ color: "#9ca3af", fontSize: 11 }}>Similarity</span>
        <span style={{ color, fontSize: 11, fontWeight: 600 }}>{pct}%</span>
      </div>
      <div style={{ background: "#1e1b3a", borderRadius: 4, height: 4 }}>
        <div style={{ width: `${pct}%`, height: 4, borderRadius: 4, background: color, transition: "width 0.3s" }} />
      </div>
    </div>
  );
}

function SourcesPanel({ sources }: { sources: Source[] }) {
  const [open, setOpen] = useState(false);
  return (
    <div style={{ marginTop: 10 }}>
      <button onClick={() => setOpen(v => !v)} style={{
        background: "none", border: "none", cursor: "pointer", color: "#a78bfa",
        fontSize: 12, display: "flex", alignItems: "center", gap: 6, padding: 0,
      }}>
        <span style={{ fontSize: 9 }}>{open ? "▼" : "▶"}</span>
        {sources.length} source{sources.length !== 1 ? "s" : ""} retrieved
      </button>
      {open && (
        <div style={{ marginTop: 10, display: "flex", flexDirection: "column", gap: 8 }}>
          {sources.map(s => (
            <div key={s.id} style={{
              background: "#13102b", borderRadius: 10, padding: 12,
              border: "1px solid #2d2460",
            }}>
              <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 6 }}>
                <span style={{ color: "#c4b5fd", fontSize: 11, fontWeight: 600 }}>{s.source}</span>
              </div>
              <ScoreBar score={s.score} />
              <p style={{ color: "#6b7280", fontSize: 11, marginTop: 6, lineHeight: 1.5,
                overflow: "hidden", display: "-webkit-box", WebkitLineClamp: 2, WebkitBoxOrient: "vertical" }}>
                {s.text}
              </p>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

function PromptPanel({ prompt }: { prompt: string }) {
  const [open, setOpen] = useState(false);
  return (
    <div style={{ marginTop: 6 }}>
      <button onClick={() => setOpen(v => !v)} style={{
        background: "none", border: "none", cursor: "pointer", color: "#4b5563",
        fontSize: 11, display: "flex", alignItems: "center", gap: 4, padding: 0,
      }}>
        <span style={{ fontSize: 9 }}>{open ? "▼" : "▶"}</span> View prompt sent to LLM
      </button>
      {open && (
        <pre style={{
          marginTop: 8, background: "#0a0818", border: "1px solid #1e1b3a", borderRadius: 8,
          padding: 12, color: "#4b5563", fontSize: 11, whiteSpace: "pre-wrap", wordBreak: "break-word",
          maxHeight: 180, overflowY: "auto", lineHeight: 1.5,
        }}>{prompt}</pre>
      )}
    </div>
  );
}

function ComparePanel({ data }: { data: CompareResponse }) {
  return (
    <div style={{ marginTop: 12, display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
      <div style={{ background: "#13102b", border: "1px solid #5b21b6", borderRadius: 12, padding: 14 }}>
        <div style={{ color: "#a78bfa", fontSize: 11, fontWeight: 700, textTransform: "uppercase",
          letterSpacing: 1, marginBottom: 8 }}>⚡ RAG Answer</div>
        <p style={{ color: "#e5e7eb", fontSize: 13, lineHeight: 1.6, whiteSpace: "pre-wrap" }}>{data.rag_answer}</p>
        {data.rag_sources.length > 0 && <SourcesPanel sources={data.rag_sources} />}
      </div>
      <div style={{ background: "#13102b", border: "1px solid #1d4ed8", borderRadius: 12, padding: 14 }}>
        <div style={{ color: "#93c5fd", fontSize: 11, fontWeight: 700, textTransform: "uppercase",
          letterSpacing: 1, marginBottom: 8 }}>🧠 Pure LLM</div>
        <p style={{ color: "#e5e7eb", fontSize: 13, lineHeight: 1.6, whiteSpace: "pre-wrap" }}>{data.llm_answer}</p>
        <p style={{ color: "#374151", fontSize: 11, marginTop: 8 }}>Latency: {data.llm_latency_s}s</p>
      </div>
    </div>
  );
}

export default function Page() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [compareMode, setCompareMode] = useState(false);
  const [serverReady, setServerReady] = useState<boolean | null>(null);
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  // Ping the backend on mount so Render wakes up before the user sends their first message,
  // then keep it alive with a ping every 6 minutes (Render free tier sleeps after 15 min).
  useEffect(() => {
    healthCheck().then(ok => setServerReady(ok));
    const keepAlive = setInterval(() => healthCheck(), 6 * 60 * 1000);
    return () => clearInterval(keepAlive);
  }, []);

  function buildContextualQuestion(q: string, history: Message[]): string {
    const recent = history.slice(-4);
    if (!recent.length) return q;
    const ctx = recent.map(m => `${m.role === "user" ? "User" : "Assistant"}: ${m.content.slice(0, 200)}`).join("\n");
    return `Conversation so far:\n${ctx}\n\nNew question: ${q}`;
  }

  async function handleSend(question?: string) {
    const q = (question ?? input).trim();
    if (!q || loading) return;
    setInput("");
    setMessages(prev => [...prev, { id: crypto.randomUUID(), role: "user", content: q }]);
    setLoading(true);
    const contextualQ = buildContextualQuestion(q, messages);
    try {
      if (compareMode) {
        const data = await compareQuery(contextualQ);
        setMessages(prev => [...prev, { id: crypto.randomUUID(), role: "assistant",
          content: data.rag_answer, sources: data.rag_sources, compareData: data }]);
      } else {
        const data = await chatQuery(contextualQ, q);
        setMessages(prev => [...prev, { id: crypto.randomUUID(), role: "assistant",
          content: data.answer, sources: data.sources, prompt: data.prompt }]);
      }
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : "Request failed.";
      setMessages(prev => [...prev, { id: crypto.randomUUID(), role: "assistant", content: `⚠️ ${msg}` }]);
    } finally {
      setLoading(false);
    }
  }

  async function handleFeedback(msg: Message, rating: number) {
    const q = [...messages].reverse().find(m => m.role === "user")?.content ?? "";
    await sendFeedback(q, msg.content, rating);
    setMessages(prev => prev.map(m => m.id === msg.id ? { ...m, feedbackSent: true } : m));
  }

  const isEmpty = messages.length === 0;

  // Gradient background
  const bg = "linear-gradient(160deg, #120a2e 0%, #0e1535 50%, #0a0e24 100%)";

  return (
    <div style={{ minHeight: "100vh", background: bg, display: "flex", flexDirection: "column" }}>

      {/* Header */}
      <div style={{
        padding: "18px 28px", display: "flex", alignItems: "center",
        justifyContent: "space-between", borderBottom: "1px solid #1e1b3a",
      }}>
        <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
          <div style={{
            width: 38, height: 38, borderRadius: 10, display: "flex", alignItems: "center",
            justifyContent: "center", fontSize: 20,
            background: "linear-gradient(135deg,#7c3aed,#3b82f6)",
          }}>🪃</div>
          <div>
            <div style={{ color: "#fff", fontWeight: 700, fontSize: 17, fontFamily: "var(--font-playfair), serif" }}>
              Sankofa AI
            </div>
            <div style={{ color: "#6b7280", fontSize: 11, marginTop: 1 }}>
              Ghana Elections & Budget Intelligence
            </div>
          </div>
        </div>
        <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
        {!isEmpty && (
          <button onClick={() => setMessages([])} style={{
            padding: "8px 18px", borderRadius: 20, fontSize: 12, fontWeight: 600, cursor: "pointer",
            border: "1px solid #2d2460", background: "rgba(255,255,255,0.04)",
            color: "#6b7280", transition: "all 0.2s",
          }}>New Chat</button>
        )}
        <button onClick={() => setCompareMode(v => !v)} style={{
          padding: "8px 18px", borderRadius: 20, fontSize: 12, fontWeight: 600, cursor: "pointer",
          border: compareMode ? "1px solid #7c3aed" : "1px solid #2d2460",
          background: compareMode ? "rgba(124,58,237,0.2)" : "rgba(255,255,255,0.04)",
          color: compareMode ? "#c4b5fd" : "#6b7280",
          transition: "all 0.2s",
        }}>
          {compareMode ? "⚡ Compare ON" : "Compare Mode"}
        </button>
        </div>
      </div>

      {/* Server wake-up banner */}
      {serverReady === null && (
        <div style={{
          background: "rgba(124,58,237,0.12)", borderBottom: "1px solid #3730a3",
          padding: "8px 28px", display: "flex", alignItems: "center", gap: 8,
        }}>
          <span style={{ display: "inline-block", width: 8, height: 8, borderRadius: "50%",
            background: "#f59e0b", animation: "pulse 1.2s ease-in-out infinite" }} />
          <span style={{ color: "#a78bfa", fontSize: 12 }}>
            Waking up backend server — this may take up to 30 s on first load…
          </span>
        </div>
      )}

      {/* Empty state — hero */}
      {isEmpty && (
        <div style={{
          flex: 1, display: "flex", flexDirection: "column",
          alignItems: "center", justifyContent: "center",
          padding: "40px 24px",
        }}>
          <div style={{
            width: 64, height: 64, borderRadius: 20, marginBottom: 24,
            background: "linear-gradient(135deg,#7c3aed,#3b82f6)",
            display: "flex", alignItems: "center", justifyContent: "center", fontSize: 30,
          }}>🪃</div>

          <h1 style={{
            color: "#fff", fontSize: 32, fontWeight: 700, marginBottom: 8, textAlign: "center",
            fontFamily: "var(--font-playfair), serif",
          }}>Ask me anything</h1>

          <p style={{ color: "#6b7280", fontSize: 14, marginBottom: 36, textAlign: "center" }}>
            Ghana Elections · 2025 Budget · Powered by llama-3.3-70b-versatile
          </p>

          {/* Search bar */}
          <form onSubmit={e => { e.preventDefault(); handleSend(); }} style={{
            width: "100%", maxWidth: 560, marginBottom: 24,
            display: "flex", gap: 10, alignItems: "center",
            background: "#16132e", border: "1px solid #3730a3",
            borderRadius: 14, padding: "12px 16px",
          }}>
            <svg width="16" height="16" fill="none" stroke="#6b7280" strokeWidth="2"
              viewBox="0 0 24 24" style={{ flexShrink: 0 }}>
              <circle cx="11" cy="11" r="8"/><path d="m21 21-4.35-4.35"/>
            </svg>
            <input
              autoFocus
              type="text"
              value={input}
              onChange={e => setInput(e.target.value)}
              placeholder="Ask about Ghana elections or the 2025 budget..."
              style={{
                flex: 1, background: "none", border: "none", outline: "none",
                color: "#fff", fontSize: 14, fontFamily: "inherit",
              }}
            />
            <button type="submit" disabled={!input.trim()} style={{
              padding: "8px 18px", borderRadius: 10, border: "none", cursor: "pointer",
              background: input.trim() ? "linear-gradient(135deg,#7c3aed,#3b82f6)" : "#1e1b3a",
              color: input.trim() ? "#fff" : "#4b5563",
              fontSize: 13, fontWeight: 600, fontFamily: "inherit", transition: "all 0.2s",
            }}>Send</button>
          </form>

          {/* Suggestion chips */}
          <div style={{ display: "flex", flexWrap: "wrap", gap: 8, justifyContent: "center", maxWidth: 560 }}>
            {SUGGESTIONS.map(s => (
              <button key={s} onClick={() => handleSend(s)} style={{
                padding: "8px 16px", borderRadius: 20, fontSize: 12, cursor: "pointer",
                background: "rgba(124,58,237,0.08)", border: "1px solid #2d2460",
                color: "#9ca3af", fontFamily: "inherit", transition: "all 0.2s",
              }}
                onMouseEnter={e => { (e.target as HTMLElement).style.borderColor = "#7c3aed"; (e.target as HTMLElement).style.color = "#c4b5fd"; }}
                onMouseLeave={e => { (e.target as HTMLElement).style.borderColor = "#2d2460"; (e.target as HTMLElement).style.color = "#9ca3af"; }}
              >{s}</button>
            ))}
          </div>
        </div>
      )}

      {/* Messages */}
      {!isEmpty && (
        <div style={{ flex: 1, overflowY: "auto", padding: "24px 16px" }}>
          <div style={{ maxWidth: 720, margin: "0 auto", display: "flex", flexDirection: "column", gap: 16 }}>
            {messages.map(msg => (
              <div key={msg.id} style={{ display: "flex", justifyContent: msg.role === "user" ? "flex-end" : "flex-start" }}>
                <div style={{
                  maxWidth: "78%",
                  background: msg.role === "user"
                    ? "linear-gradient(135deg,#7c3aed,#3b82f6)"
                    : "#16132e",
                  border: msg.role === "user" ? "none" : "1px solid #1e1b3a",
                  borderRadius: msg.role === "user" ? "18px 18px 4px 18px" : "4px 18px 18px 18px",
                  padding: "12px 16px",
                  color: "#e5e7eb",
                }}>
                  {msg.compareData
                    ? <ComparePanel data={msg.compareData} />
                    : <p style={{ fontSize: 14, lineHeight: 1.65, whiteSpace: "pre-wrap", margin: 0 }}>{msg.content}</p>
                  }
                  {msg.role === "assistant" && !msg.compareData && msg.sources && msg.sources.length > 0 && (
                    <SourcesPanel sources={msg.sources} />
                  )}
                  {msg.role === "assistant" && !msg.compareData && msg.prompt && (
                    <PromptPanel prompt={msg.prompt} />
                  )}
                  {msg.role === "assistant" && (
                    <div style={{ marginTop: 8, display: "flex", gap: 8, alignItems: "center" }}>
                      {msg.feedbackSent
                        ? <span style={{ color: "#374151", fontSize: 11 }}>Thanks!</span>
                        : <>
                          <button onClick={() => handleFeedback(msg, 5)} style={{ background: "none", border: "none", cursor: "pointer", fontSize: 14, opacity: 0.5, transition: "opacity 0.2s" }}
                            onMouseEnter={e => (e.target as HTMLElement).style.opacity = "1"}
                            onMouseLeave={e => (e.target as HTMLElement).style.opacity = "0.5"}>👍</button>
                          <button onClick={() => handleFeedback(msg, 1)} style={{ background: "none", border: "none", cursor: "pointer", fontSize: 14, opacity: 0.5, transition: "opacity 0.2s" }}
                            onMouseEnter={e => (e.target as HTMLElement).style.opacity = "1"}
                            onMouseLeave={e => (e.target as HTMLElement).style.opacity = "0.5"}>👎</button>
                        </>
                      }
                    </div>
                  )}
                </div>
              </div>
            ))}
            {loading && (
              <div style={{ display: "flex", justifyContent: "flex-start" }}>
                <div style={{ background: "#16132e", border: "1px solid #1e1b3a", borderRadius: "4px 18px 18px 18px" }}>
                  <Dots />
                </div>
              </div>
            )}
            <div ref={bottomRef} />
          </div>
        </div>
      )}

      {/* Input bar (chat mode) */}
      {!isEmpty && (
        <div style={{ padding: "16px", borderTop: "1px solid #1e1b3a" }}>
          <form onSubmit={e => { e.preventDefault(); handleSend(); }} style={{
            maxWidth: 720, margin: "0 auto", display: "flex", gap: 10, alignItems: "center",
            background: "#16132e", border: "1px solid #3730a3",
            borderRadius: 14, padding: "12px 16px",
          }}>
            <input
              type="text"
              value={input}
              onChange={e => setInput(e.target.value)}
              disabled={loading}
              placeholder="Ask a follow-up..."
              style={{
                flex: 1, background: "none", border: "none", outline: "none",
                color: "#fff", fontSize: 14, fontFamily: "inherit", opacity: loading ? 0.5 : 1,
              }}
            />
            <button type="submit" disabled={loading || !input.trim()} style={{
              padding: "8px 18px", borderRadius: 10, border: "none", cursor: "pointer",
              background: (!loading && input.trim()) ? "linear-gradient(135deg,#7c3aed,#3b82f6)" : "#1e1b3a",
              color: (!loading && input.trim()) ? "#fff" : "#4b5563",
              fontSize: 13, fontWeight: 600, fontFamily: "inherit",
            }}>Send</button>
          </form>
          <p style={{ textAlign: "center", color: "#374151", fontSize: 11, marginTop: 8 }}>
            Sankofa AI · Eli Afi Ayekpley · 10012200058
          </p>
        </div>
      )}
    </div>
  );
}
