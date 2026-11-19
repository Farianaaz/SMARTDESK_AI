import { useEffect, useRef, useState } from "react";

// Anonymous id kept in the browser so the backend can remember this user's email across sessions
function getUserId() {
  try {
    let id = localStorage.getItem("smartdesk_user_id");
    if (!id) {
      id = crypto.randomUUID();
      localStorage.setItem("smartdesk_user_id", id);
    }
    return id;
  } catch {
    return crypto.randomUUID(); // storage blocked: works, but the email is asked again next time
  }
}

const WELCOME = {
  role: "assistant",
  text: "Hi! I can answer IT/HR questions, raise a support ticket, or check a ticket's status.",
};

export default function App() {
  const [userId] = useState(getUserId);
  const [sessionId, setSessionId] = useState(() => crypto.randomUUID());
  const [messages, setMessages] = useState([WELCOME]);
  const [input, setInput] = useState("");
  const [status, setStatus] = useState("idle"); // idle | loading | error
  const [awaiting, setAwaiting] = useState(false); // backend is waiting for our answer
  const endRef = useRef(null);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, status]);

  async function send(text) {
    const msg = text.trim();
    if (!msg || status === "loading") return;
    setInput("");
    setMessages((m) => [...m, { role: "user", text: msg }]);
    setStatus("loading");
    try {
      const res = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: sessionId, user_id: userId, message: msg }),
      });
      if (!res.ok) throw new Error(`Server error (${res.status})`);
      const data = await res.json();
      setMessages((m) => [...m, { role: "assistant", text: data.reply, cached: Boolean(data.cached) }]);
      setAwaiting(Boolean(data.awaiting_input));
      setStatus("idle");
    } catch {
      setStatus("error");
    }
  }

  async function newChat() {
    await fetch(`/api/reset/${sessionId}`, { method: "POST" }).catch(() => {});
    setSessionId(crypto.randomUUID());
    setMessages([WELCOME]);
    setAwaiting(false);
    setStatus("idle");
  }

  // Show Yes / No buttons when the bot is asking a yes/no question
  const last = messages[messages.length - 1];
  const showYesNo =
    awaiting && last?.role === "assistant" && /\(yes\/no\)/i.test(last.text) && status !== "loading";

  return (
    <div style={s.app}>
      <header style={s.header}>
        <b>SmartDesk AI</b>
        <button onClick={newChat} style={s.ghost}>New chat</button>
      </header>

      <div style={s.messages}>
        {messages.map((m, i) => (
          <div key={i} style={{ ...s.bubble, ...(m.role === "user" ? s.user : s.bot) }}>
            {m.text}
          </div>
        ))}
        {status === "loading" && <div style={{ ...s.bubble, ...s.bot, color: "#6b7280" }}>Thinking...</div>}
        {status === "error" && (
          <div style={{ ...s.bubble, ...s.bot, background: "#fee2e2", color: "#991b1b" }}>
            Could not reach the server. Please try again.
          </div>
        )}
        <div ref={endRef} />
      </div>

      {showYesNo && (
        <div style={s.quick}>
          <button style={s.quickBtn} onClick={() => send("yes")}>Yes</button>
          <button style={s.quickBtn} onClick={() => send("no")}>No</button>
        </div>
      )}

      <div style={s.inputRow}>
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && send(input)}
          placeholder={awaiting ? "Type your answer..." : "Ask an IT or HR question..."}
          disabled={status === "loading"}
          style={s.input}
        />
        <button onClick={() => send(input)} disabled={status === "loading"} style={s.send}>
          Send
        </button>
      </div>
    </div>
  );
}

const s = {
  app: { maxWidth: 720, margin: "0 auto", height: "100vh", display: "flex", flexDirection: "column", fontFamily: "Arial, sans-serif", background: "#fff" },
  header: { display: "flex", justifyContent: "space-between", alignItems: "center", padding: "12px 16px", background: "#1f3864", color: "#fff" },
  ghost: { background: "#fff", color: "#1f3864", border: 0, padding: "6px 12px", borderRadius: 6, cursor: "pointer" },
  messages: { flex: 1, overflowY: "auto", padding: 16, display: "flex", flexDirection: "column", gap: 8, background: "#f3f4f6" },
  bubble: { maxWidth: "80%", padding: "10px 14px", borderRadius: 12, whiteSpace: "pre-wrap", lineHeight: 1.4 },
  user: { alignSelf: "flex-end", background: "#1f3864", color: "#fff" },
  bot: { alignSelf: "flex-start", background: "#fff", border: "1px solid #e5e7eb" },
  cachedTag: { marginTop: 6, fontSize: 11, color: "#6b7280" },
  quick: { display: "flex", gap: 8, padding: "8px 12px 0", background: "#fff" },
  quickBtn: { padding: "6px 16px", borderRadius: 16, border: "1px solid #1f3864", background: "#fff", color: "#1f3864", cursor: "pointer" },
  inputRow: { display: "flex", gap: 8, padding: 12, borderTop: "1px solid #e5e7eb" },
  input: { flex: 1, padding: 10, border: "1px solid #d1d5db", borderRadius: 8, fontSize: 14 },
  send: { padding: "10px 16px", background: "#1f3864", color: "#fff", border: 0, borderRadius: 8, cursor: "pointer" },
};
