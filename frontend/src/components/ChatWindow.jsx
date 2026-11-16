import { useEffect, useRef, useState } from "react";

export default function ChatWindow({ messages, status, onSend }) {
  const [input, setInput] = useState("");
  const endRef = useRef(null);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, status]);

  function submit(e) {
    e.preventDefault();
    const text = input.trim();
    if (!text || status === "loading") return;
    setInput("");
    onSend(text);
  }

  return (
    <div className="chat">
      <div className="messages">
        {messages.map((m, i) => (
          <div key={i} className={`bubble ${m.role}`}>
            {m.text}
          </div>
        ))}
        {status === "loading" && <div className="bubble assistant muted">Thinking...</div>}
        {status === "error" && (
          <div className="bubble assistant error">
            Something went wrong. Please try again.
          </div>
        )}
        <div ref={endRef} />
      </div>
      <form onSubmit={submit} className="input-row">
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Ask an IT or HR question..."
        />
        <button type="submit" disabled={status === "loading"}>Send</button>
      </form>
    </div>
  );
}
