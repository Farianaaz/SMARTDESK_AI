import { useState } from "react";
import ChatWindow from "./components/ChatWindow.jsx";
import { sendMessage, resetSession } from "./api.js";

const WELCOME = {
  role: "assistant",
  text: "Hi! I can answer IT/HR questions, raise a support ticket, or check a ticket's status.",
};

export default function App() {
  const [sessionId, setSessionId] = useState(() => crypto.randomUUID());
  const [messages, setMessages] = useState([WELCOME]);
  const [status, setStatus] = useState("idle"); // idle | loading | error

  async function handleSend(text) {
    setMessages((m) => [...m, { role: "user", text }]);
    setStatus("loading");
    try {
      const data = await sendMessage(sessionId, text);
      setMessages((m) => [...m, { role: "assistant", text: data.reply }]);
      setStatus("idle");
    } catch {
      setStatus("error");
    }
  }

  async function handleReset() {
    await resetSession(sessionId).catch(() => {});
    setSessionId(crypto.randomUUID());
    setMessages([WELCOME]);
    setStatus("idle");
  }

  return (
    <div className="app">
      <header>
        <h1>SmartDesk AI</h1>
        <button onClick={handleReset}>New chat</button>
      </header>
      <ChatWindow messages={messages} status={status} onSend={handleSend} />
    </div>
  );
}
