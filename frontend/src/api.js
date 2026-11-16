export async function sendMessage(sessionId, message) {
  const res = await fetch("/api/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: sessionId, message }),
  });
  if (!res.ok) throw new Error(`Server error (${res.status})`);
  return res.json();
}

export async function resetSession(sessionId) {
  await fetch(`/api/reset/${sessionId}`, { method: "POST" });
}
