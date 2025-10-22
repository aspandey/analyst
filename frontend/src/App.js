import React, { useState, useEffect, useRef } from "react";
import ReactDOM from "react-dom/client";
import "./App.css";
import ReactMarkdown from "react-markdown";


function ChatApp() {
  const [messages, setMessages] = useState([
    { role: "bot", text: "Hello — ask me anything." },
  ]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const listRef = useRef(null);

  useEffect(() => {
    if (listRef.current) {
      listRef.current.scrollTop = listRef.current.scrollHeight;
    }
  }, [messages, loading]);

  let activeController = null;
  let activeReader = null;

  async function sendMessage() {
    const trimmed = input.trim();
    if (!trimmed) return;

    try {
      if (activeReader?.cancel) activeReader.cancel();
      if (activeController) activeController.abort();
    } catch (e) {}

    const userMsg = { role: "user", text: trimmed };
    setMessages((prev) => [...prev, userMsg]);
    setInput("");
    setLoading(true);
    setError(null);

    setMessages((prev) => [...prev, { role: "bot", text: "" }]);

    const controller = new AbortController();

    // label observer
    if (!window.__analystChatObserver) {
      const observer = new MutationObserver(() => {
        const container = listRef.current;
        if (!container) return;
        container.querySelectorAll(".message").forEach((el) => {
          if (el.querySelector(".label")) return;
          const label = document.createElement("span");
          label.className = "label";
          label.textContent = el.classList.contains("user")
            ? "You: "
            : "Analyst: ";
          el.insertBefore(label, el.firstChild);
        });
      });
      if (listRef.current)
        observer.observe(listRef.current, {
          childList: true,
          subtree: true,
          characterData: true,
        });
      window.__analystChatObserver = observer;
    }

    activeController = controller;

    try {
      const res = await fetch("http://localhost:8000/stocks_info", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: trimmed }),
        signal: controller.signal,
      });

      if (!res.ok) throw new Error("Network error");

      if (!res.body) {
        const text = await res.text();
        setMessages((prev) => {
          const copy = [...prev];
          copy[copy.length - 1] = { role: "bot", text };
          return copy;
        });
        return;
      }

      const reader = res.body.getReader();
      activeReader = reader;
      const decoder = new TextDecoder();
      let done = false;
      let accumulated = "";

      while (!done) {
        const { value, done: readerDone } = await reader.read();
        if (value) {
          const chunk = decoder.decode(value, { stream: !readerDone });
          accumulated += chunk;

          setMessages((prev) => {
            const copy = [...prev];
            const last = copy[copy.length - 1];
            if (last && last.role === "bot") {
              copy[copy.length - 1] = { role: "bot", text: accumulated };
            } else {
              copy.push({ role: "bot", text: accumulated });
            }
            return copy;
          });
        }
        done = readerDone;
      }
    } catch (err) {
      if (err.name !== "AbortError") {
        console.error(err);
        setError(err.message || "Unknown error");
        setMessages((prev) => {
          const copy = [...prev];
          copy[copy.length - 1] = {
            role: "bot",
            text: "Sorry, I could not get a response.",
          };
          return copy;
        });
      }
    } finally {
      setLoading(false);
      activeController = null;
      activeReader = null;
    }
  }

  function onKeyDown(e) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      if (!loading) sendMessage();
    }
  }

  return (
    <div className="chat-wrapper">
      <div className="chat-card">
        <div className="chat-header">
          <h1>Analyst Chat</h1>
          <p className="subtitle">Your friendly stock assistant</p>
        </div>

        <div className="chat-messages" ref={listRef}>
          {messages.map((m, i) => (
            <div
              key={i}
              className={`chat-bubble ${m.role === "user" ? "user" : "bot"}`}
            >
            <ReactMarkdown>{m.text}</ReactMarkdown>
            </div>
          ))}
          {loading && <div className="typing">Analyst is typing...</div>}
        </div>

        <div className="chat-input">
          <textarea
            placeholder="Ask your question..."
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={onKeyDown}
          />
          <button onClick={sendMessage} disabled={loading}>
            {loading ? "..." : "Send"}
          </button>
        </div>

        {error && <div className="error-box">⚠️ {error}</div>}
      </div>
    </div>
  );
}

const root = ReactDOM.createRoot(document.getElementById("root"));
root.render(<ChatApp />);

export default ChatApp;
