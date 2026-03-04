import React, { useEffect, useState } from "react";
import { fetchHealth } from "./lib/api";
import { useTheme } from "./hooks/useTheme";
import { useChat } from "./hooks/useChat";
import { Landing } from "./components/Landing";
import { Sidebar } from "./components/Sidebar";
import { ChatWindow } from "./components/ChatWindow";
import { InputBar } from "./components/InputBar";
import { TopBar } from "./components/TopBar";
import type { Filters, HealthStatus } from "./types";

class ErrorBoundary extends React.Component<
  { children: React.ReactNode },
  { error: Error | null }
> {
  constructor(props: { children: React.ReactNode }) {
    super(props);
    this.state = { error: null };
  }
  static getDerivedStateFromError(error: Error) {
    return { error };
  }
  render() {
    if (this.state.error) {
      return (
        <div style={{ padding: 40, fontFamily: "monospace", color: "#f0f0f0", background: "#080808", height: "100vh" }}>
          <h2 style={{ marginBottom: 12 }}>Something went wrong</h2>
          <p style={{ color: "#888", marginBottom: 20 }}>{this.state.error.message}</p>
          <button
            onClick={() => window.location.reload()}
            style={{ padding: "8px 16px", background: "#1a1a1a", border: "1px solid #333", color: "#f0f0f0", borderRadius: 6, cursor: "pointer" }}
          >
            Reload
          </button>
        </div>
      );
    }
    return this.props.children;
  }
}

function AppInner() {
  const [page, setPage] = useState<"landing" | "chat">("landing");
  const { theme, toggle } = useTheme();
  const { conversations, active, isStreaming, send, stop, newChat, selectConv } = useChat();

  const [health, setHealth] = useState<HealthStatus | null>(null);
  const [filters, setFilters] = useState<Filters>({ title_num: [] });

  // Poll health every 15 s
  useEffect(() => {
    const poll = async () => {
      try {
        const h = await fetchHealth();
        setHealth(h);
      } catch {
        setHealth({
          status: "error",
          qdrant_connected: false,
          qdrant_points: 0,
          qdrant_collection: "comar_regulations",
          llm_ready: false,
          llm_model: "—",
        });
      }
    };
    poll();
    const id = setInterval(poll, 15_000);
    return () => clearInterval(id);
  }, []);

  const handleSend = (text: string) => send(text, filters);

  if (page === "landing") {
    return <Landing onEnter={() => setPage("chat")} />;
  }

  return (
    <div className="app page-enter">
      {/* Sidebar */}
      <Sidebar
        conversations={conversations}
        activeId={active.id}
        onSelect={selectConv}
        onNew={newChat}
        filters={filters}
        onFiltersChange={setFilters}
        onHome={() => setPage("landing")}
        isDark={theme === "dark"}
        onToggleTheme={toggle}
      />

      {/* Main area */}
      <main className="main">
        <TopBar
          conversation={active}
          health={health}
          onHome={() => setPage("landing")}
          isDark={theme === "dark"}
          onToggleTheme={toggle}
        />
        <div className="legal-strip">
          Research tool — not legal advice. Verify at{" "}
          <a href="https://regs.maryland.gov" target="_blank" rel="noopener noreferrer">
            regs.maryland.gov
          </a>
          {" "}before taking any regulatory action.
        </div>
        <ChatWindow conversation={active} isStreaming={isStreaming} />
        <InputBar onSend={handleSend} onStop={stop} isStreaming={isStreaming} />
      </main>
    </div>
  );
}

export default function App() {
  return (
    <ErrorBoundary>
      <AppInner />
    </ErrorBoundary>
  );
}
