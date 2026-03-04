import { useState } from "react";
import type { Message } from "../types";
import { CitationCard } from "./CitationCard";

interface Props {
  message: Message;
}

/* ── Lightweight markdown renderer ───────────────────────────────────── */
function inline(text: string): React.ReactNode {
  const parts = text.split(/(\*\*[^*]+\*\*|`[^`]+`|\*[^*]+\*)/g);
  return parts.map((part, i) => {
    if (part.startsWith("**") && part.endsWith("**"))
      return <strong key={i}>{part.slice(2, -2)}</strong>;
    if (part.startsWith("`") && part.endsWith("`"))
      return <code key={i}>{part.slice(1, -1)}</code>;
    if (part.startsWith("*") && part.endsWith("*"))
      return <em key={i}>{part.slice(1, -1)}</em>;
    return part;
  });
}

function renderContent(text: string): React.ReactNode {
  const lines = text.split("\n");
  const nodes: React.ReactNode[] = [];
  let i = 0;
  while (i < lines.length) {
    const line = lines[i];
    // Unordered list — collect consecutive bullet lines into one <ul>
    if (line.startsWith("- ") || line.startsWith("* ")) {
      const items: React.ReactNode[] = [];
      while (i < lines.length && (lines[i].startsWith("- ") || lines[i].startsWith("* "))) {
        items.push(<li key={i}>{inline(lines[i].slice(2))}</li>);
        i++;
      }
      nodes.push(<ul key={`ul-${nodes.length}`}>{items}</ul>);
      continue;
    }
    // Ordered list — collect consecutive numbered lines into one <ol>
    const numbered = line.match(/^(\d+)\. (.+)/);
    if (numbered) {
      const items: React.ReactNode[] = [];
      while (i < lines.length) {
        const m = lines[i].match(/^(\d+)\. (.+)/);
        if (!m) break;
        items.push(<li key={i}>{inline(m[2])}</li>);
        i++;
      }
      nodes.push(<ol key={`ol-${nodes.length}`}>{items}</ol>);
      continue;
    }
    if (line.startsWith("### ")) { nodes.push(<h3 key={i}>{inline(line.slice(4))}</h3>); i++; continue; }
    if (line.startsWith("## "))  { nodes.push(<h2 key={i}>{inline(line.slice(3))}</h2>); i++; continue; }
    if (line.startsWith("# "))   { nodes.push(<h2 key={i}>{inline(line.slice(2))}</h2>); i++; continue; }
    if (line.startsWith("> "))   { nodes.push(<blockquote key={i}>{inline(line.slice(2))}</blockquote>); i++; continue; }
    if (!line.trim())             { nodes.push(<br key={i} />); i++; continue; }
    nodes.push(<p key={i}>{inline(line)}</p>);
    i++;
  }
  return nodes;
}

function TypingDots() {
  return (
    <div className="typing">
      <span /><span /><span />
    </div>
  );
}

function ChevronDown() {
  return (
    <svg width={10} height={10} fill="none" stroke="currentColor" strokeWidth={2}
      strokeLinecap="round" strokeLinejoin="round" viewBox="0 0 24 24">
      <polyline points="6 9 12 15 18 9"/>
    </svg>
  );
}
function ChevronUp() {
  return (
    <svg width={10} height={10} fill="none" stroke="currentColor" strokeWidth={2}
      strokeLinecap="round" strokeLinejoin="round" viewBox="0 0 24 24">
      <polyline points="18 15 12 9 6 15"/>
    </svg>
  );
}

export function MessageBubble({ message }: Props) {
  const isUser = message.role === "user";
  const [sourcesOpen, setSourcesOpen] = useState(false);
  const hasSources = message.sources.length > 0;

  return (
    <div className={`msg msg-${message.role}`}>
      <div className="msg-bubble">
        {isUser ? (
          <p>{message.content}</p>
        ) : (
          <div className="msg-prose">
            {message.content
              ? renderContent(message.content)
              : message.isStreaming
              ? <TypingDots />
              : null}
          </div>
        )}
      </div>

      {/* Meta row (assistant only, after streaming) */}
      {!isUser && !message.isStreaming && (
        <div className="msg-meta">
          {message.retrieval_ms !== undefined && message.retrieval_ms > 0 && (
            <span className="msg-meta-item">
              {Math.round(message.retrieval_ms)}ms
            </span>
          )}
          {message.mode === "stub" && (
            <span className="msg-meta-item" style={{ color: "var(--text-muted)" }}>
              stub mode
            </span>
          )}
          {hasSources && (
            <button className="msg-sources-btn" onClick={() => setSourcesOpen((o) => !o)}>
              {message.sources.length} source{message.sources.length !== 1 ? "s" : ""}
              {sourcesOpen ? <ChevronUp /> : <ChevronDown />}
            </button>
          )}
        </div>
      )}

      {/* Expanded citation cards */}
      {hasSources && sourcesOpen && (
        <div style={{ display: "flex", flexDirection: "column", gap: 6, width: "100%", maxWidth: 640 }}>
          {message.sources.map((s, i) => (
            <CitationCard key={s.citation + i} source={s} rank={i + 1} />
          ))}
        </div>
      )}
    </div>
  );
}
