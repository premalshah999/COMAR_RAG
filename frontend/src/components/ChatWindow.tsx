import { useEffect, useRef } from "react";
import type { Conversation } from "../types";
import { MessageBubble } from "./MessageBubble";

interface Props {
  conversation: Conversation;
  isStreaming: boolean;
}

function LawIcon() {
  return (
    <svg width={24} height={24} fill="none" stroke="currentColor" strokeWidth={2}
      strokeLinecap="round" strokeLinejoin="round" viewBox="0 0 24 24">
      <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>
    </svg>
  );
}

const EXAMPLE_PROMPTS = [
  { label: "Pesticides",      sub: "Regulations under COMAR Title 15 for pesticide applicators" },
  { label: "Water Quality",   sub: "Title 26 water quality standards and permit requirements" },
  { label: "Penalties",       sub: "Enforcement actions and civil penalties in agriculture" },
  { label: "Definitions",     sub: "COMAR definitions for key regulatory terms" },
];

function WelcomeScreen() {
  return (
    <div className="empty-state fade-in">
      <div className="empty-icon"><LawIcon /></div>
      <div className="empty-title">Ask anything about COMAR</div>
      <div className="empty-sub">
        Maryland Agriculture (Title 15) and Environment (Title 26) regulations —
        semantically searched and answered with full citations.
      </div>
      <div className="suggestions">
        {EXAMPLE_PROMPTS.map((p) => (
          <div key={p.label} className="suggestion-chip">
            <div className="suggestion-chip-label">{p.label}</div>
            <div className="suggestion-chip-sub">{p.sub}</div>
          </div>
        ))}
      </div>
    </div>
  );
}

export function ChatWindow({ conversation, isStreaming }: Props) {
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (isStreaming) {
      bottomRef.current?.scrollIntoView({ behavior: isStreaming ? "auto" : "smooth" });
    }
  }, [conversation.messages, isStreaming]);

  const isEmpty = conversation.messages.length === 0;

  return (
    <div className="messages-wrap">
      {isEmpty ? (
        <WelcomeScreen />
      ) : (
        <div className="messages-inner">
          {conversation.messages.map((msg) => (
            <MessageBubble key={msg.id} message={msg} />
          ))}
        </div>
      )}
      <div ref={bottomRef} />
    </div>
  );
}
