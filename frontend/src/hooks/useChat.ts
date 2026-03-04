import { useCallback, useRef, useState } from "react";
import { streamChat } from "../lib/api";
import type { Conversation, Filters, Message, Source } from "../types";

const newConv = (): Conversation => ({
  id: crypto.randomUUID(),
  title: "New conversation",
  messages: [],
  createdAt: new Date(),
  updatedAt: new Date(),
});

export function useChat() {
  const [conversations, setConversations] = useState<Conversation[]>([newConv()]);
  const [activeId, setActiveId] = useState<string>(() => conversations[0].id);
  const [isStreaming, setIsStreaming] = useState(false);
  const abortRef = useRef<AbortController | null>(null);

  const active = conversations.find((c) => c.id === activeId) ?? conversations[0];

  // ── Helpers ──────────────────────────────────────────────────────────────

  const updateConv = useCallback((id: string, updater: (c: Conversation) => Conversation) => {
    setConversations((prev) => prev.map((c) => (c.id === id ? updater(c) : c)));
  }, []);

  const patchMessage = useCallback(
    (convId: string, msgId: string, patch: Partial<Message>) => {
      updateConv(convId, (c) => ({
        ...c,
        messages: c.messages.map((m) => (m.id === msgId ? { ...m, ...patch } : m)),
        updatedAt: new Date(),
      }));
    },
    [updateConv]
  );

  // ── Send ─────────────────────────────────────────────────────────────────

  const send = useCallback(
    async (text: string, filters: Filters) => {
      if (!text.trim() || isStreaming) return;

      const convId = activeId;
      const userMsg: Message = {
        id: crypto.randomUUID(),
        role: "user",
        content: text.trim(),
        sources: [],
        timestamp: new Date(),
      };
      const assistantMsgId = crypto.randomUUID();
      const assistantMsg: Message = {
        id: assistantMsgId,
        role: "assistant",
        content: "",
        sources: [],
        timestamp: new Date(),
        isStreaming: true,
      };

      // Derive conversation title from first message
      updateConv(convId, (c) => ({
        ...c,
        title:
          c.messages.length === 0
            ? text.slice(0, 48) + (text.length > 48 ? "…" : "")
            : c.title,
        messages: [...c.messages, userMsg, assistantMsg],
        updatedAt: new Date(),
      }));

      setIsStreaming(true);
      abortRef.current = new AbortController();

      try {
        let accumulated = "";
        let finalSources: Source[] = [];
        let mode: "live" | "stub" = "stub";
        let retrieval_ms = 0;

        for await (const event of streamChat(
          text,
          convId,
          filters,
          abortRef.current.signal
        )) {
          if (event.done) {
            finalSources = event.sources ?? [];
            mode = event.mode ?? "stub";
            retrieval_ms = event.retrieval_ms ?? 0;
            break;
          }
          accumulated += event.token;
          patchMessage(convId, assistantMsgId, { content: accumulated });
        }

        patchMessage(convId, assistantMsgId, {
          content: accumulated,
          sources: finalSources,
          isStreaming: false,
          mode,
          retrieval_ms,
        });
      } catch (err: unknown) {
        if (err instanceof Error && err.name === "AbortError") {
          patchMessage(convId, assistantMsgId, { isStreaming: false });
        } else {
          patchMessage(convId, assistantMsgId, {
            content: "⚠️ Something went wrong connecting to the API. Make sure the backend is running on port 8000.",
            isStreaming: false,
          });
        }
      } finally {
        setIsStreaming(false);
        abortRef.current = null;
      }
    },
    [activeId, isStreaming, updateConv, patchMessage]
  );

  const stop = useCallback(() => {
    abortRef.current?.abort();
  }, []);

  const newChat = useCallback(() => {
    const c = newConv();
    setConversations((prev) => [c, ...prev]);
    setActiveId(c.id);
  }, []);

  const selectConv = useCallback((id: string) => {
    setActiveId(id);
  }, []);

  return {
    conversations,
    active,
    isStreaming,
    send,
    stop,
    newChat,
    selectConv,
  };
}
