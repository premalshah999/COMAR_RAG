/**
 * API client for COMAR RAG backend.
 * Handles health checks, search requests, and SSE streaming chat.
 */

import type { Filters, HealthStatus, Source } from "../types";

// Base URL — in dev the Vite proxy handles /api, in prod use env var or empty
const API_BASE = import.meta.env.VITE_API_URL ?? "";

// Optional API key for authenticated requests
const API_KEY = import.meta.env.VITE_API_KEY ?? "";

/**
 * Build request headers with optional API key.
 */
function headers(contentType?: string): HeadersInit {
  const h: Record<string, string> = {};
  if (contentType) h["Content-Type"] = contentType;
  if (API_KEY) h["X-API-Key"] = API_KEY;
  return h;
}

// ── Health ──────────────────────────────────────────────────────────────────

/**
 * Fetch backend health status.
 * @returns HealthStatus object with Qdrant and LLM readiness info
 */
export async function fetchHealth(): Promise<HealthStatus> {
  const res = await fetch(`${API_BASE}/api/health`, {
    method: "GET",
    headers: headers(),
  });
  if (!res.ok) {
    throw new Error(`Health check failed: ${res.status}`);
  }
  return res.json();
}

// ── Search ──────────────────────────────────────────────────────────────────

export interface SearchResult {
  results: Source[];
  query: string;
  retrieval_ms: number;
}

/**
 * Direct search without LLM generation.
 * @param query - Search query
 * @param topK - Number of results to return (default 10)
 * @param filters - Optional filters (e.g., { title_num: ["15"] })
 */
export async function search(
  query: string,
  topK = 10,
  filters?: Filters
): Promise<SearchResult> {
  const res = await fetch(`${API_BASE}/api/search`, {
    method: "POST",
    headers: headers("application/json"),
    body: JSON.stringify({
      query,
      top_k: topK,
      filters: filters ?? {},
    }),
  });
  if (!res.ok) {
    throw new Error(`Search failed: ${res.status}`);
  }
  return res.json();
}

// ── Chat (SSE Streaming) ────────────────────────────────────────────────────

export interface StreamEvent {
  token: string;
  done: boolean;
  sources?: Source[];
  conversation_id?: string;
  message_id?: string;
  mode?: "live" | "stub";
  retrieval_ms?: number;
  searching?: boolean;
  intent?: string;
}

/**
 * Stream a chat response as an async generator of events.
 * Uses Server-Sent Events (SSE) for real-time token streaming.
 *
 * @param message - User's message
 * @param conversationId - Conversation ID for context
 * @param filters - Optional title filters
 * @param signal - AbortSignal for cancellation
 * @yields StreamEvent objects containing tokens or final response
 */
export async function* streamChat(
  message: string,
  conversationId: string,
  filters: Filters,
  signal?: AbortSignal
): AsyncGenerator<StreamEvent, void, unknown> {
  const res = await fetch(`${API_BASE}/api/chat`, {
    method: "POST",
    headers: headers("application/json"),
    body: JSON.stringify({
      message,
      conversation_id: conversationId,
      filters,
      top_k: 10,
    }),
    signal,
  });

  if (!res.ok) {
    throw new Error(`Chat request failed: ${res.status}`);
  }

  if (!res.body) {
    throw new Error("No response body");
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });

      // Parse SSE events from buffer
      const lines = buffer.split("\n");
      buffer = lines.pop() ?? ""; // Keep incomplete line in buffer

      for (const line of lines) {
        const trimmed = line.trim();
        if (!trimmed || trimmed.startsWith(":")) continue; // Skip comments/keepalives

        if (trimmed.startsWith("data: ")) {
          const jsonStr = trimmed.slice(6);
          if (jsonStr === "[DONE]") {
            // OpenAI-style done signal
            return;
          }
          try {
            const event = JSON.parse(jsonStr) as StreamEvent;
            yield event;
            if (event.done) return;
          } catch {
            // Skip malformed JSON
            console.warn("Failed to parse SSE event:", jsonStr);
          }
        }
      }
    }

    // Process any remaining data in buffer
    if (buffer.trim()) {
      const trimmed = buffer.trim();
      if (trimmed.startsWith("data: ")) {
        const jsonStr = trimmed.slice(6);
        if (jsonStr !== "[DONE]") {
          try {
            const event = JSON.parse(jsonStr) as StreamEvent;
            yield event;
          } catch {
            // Ignore
          }
        }
      }
    }
  } finally {
    reader.releaseLock();
  }
}

// ── Intent Classification ───────────────────────────────────────────────────

export interface IntentResult {
  intent: "regulatory" | "definitional" | "procedural" | "comparative" | "conversational";
  confidence: number;
}

/**
 * Classify the intent of a user message.
 * Useful for UI hints or conditional logic.
 */
export async function classifyIntent(message: string): Promise<IntentResult> {
  const res = await fetch(`${API_BASE}/api/intent`, {
    method: "POST",
    headers: headers("application/json"),
    body: JSON.stringify({ message }),
  });
  if (!res.ok) {
    throw new Error(`Intent classification failed: ${res.status}`);
  }
  return res.json();
}

// ── Stats ───────────────────────────────────────────────────────────────────

export interface Stats {
  regulations: number;
  chunks: number;
  graph_nodes: number;
  graph_edges: number;
  titles: string[];
  definitions: number;
}

/**
 * Fetch ingestion statistics.
 */
export async function fetchStats(): Promise<Stats> {
  const res = await fetch(`${API_BASE}/api/stats`, {
    method: "GET",
    headers: headers(),
  });
  if (!res.ok) {
    throw new Error(`Stats fetch failed: ${res.status}`);
  }
  return res.json();
}
