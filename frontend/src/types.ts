export interface Source {
  citation: string;
  title_name: string;
  subtitle_name: string;
  chapter_name: string;
  regulation_name: string;
  text_snippet: string;
  score: number;
  chunk_type: "regulation" | "definition" | "subsection";
  effective_date: string;
  context_path: string;
}

export interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  sources: Source[];
  timestamp: Date;
  isStreaming?: boolean;
  mode?: "live" | "stub";
  retrieval_ms?: number;
}

export interface Conversation {
  id: string;
  title: string;
  messages: Message[];
  createdAt: Date;
  updatedAt: Date;
}

export interface HealthStatus {
  status: "ok" | "degraded" | "error" | "loading";
  qdrant_connected: boolean;
  qdrant_points: number;
  qdrant_collection: string;
  llm_ready: boolean;
  llm_model: string;
}

export interface Stats {
  regulations: number;
  chunks: number;
  graph_nodes: number;
  graph_edges: number;
  titles: string[];
  definitions: number;
}

export type Theme = "dark" | "light";

export interface Filters {
  title_num: string[];
}
