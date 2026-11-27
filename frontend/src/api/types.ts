// Mirrors backend/app/schemas (rag.py + api.py).

export type ConfidenceLevel = "HIGH" | "MEDIUM" | "LOW";

export interface Citation {
  id: number;
  source: string;
  title: string;
  page: number;
  snippet: string;
  similarity: number;
}

export interface Confidence {
  level: ConfidenceLevel;
  score: number;
  reasons: string[];
}

export interface QueryResponse {
  question: string;
  answer: string;
  citations: Citation[];
  confidence: Confidence;
  insufficient_evidence: boolean;
  retrieved_chunks: number;
  model: string;
  latency_ms: number;
  /** true when the server answered from its cache (no new LLM call) */
  cached?: boolean;
}

export interface QueryRequest {
  question: string;
  top_k?: number;
}

export interface HealthResponse {
  status: "ok" | "degraded";
  version: string;
  environment: string;
  llm_model: string;
  llm_configured: boolean;
  index: { documents: number; chunks: number } | null;
  issues: string[];
}

export interface SourceDocument {
  source: string;
  title: string;
  total_pages: number | null;
  chunks: number;
}

export interface SourcesResponse {
  count: number;
  documents: SourceDocument[];
}

export interface ErrorBody {
  error: { code: string; message: string; details?: { field: string; message: string }[] };
  request_id?: string | null;
}
