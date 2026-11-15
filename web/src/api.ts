// Types mirror rag/schemas.py (the API's OpenAPI models).
export type Mode = "dense" | "bm25" | "hybrid" | "hybrid_rerank";
export const MODES: Mode[] = ["hybrid_rerank", "hybrid", "bm25", "dense"];

export interface Passage {
  chunk_id: number;
  doc_id: string;
  page_start: number;
  page_end: number;
  text: string;
  score: number;
}
export interface Citation {
  doc_id: string;
  page: number;
}
export interface AskResponse {
  question: string;
  answer: string;
  abstained: boolean;
  provider: string;
  top_score: number | null;
  citations: Citation[];
  invalid_citations: Citation[];
  unsupported_sentences: string[];
  passages: Passage[];
}
export interface SearchResponse {
  query: string;
  mode: Mode;
  k: number;
  passages: Passage[];
}
export interface DocumentResponse {
  doc_id: string;
  title: string;
  url: string;
  licence: string;
  n_pages: number;
  pages: string[];
}

async function getJson<T>(url: string, init?: RequestInit): Promise<T> {
  const r = await fetch(url, init);
  if (!r.ok) throw new Error(`${r.status} ${r.statusText}`);
  return (await r.json()) as T;
}

export const api = {
  search: (q: string, mode: Mode, k = 10) =>
    getJson<SearchResponse>(`/api/search?${new URLSearchParams({ q, mode, k: String(k) })}`),
  ask: (question: string) =>
    getJson<AskResponse>("/api/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, k: 5 }),
    }),
  document: (id: string) => getJson<DocumentResponse>(`/api/documents/${encodeURIComponent(id)}`),
};
