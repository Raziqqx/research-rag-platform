import { useState } from "react";
import { AnswerPanel } from "./AnswerPanel";
import { api, MODES, type AskResponse, type Citation, type Mode, type SearchResponse } from "./api";

export default function App() {
  const [q, setQ] = useState("");
  const [mode, setMode] = useState<Mode>("hybrid_rerank");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [answer, setAnswer] = useState<AskResponse | null>(null);
  const [results, setResults] = useState<SearchResponse | null>(null);
  const [page, setPage] = useState<{ c: Citation; title: string; text: string } | null>(null);

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    if (!q.trim()) return;
    setBusy(true);
    setError(null);
    setPage(null);
    try {
      const [a, s] = await Promise.all([api.ask(q), api.search(q, mode)]);
      setAnswer(a);
      setResults(s);
    } catch (err) {
      setError(String(err));
    } finally {
      setBusy(false);
    }
  }

  async function openPage(c: Citation) {
    const d = await api.document(c.doc_id);
    setPage({ c, title: d.title, text: d.pages[c.page - 1] ?? "" });
  }

  return (
    <main>
      <h1>Research RAG</h1>
      <form onSubmit={submit}>
        <input
          aria-label="question"
          value={q}
          onChange={(e) => setQ(e.target.value)}
          placeholder="Ask about the economics / finance corpus"
        />
        <select aria-label="mode" value={mode} onChange={(e) => setMode(e.target.value as Mode)}>
          {MODES.map((m) => (
            <option key={m}>{m}</option>
          ))}
        </select>
        <button disabled={busy}>{busy ? "Searching…" : "Search"}</button>
      </form>
      {error && <p className="warn">{error}</p>}
      {answer && <AnswerPanel result={answer} onOpenPage={openPage} />}
      {page && (
        <section className="panel" aria-label="document page">
          <h2>
            {page.title}: page {page.c.page}
          </h2>
          <pre>{page.text}</pre>
        </section>
      )}
      {results && (
        <section className="panel" aria-label="results">
          <h2>
            Results ({results.mode}, {results.passages.length})
          </h2>
          <ol>
            {results.passages.map((p) => (
              <li key={p.chunk_id}>
                <strong>
                  {p.doc_id} p.{p.page_start}
                  {p.page_end !== p.page_start && `–${p.page_end}`}
                </strong>{" "}
                <small>score {p.score.toFixed(3)}</small>
                <p>{p.text}</p>
              </li>
            ))}
          </ol>
        </section>
      )}
    </main>
  );
}
