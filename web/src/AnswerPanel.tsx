import { useState } from "react";
import type { AskResponse, Citation, Passage } from "./api";

const CITE = /\[([A-Za-z0-9][\w.-]*):(\d+)\]/g;

/** Split answer text into plain strings and citation markers. */
export function parseAnswer(text: string): (string | Citation)[] {
  const out: (string | Citation)[] = [];
  let last = 0;
  for (const m of text.matchAll(CITE)) {
    if (m.index! > last) out.push(text.slice(last, m.index));
    out.push({ doc_id: m[1], page: Number(m[2]) });
    last = m.index! + m[0].length;
  }
  if (last < text.length) out.push(text.slice(last));
  return out;
}

function passageFor(c: Citation, passages: Passage[]): Passage | undefined {
  return passages.find((p) => p.doc_id === c.doc_id && p.page_start <= c.page && c.page <= p.page_end);
}

interface Props {
  result: AskResponse;
  onOpenPage?: (c: Citation) => void;
}

/** Answer with clickable [doc:page] citations; a click shows the cited passage and page. */
export function AnswerPanel({ result, onOpenPage }: Props) {
  const [open, setOpen] = useState<Citation | null>(null);
  const passage = open && passageFor(open, result.passages);
  return (
    <section className="panel" aria-label="answer">
      <h2>Answer</h2>
      <p className={result.abstained ? "abstained" : undefined}>
        {parseAnswer(result.answer).map((part, i) =>
          typeof part === "string" ? (
            <span key={i}>{part}</span>
          ) : (
            <button key={i} className="cite" onClick={() => setOpen(part)}>
              [{part.doc_id}:{part.page}]
            </button>
          ),
        )}
      </p>
      {result.invalid_citations.length > 0 && (
        <p className="warn">Removed {result.invalid_citations.length} unsupported citation(s).</p>
      )}
      {result.unsupported_sentences.length > 0 && (
        <p className="warn">Uncited: {result.unsupported_sentences.join(" ")}</p>
      )}
      <small>
        provider {result.provider} · top rerank score {result.top_score?.toFixed(2) ?? "n/a"}
      </small>
      {open && (
        <blockquote data-testid="cited-passage">
          <strong>
            {open.doc_id}, page {open.page}
          </strong>
          <p>{passage ? passage.text : "Passage not in the retrieved set."}</p>
          {onOpenPage && <button onClick={() => onOpenPage(open)}>Show full page</button>}
        </blockquote>
      )}
    </section>
  );
}
