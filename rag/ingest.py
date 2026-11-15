"""Ingest pipeline: manifest PDFs -> parse -> chunk -> embed -> Postgres.

python -m rag.ingest                          # all strategies, default model
python -m rag.ingest --strategies recursive   # one strategy
"""

from __future__ import annotations

import argparse
import json
import time

from rag import db
from rag.chunking import STRATEGIES, Span, chunk
from rag.config import get_settings
from rag.embed import Embedder
from rag.parse import Document, parse_pdf


def chunk_rows(doc: Document, spans: list[Span], embed: Embedder):
    texts = [doc.text[s.start : s.end] for s in spans]
    vecs = embed(texts) if texts else []
    for s, t, v in zip(spans, texts, vecs, strict=True):
        yield (s.start, s.end, doc.page_of(s.start), doc.page_of(s.end - 1), t, v)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategies", nargs="+", default=list(STRATEGIES), choices=STRATEGIES)
    ap.add_argument("--model", default=None, help="embedding model (default from env)")
    args = ap.parse_args()

    s = get_settings()
    model = args.model or s.embedding_model
    embed = Embedder(model)
    conn = db.connect(s.database_url)
    print("migrations applied:", db.migrate(conn) or "none")
    manifest = json.loads(s.manifest_path.read_text())

    t0 = time.perf_counter()
    totals = dict.fromkeys(args.strategies, 0)
    for meta in manifest:
        doc = parse_pdf(s.pdf_dir / f"{meta['doc_id']}.pdf", meta["doc_id"])
        db.upsert_document(conn, meta, doc.pages)
        for strat in args.strategies:
            spans = chunk(doc.text, strat, embed=embed)
            totals[strat] += db.replace_chunks(
                conn, doc.doc_id, strat, model, chunk_rows(doc, spans, embed)
            )
        print(f"{doc.doc_id}: {len(doc.pages)} pages, {len(doc.text)} chars")
    for strat in args.strategies:
        db.ensure_hnsw_index(conn, strat, model)
    print(f"chunks per strategy: {totals}  ({time.perf_counter() - t0:.0f}s)")


if __name__ == "__main__":
    main()
