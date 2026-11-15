# How the eval set was built

`eval/questions.jsonl` holds 58 questions over the 22-document corpus in `corpus/manifest.json`.

## Procedure
1. Each PDF was parsed with the production parser (`rag/parse.py`), so quotes are taken from the
   same canonical text the system indexes.
2. For every document, the annotator read the first body paragraph plus three paragraphs sampled
   (seeded by doc id) from the first 80 % of the document, skipping reference lists. Where that
   sample was thin (math-heavy papers whose layout blocks are single lines), the abstract and
   introduction were read instead.
3. Questions were written from a passage, then **reworded away from the passage's wording**
   wherever a natural paraphrase existed (e.g. "China+1 opportunity" became "firms diversifying
   production away from China"), so lexical match alone doesn't solve the set.
4. Each gold passage records `doc_id`, the 1-based `page` and a verbatim `quote` (a sentence or
   clause that contains the answer). A reference `answer` is stored for run C's faithfulness
   scoring.
5. Unanswerable questions were written to be topically close to the corpus (central banks,
   sovereign debt, hedging, crypto) but about facts no document contains; the annotator
   checked the corpus by keyword search for each one.

## Validation (automatic, on every load)
`rag/evalset.py` locates every quote in its document (case- and whitespace-insensitive) and
fails if a quote is missing or starts on a different page than recorded. All 58 load cleanly.

## Mix
| type | n | what it tests |
|---|---|---|
| factual | 20 | single-fact lookup, mostly paraphrased |
| conceptual | 15 | "why/how" questions whose answer is a claim or mechanism |
| numeric | 13 | a specific number, where exact-term (BM25) matching should help |
| multi_doc | 3 | two gold passages in different documents; recall@k needs both |
| no_answer | 7 | nothing in the corpus answers it; excluded from recall/MRR, used for abstention in run C |

## Relevance rule
A retrieved chunk covers a gold passage if it is from the same document and overlaps at least
50 % of the quote's characters (`rag/metrics.py:covers`). Matching on character ranges rather
than chunk ids lets one eval set score every chunking strategy.

## Known biases
- One annotator (the builder), and questions were written from sampled passages, so the set
  over-represents abstracts/introductions (page 1–2) where papers state their findings.
- Gold passages are one quote each; another chunk could also answer a question and is counted
  as a miss. This makes recall a lower bound.
- 51 answerable questions give coarse estimates: one question is ~2 recall points.
