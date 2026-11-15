# Answer quality (end to end)

Produced by `scripts/eval_answers.py`: all 58 eval questions through `Platform.ask`
(recursive chunker, all-MiniLM-L6-v2, hybrid_rerank, k = 5, abstention threshold
-1.31). **Provider: `mock-extractive`.** With the extractive mock every
answer sentence is copied from a retrieved passage, so faithfulness here mostly checks the
citation plumbing (right passage, right page), not an LLM's tendency to hallucinate; an LLM
provider must be re-measured with the same script.

## Faithfulness
Answer sentences supported by a passage they cite (>= 80% of the sentence's
content tokens occur in that passage; uncited sentences count as unsupported):

- sentence level: **139/139 = 1.000**
- mean per answer: 1.000
  over 51 answered questions
- control (same sentences, scored against another question's passages):
  11/139 = 0.079 supported. The check does reject
  text that the cited passages don't contain.

## Citation accuracy
Over the 51 answerable questions that were answered:

- citations pointing at a gold page: **37/128 = 0.289**
  (the mock cites up to 3 passages, and only one usually holds the gold quote, so this
  precision is capped well below 1 by design)
- answers with at least one gold citation: **37/51
  = 0.725**
- answerable questions answered at all: 51/51

## Abstention (positive = abstain; 7 unanswerable, 51 answerable)
| threshold | precision | recall | answerable wrongly abstained |
|---|---|---|---|
| deployed -1.31 (fitted on these 58: in-sample) | 1.000 | 1.000 | 0/51 |
| leave-one-out (each question judged by a threshold fitted on the other 57) | **0.857** | **0.857** | 1/51 |

Leave-one-out thresholds ranged -2.29 .. -0.34. With only 7 negatives, one
question moves recall by 0.14, so these are coarse estimates.
