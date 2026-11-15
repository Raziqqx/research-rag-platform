# Abstention threshold calibration (hybrid_rerank, recursive collection)

Produced by `scripts/calibrate_abstain.py`. Score = top cross-encoder logit.

- answerable questions: 51, top score min/median/max = -0.70 / 4.83 / 9.05
- unanswerable questions: 7, top score min/median/max = -9.57 / -4.38 / -1.91
- chosen threshold: **-1.31** (balanced accuracy 1.000): answers 51/51 answerable, abstains on 7/7 unanswerable

Calibrated and reported on the same 58 questions (no held-out split; 7 negatives).
