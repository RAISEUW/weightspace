# Stages 1–2: reproduce the evaluation and verify the released edits

Authors' `think35` protocol via their `evaluate.py` (vLLM 0.32.0+cu129, 4× H100 NVL, one shard per GPU): chat template
with thinking on, temperature 0.6, top-p 0.95, top-k 20, presence penalty 1.5, max 32,768 new tokens, last `\boxed{}`
graded with math-verify 0.9.0. MATH-500: 4 samples/problem (2,000 answers); GSM8K: 2 samples/problem (2,638 answers).
Edited checkpoints rebuilt from the public `Qwen/Qwen3.5-0.8B` (rev `2fc0636`) with the released diffs; every edited
tensor matches the authors' sha256.

| Model | MATH-500 ours (±SE) | paper | GSM8K ours (±SE) | paper | cut at limit, MATH-500 ours / paper |
|---|---|---|---|---|---|
| Unedited | 8.0 ± 0.6 | 7.9 | 13.0 ± 0.7 | 13.8 | 92.7 / 93.0 |
| p10c (10 passes) | 13.5 ± 0.8 | 12.3 | 25.5 ± 0.8 | 25.7 | 86.5 / 87.2 |
| p25c (25 passes) | 14.2 ± 0.8 | 14.1 | 25.9 ± 0.9 | 25.1 | 86.2 / 86.4 |
| p50c (50 passes) | 48.9 ± 1.1 | 48.1 | 68.1 ± 0.9 | 67.1 | 32.4 / 33.1 |

Every number is within about 1.5 standard errors of the paper's. The 50-pass edit (131,072 weights in the MLP
down-projections of layers 15 and 19) raises MATH-500 from 8.0 to 48.9 and GSM8K from 13.0 to 68.1, mostly by stopping
the thinking loops (answers cut at 32k tokens fall from 93% to 32% on MATH-500).

Files: `<model>/merged.json` (from `ws/merge.py`), `<model>/parts/s<g>of4/` (`scores.json` and per-problem 0/1 results).

See `../../DISCUSSION.md` for the criticisms, thinking-block statistics and thinking-off evaluations.
