# Discussion: criticisms of Experiment 8 and what our runs show

Experiment 8 of *Sparks of Weight Space Programming* reports that an LLM editor (`gpt-6-astra`) raised Qwen3.5-0.8B
from 7.9% to 48.1% on MATH-500 by editing 0.016% of its weights, with no gradient descent. We reproduced the evaluation
and the released edits (`results/stage1_2/RESULTS.md`), then tested the main criticisms raised online.

## Summary

- **The published numbers reproduce.** Thinking on, the base model scores 8.0 / 13.0 (MATH-500 / GSM8K) and the 50-pass
  edit scores 48.9 / 68.1, against 7.9 / 13.8 and 48.1 / 67.1 in the post.
- **Every published number is thinking-on against thinking-on.** The chat template opened a `<think>` block for every
  answer in our evaluation and in the authors'.
- **The edit does not switch thinking off.** The edited model still reasons at length before it answers (median about
  16k thinking tokens on MATH-500). It closes its thinking block far more often (76% vs 6.5%) instead of looping until
  the 32k-token cap.
- **The gain is a loop fix in thinking mode.** With thinking off, the unedited model already scores 50.6 / 56.2, and the
  edit changes nothing there (50.5 / 55.8). The weights could already solve these problems. Thinking mode was failing
  because it looped, and the edit repairs that.
- **Our view:** thinking-on and thinking-off Qwen behave like two different models, so the right control for the edit
  is the unedited model in the same mode, and that is the comparison the paper makes. Repairing a known failure of
  thinking mode by editing weights directly, with a mechanism you can point to, is still an interesting result. But the
  post's framing ("better at math") overstates it. The edit does not add math ability the model lacked, and the
  post does not say that the same model scores about 50% with thinking off.

## The criticisms

1. **"The baseline is broken / cherry-picked."** Qwen's model card says the 0.8B "is more prone to entering thinking
   loops compared to other Qwen3.5 models" and "operates in non-thinking mode by default"
   ([model card](https://huggingface.co/Qwen/Qwen3.5-0.8B)). The authors evaluated thinking mode, where 93% of MATH-500
   answers hit the 32,768-token cap. Hacker News commenters reported about 52–55% for thinking off
   ([HN thread](https://news.ycombinator.com/item?id=50341174); summarized on
   [dev.to](https://dev.to/alanwest/gpt-6-astra-edited-qwen35-08bs-weights-the-math-500-jump-was-a-loop-fix-2g5a)).
   Carlos Toledo's re-analysis found 44–50% for the unedited model under the card's non-thinking settings and concluded
   "the gain is termination" ([report](https://carlostoledo.co/reports/weight-space-programming.html)).
2. **"The edit just turned thinking off."** We found no evidence for this (see the next section). The authors' code
   does contain thinking-off settings (`profiles.py`, profile `qwen35`: "thinking OFF (2026-10-23)", and `evaluate.py`,
   protocol `nothink35`), but those are for their earlier prompt-v2 runs. The three published runs record
   `mode: qwen35think` in `state.json`. Every evaluation shard behind the published table records protocol `think35`
   with `thinking: true`, and all of Astra's generation requests that set the flag used `"enable_thinking": true`.
3. **"A human would have done the same."** Critics argued that the Qwen fix was standard ML engineering, the kind of
   thing a competent researcher would have done
   ([R&D World](https://www.rdworldonline.com/mit-team-says-gpt-6-astra-can-fix-an-ai-model-by-editing-its-weights-critics-say-the-method-was-standard-ml-engineering/)).
   That is a fair point about novelty, but it is not a problem with the measurement. We have not yet tested a
   non-agentic baseline (for example, ablating the top repetition-correlated direction automatically).

## Finding 1: the edited model still thinks

These numbers come from the authors' released first sample of each problem (200 per set per model), using the same
weights we evaluated. Thinking length is the number of tokens before `</think>`, or the whole answer if the block never
closes. Script: `analysis/thinking_stats.py`, output in `analysis/thinking_stats.json`.

| Model | Set | Closed `</think>` | Median thinking tokens (closed answers) | Thinking < 20 tokens | Accuracy when closed |
|---|---|---|---|---|---|
| Unedited | MATH-500 | 6.5% | 5,398 | 0 | 100% |
| Unedited | GSM8K | 10.5% | 3,650 | 0 | 95% |
| p10c | MATH-500 | 20.5% | 12,153 | 0 | 81% |
| p25c | MATH-500 | 14.5% | 8,028 | 0 | 86% |
| p50c | MATH-500 | 76.0% | 15,944 | 0 | 61% |
| p50c | GSM8K | 94.0% | 8,995 | 0 | 65% |

No model, edited or not, produces an empty thinking block. The unedited model closes its thinking only on easy
problems, which it then gets right. The 50-pass edit lets it finish on hard ones too, after longer reasoning, so its
accuracy on closed answers is lower but far more answers get closed.

## Finding 2: thinking on vs off, all four models

We ran our evaluation in three settings, each with 4 samples per problem on MATH-500 and 2 on GSM8K:

- **On:** the authors' `think35` protocol, with thinking on, temperature 0.6, top-p 0.95, top-k 20, presence penalty 1.5
  and a 32k cap.
- **Off (same sampling):** thinking off, with the same sampling and 32k cap (`think35off`, added in `ws/evaluate.py`).
- **Off (card):** the model card's non-thinking setting as encoded in the authors' `nothink35`: thinking off,
  temperature 0.7, top-p 0.8, top-k 20, presence penalty 1.5 and an 8,192-token cap.

| Model | MATH-500 on | MATH-500 off (same sampling) | MATH-500 off (card) | GSM8K on | GSM8K off (same sampling) | GSM8K off (card) |
|---|---|---|---|---|---|---|
| Unedited | 8.0 | 50.6 | 43.6 | 13.0 | 56.2 | 57.2 |
| p10c | 13.5 | 49.0 | 40.5 | 25.5 | 53.9 | 55.3 |
| p25c | 14.2 | 49.5 | 44.3 | 25.9 | 55.2 | 56.3 |
| p50c | 48.9 | 50.5 | 43.7 | 68.1 | 55.8 | 55.1 |

Share of answers cut at the token cap on MATH-500:

| Model | On | Off (same sampling) | Off (card, 8k cap) |
|---|---|---|---|
| Unedited | 92.7% | 9.3% | 32.3% |
| p50c | 32.4% | 0.2% | 31.2% |

Standard errors are about ±1.1 points on MATH-500 and ±1.0 on GSM8K.

How to read this:

- **The edit only matters in thinking mode.** With thinking off, all four models are within about 2 points of each
  other on both sets, so the edit neither helps nor hurts non-thinking mode.
- **On MATH-500, the edit brings thinking mode up to roughly where non-thinking mode already was** (48.9 vs 50.6). It
  recovers ability the model already had, instead of adding new ability.
- **On GSM8K, edited thinking mode beats every non-thinking setting** (68.1 vs about 56). So on the easier set, once
  the loops are fixed, thinking does add something.
- **The card's non-thinking setting scores lower on MATH-500 than thinking off with the same sampling** (43.6 vs 50.6).
  That comes from the 8,192-token cap: 32% of answers are cut there, so the model also runs long without thinking.

## Bottom line

The experiment shows an LLM finding a specific failure mechanism (a looping direction written by two MLP layers) and
fixing it with a tiny, targeted weight edit. The fix lets thinking mode reach, and on GSM8K exceed, what the same weights
do without thinking. That is a real and reproducible result. "Made the model better at math" is the wrong summary,
though. A fairer one is "repaired the model's broken thinking mode by editing weights". The post should have reported
the thinking-off numbers alongside its table.

## Files

- `results/stage1_2/`: thinking-on evaluation of all four models.
- `results/thinking_off/<model>__think35off/`, `<model>__nothink35/`: thinking-off evaluations.
- `analysis/thinking_stats.py`, `analysis/thinking_stats.json`: thinking-block statistics from the authors' generations.
- Reproduce the thinking-off runs with
  `PROTOCOL=think35off scripts/eval_all.sh "base p50c p25c p10c" math500,gsm8k` (and `PROTOCOL=nothink35`).
