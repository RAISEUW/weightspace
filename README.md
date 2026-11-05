# weightspace

Replication of Experiment 8 ("Make a language model better at math") from
[Sparks of Weight Space Programming](https://rikiyat.github.io/blog/weight-space-programming/):
an LLM edits Qwen3.5-0.8B's weights directly, with no gradient descent, to improve math accuracy.

Reference logs and code: [RikiyaT/lm_understand-logs](https://github.com/RikiyaT/lm_understand-logs).

## Layout

- `third_party/lm_understand_logs/` — files from the authors' release (commit in `UPSTREAM_COMMIT`): the Experiment 8
  evaluation code, the text-only model config, the three released weight edits (`runs/<run>/edit/`) and their
  reported scores (`eval/<run>/merged.json`). Scripts MIT, logs/data CC BY 4.0 (see its `LICENSE`).
- `ws/` — the evaluation code we run: the authors' `evaluate.py`, `mathqa.py`, `merge.py`, `apply_edit.py`,
  `st_numpy.py` with only import paths changed, a stub for their unreleased `mcqa.py`, and `build_text_ckpt.py`
  (text-only checkpoint from the public release, optionally with an edit applied and hash-checked).
- `scripts/eval_all.sh` — one shard per GPU over several checkpoints, `think35` protocol.

## Reproduce

```bash
WS=/data/billhowe/weightspace            # big files live here
python ws/build_text_ckpt.py $WS/models/Qwen3.5-0.8B $WS/models/base
for r in p10c p25c p50c; do
  python ws/build_text_ckpt.py $WS/models/Qwen3.5-0.8B $WS/models/$r third_party/lm_understand_logs/exp8_math/runs/$r/edit
done
python ws/evaluate.py prepare
scripts/eval_all.sh "base p50c p25c p10c" math500,gsm8k
```

Environment: vLLM 0.32.0+cu129, torch 2.13.0+cu129, math-verify 0.9.0 (the machine's driver supports CUDA 12.8,
so vLLM 0.33's CUDA 13 wheels do not run).

See [DISCUSSION.md](DISCUSSION.md) for the criticisms of the result and our thinking-on/off comparison.
