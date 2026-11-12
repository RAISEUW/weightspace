# Stage 3: the rebuilt agent harness

The authors released Experiment 8's logs but not the harness (sandbox, Python kernel, forward-pass server). This folder
rebuilds it from what the logs show: the task message (`runs/p50c/message.txt`, used verbatim, with the EXPLAIN EVERY
STEP paragraph from turn 1), the tool descriptions in it, the returned summaries (`requests/*.done.json`), the harness
messages in `turns/*/prompt.txt` (pass results, failures not counted, explanation refusals, content-filter refusals,
repeated-line filter, 10,000-character cut) and the conversation handling (`driver.log`: a new conversation every 40
replies with the task, counters, notebook tail and the last result).

- `driver.py` — the loop: editor call, block parsing, Python kernel, forward passes, handoffs, logs, resumable state.
- `kernel.py` — persistent CPU Python kernel with `NAMES`, `load`, `load_original`, `save_edit`, `drop_edit`,
  `edited`, `tok`; gradients off; `save_edit` refuses the final norm, embedding / output layer and pure scalings.
- `fwd.py` — one forward pass on one GPU: vLLM generation with the evaluation protocol (per-sample seeds fixed by
  problem id and sample index, so both models share them), math-verify grading, activations and teacher-forced reads
  from a HF transformers forward.
- `new_run.py` — makes a run folder (`config.json`, `model` -> base checkpoint, read-only `practice/`, `work/`).
- `prompts/` — the authors' task message, instructions file and notices 001-002 (delivered at turn 1 here).

## Known differences from the authors' harness

1. **Editor.** Claude (Opus 5.5 or Fable 5.2, effort `max`, adaptive thinking) through the Messages API instead of
   `gpt-6-astra` through the Codex CLI. No server-side model fallback, so every reply comes from the named model.
2. **No container.** The machine has no user namespaces or bwrap, so the kernel is a plain subprocess: the sandbox
   paths `/model`, `/practice`, `/work` are rewritten to the run's folders in every message; the network is blocked
   inside the kernel by patching `socket` and dead proxies, not by the OS; no GPU (`CUDA_VISIBLE_DEVICES=""`); 96 GB
   address-space limit and 900 s timeout as in the task text.
3. **Notices.** 001 (pass size) and 002 (tool facts) are part of the first message instead of arriving mid-run.
   003 and 004 were not delivered to the published runs and are not used.
4. **Teacher-forced and activation reads** use HF transformers (bf16, reference PyTorch kernels for the Gated DeltaNet
   layers), generation uses vLLM 0.32; the authors' implementation details here are unknown.
5. **Refusals.** A Claude `refusal` stop reason is handled like the authors' content-filter refusal: the message is
   withdrawn and the editor is told the output could not be delivered.
6. **Spend cap.** Each run stops if its API spend passes `max_usd` (config).
