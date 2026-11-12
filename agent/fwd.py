"""Forward-pass worker for the rebuilt Experiment 8 harness (one pass, one model, one subprocess on one GPU).

    python agent/fwd.py <run_dir> <request.json> <model: original|edited> <out_prefix>

The authors' harness is not released; this follows the tool description in their turn-1 prompt
(third_party/.../runs/p50c/turns/t001/prompt.txt) and the summaries their tool returned (runs/*/requests/*.done.json).
- generate: vLLM with the evaluation protocol (thinking on, T 0.6, top-p 0.95, top-k 20, presence penalty 1.5);
  the seed of each sample is fixed by (problem id or prompt text, sample index), identical for both models.
- activations (generate): residual stream at the requested layers (0 = embeddings, l = after layer l) at the last
  prompt token and averaged over the answer's first 8,192 tokens, from a teacher-forced HF pass over prompt + answer;
  "mlp": down_proj inputs of model.layers[l] for requested l < 24.
- teacher_forced: HF transformers forward; token_logprobs, topk, hidden_<l>, mlp_act_<l>, attn_<l>.
Writes <out_prefix>.result.json (summary, or {"ok": false, "error": ...}).
"""
import hashlib, json, os, sys, time, traceback
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")  # the parent may have run torch CPU threads (edited checkpoint build): forking then can deadlock

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "ws"))
INSTR = "Please reason step by step, and put your final answer within \\boxed{}."
SAMPLING = dict(temperature=0.6, top_p=0.95, top_k=20, presence_penalty=1.5)
MAXLEN = 32768
ACT_TOKENS = 8192


def seed_of(key, j):
    return int(hashlib.sha256(f"{key}|{j}".encode()).hexdigest()[:8], 16)


def build_ckpt(run, which):
    """original: the base checkpoint; edited: base + /work/edited/*.npy cast to bfloat16 (cached by content hash)."""
    base = json.load(open(os.path.join(run, "config.json")))["base_ckpt"]
    if which == "original":
        return base
    import numpy as np, torch
    from safetensors.torch import load_file, save_file
    ed = os.path.join(run, "work", "edited")
    files = sorted(f for f in os.listdir(ed) if f.endswith(".npy")) if os.path.isdir(ed) else []
    h = hashlib.sha256()
    for f in files:
        h.update(f.encode()); h.update(open(os.path.join(ed, f), "rb").read())
    out = os.path.join(run, "ckpt_cache", h.hexdigest()[:16])
    if os.path.exists(os.path.join(out, "model.safetensors")):
        return out
    os.makedirs(out + ".tmp", exist_ok=True)
    for f in os.listdir(base):
        if f != "model.safetensors":
            os.symlink(os.path.join(base, f), os.path.join(out + ".tmp", f)) if not os.path.exists(os.path.join(out + ".tmp", f)) else None
    T = load_file(os.path.join(base, "model.safetensors"))
    for f in files:
        name = f[:-4]
        a = torch.from_numpy(np.load(os.path.join(ed, f)))
        assert name in T and tuple(a.shape) == tuple(T[name].shape), name
        T[name] = a.to(torch.bfloat16)
    save_file(T, os.path.join(out + ".tmp", "model.safetensors"), {"format": "pt"})
    os.replace(out + ".tmp", out)
    return out


def load_practice(run):
    P = {}
    for f in ("gsm8k_train.jsonl", "math_train.jsonl", "aime.jsonl"):
        for l in open(os.path.join(run, "practice", f)):
            r = json.loads(l); P[r["id"]] = r
    return P


def items_of(req, P):
    """[(key, user_message, practice_row_or_None)]"""
    its = []
    bad = [p for p in req.get("problems", []) if p not in P]
    if bad:
        raise ValueError(f"unknown practice ids: {bad[:5]}")
    for p in req.get("problems", []):
        its.append((p, P[p]["problem"] + "\n" + INSTR, P[p]))
    for u in req.get("prompts", []) or []:
        its.append(("prompt:" + u, u, None))
    if not its:
        raise ValueError("no problems or prompts")
    return its


def grade(text, row, finish):
    import mathqa as Q
    pred, _ = Q.extract(text)
    ok = bool(row is not None and Q.is_correct(pred, row["answer"]))
    closed = "</think>" in text
    if finish == "length":
        cat = "cut_at_max_tokens"
    elif ok:
        cat = None
    elif pred is None:
        cat = "no_answer_finished"
    else:
        cat = "wrong_answer"
    return {"pred": pred if pred is None or len(pred) <= 200 else pred[:200] + "...", "correct": ok,
            "thinking_closed": closed, "failure": cat}


def hf_model(ckpt, eager=False):
    import torch
    from transformers import AutoModelForCausalLM
    kw = {"attn_implementation": "eager"} if eager else {}
    return AutoModelForCausalLM.from_pretrained(ckpt, dtype=torch.bfloat16, **kw).cuda().eval()


def capture(m, ids, layers_h, layers_mlp, want_logits=False, attn_layers=()):
    """one text: hidden states (tuple of 25), mlp down_proj inputs for layers_mlp, final hidden for logits."""
    import torch
    acts, hooks = {}, []
    for l in layers_mlp:
        mod = m.model.layers[l].mlp.down_proj
        hooks.append(mod.register_forward_hook(lambda mod, inp, out, l=l: acts.__setitem__(l, inp[0][0].float())))
    try:
        with torch.no_grad():
            o = m.model(input_ids=ids[None], output_hidden_states=True, output_attentions=bool(attn_layers))
    finally:
        for h in hooks:
            h.remove()
    hs = {l: o.hidden_states[l][0].float() for l in layers_h}
    last = o.hidden_states[-1][0] if want_logits else None
    att = {}
    if attn_layers:
        for l in attn_layers:
            a = o.attentions[(3, 7, 11, 15, 19, 23).index(l)]  # HF returns the 6 full-attention layers only
            assert a is not None, f"layer {l} returned no attention probabilities (not a full-attention layer?)"
            att[l] = a[0].float().cpu().numpy()
    return hs, acts, last, att


def generate(run, req, which, out):
    import numpy as np
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer
    P = load_practice(run)
    its = items_of(req, P)
    n = int(req.get("n", 1))
    mt = int(req.get("max_tokens", MAXLEN))
    assert 1 <= mt <= MAXLEN, "max_tokens must be in 1..32768"
    cap = 256 if req.get("model") == "both" else 512
    if len(its) * n > cap:
        raise ValueError(f"items x n = {len(its) * n} > {cap}")
    ckpt = build_ckpt(run, which)
    tok = AutoTokenizer.from_pretrained(ckpt)
    prompts = [tok.apply_chat_template([{"role": "user", "content": u}], tokenize=False, add_generation_prompt=True,
                                       enable_thinking=True) for _, u, _ in its]
    llm = LLM(model=ckpt, dtype="bfloat16", gpu_memory_utilization=float(os.environ.get("FWD_GPU_MEM", 0.85)),
              max_model_len=MAXLEN + 2048, seed=0, enable_prefix_caching=True)
    flat_p, flat_sp, meta = [], [], []
    for (k, u, r), p in zip(its, prompts):
        for j in range(n):
            flat_p.append(p)
            flat_sp.append(SamplingParams(max_tokens=mt, seed=seed_of(k, j), **SAMPLING))
            meta.append((k, j, r))
    outs = llm.generate(flat_p, flat_sp)
    recs = []
    for (k, j, r), p, o in zip(meta, flat_p, outs):
        s = o.outputs[0]
        rec = {"id": k if r is not None else None, "prompt": None if r is not None else k[7:], "sample": j,
               "source": r["source"] if r else None, "answer": r["answer"] if r else None, "text": s.text,
               "tokens": len(s.token_ids), "finish": s.finish_reason}
        rec.update(grade(s.text, r, s.finish_reason))
        if r is None:
            rec["correct"] = None
        recs.append(rec)
    with open(out + ".jsonl", "w") as f:
        for rec in recs:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    res = {"file": out + ".jsonl", "generations": len(recs), "records": len(recs)}
    act = req.get("activations")
    if act:
        del llm
        import gc, torch
        gc.collect(); torch.cuda.empty_cache()
        layers = [int(l) for l in act.get("layers", [])]
        assert layers and all(0 <= l <= 24 for l in layers), "activation layers must be in 0..24"
        lm = [l for l in layers if l < 24] if act.get("mlp") else []
        m = hf_model(ckpt)
        H_last = np.zeros((len(recs), len(layers), 1024), np.float32); H_mean = np.zeros_like(H_last)
        M_last = np.zeros((len(recs), len(lm), 3584), np.float32); M_mean = np.zeros_like(M_last)
        for i, (rec, p) in enumerate(zip(recs, flat_p)):
            pid = tok(p, add_special_tokens=False)["input_ids"]
            aid = tok(rec["text"], add_special_tokens=False)["input_ids"][:ACT_TOKENS]
            ids = torch.tensor(pid + aid, device="cuda")
            hs, ms, _, _ = capture(m, ids, layers, lm)
            L = len(pid)
            for a, l in enumerate(layers):
                H_last[i, a] = hs[l][L - 1].cpu().numpy()
                H_mean[i, a] = (hs[l][L:] if len(aid) else hs[l][L - 1:L]).mean(0).cpu().numpy()
            for a, l in enumerate(lm):
                M_last[i, a] = ms[l][L - 1].cpu().numpy()
                M_mean[i, a] = (ms[l][L:] if len(aid) else ms[l][L - 1:L]).mean(0).cpu().numpy()
        d = out + "_acts"
        os.makedirs(d, exist_ok=True)
        files = {"hidden_last_prompt.npy": H_last, "hidden_mean_answer.npy": H_mean}
        if lm:
            files.update({"mlp_last_prompt.npy": M_last, "mlp_mean_answer.npy": M_mean})
        for k, v in files.items():
            np.save(os.path.join(d, k), v)
        json.dump({"layers": layers, "mlp_layers": lm, "answer_tokens_averaged": ACT_TOKENS,
                   "samples": [{"row": i, "id": r["id"], "sample": r["sample"], "correct": r["correct"],
                                "finish": r["finish"]} for i, r in enumerate(recs)]},
                  open(os.path.join(d, "index.json"), "w"))
        res["activations"] = {"dir": d + "/", "files": {k: list(v.shape) for k, v in files.items()}}
    return res


def teacher_forced(run, req, which, out):
    import numpy as np, torch
    from transformers import AutoTokenizer
    ckpt = build_ckpt(run, which)
    tok = AutoTokenizer.from_pretrained(ckpt)
    if req.get("texts") is not None:
        texts = list(req["texts"])
    elif req.get("chats") is not None:
        texts = [tok.apply_chat_template(c, tokenize=False, enable_thinking=bool(req.get("enable_thinking", False)))
                 for c in req["chats"]]
    else:
        raise ValueError("give texts or chats")
    if not 1 <= len(texts) <= 1024:
        raise ValueError("1..1024 texts")
    ids = [tok(t, add_special_tokens=False)["input_ids"] for t in texts]
    lens = [len(i) for i in ids]
    if max(lens) > 32768 or sum(lens) > 524288:
        raise ValueError(f"too long: max {max(lens)} tokens (limit 32,768), total {sum(lens)} (limit 524,288)")
    ret = req.get("return", {}) or {}
    hl = [int(l) for l in ret.get("hidden_states", []) or []]
    ml = [int(l) for l in ret.get("mlp_act", []) or []]
    al = [int(l) for l in ret.get("attn_layers", []) or []]
    k = int(ret.get("topk", 0) or 0)
    assert all(0 <= l <= 24 for l in hl) and all(0 <= l < 24 for l in ml) and 0 <= k <= 50
    if al:
        assert all(l in (3, 7, 11, 15, 19, 23) for l in al), "attn_layers: full-attention layers only (3, 7, 11, 15, 19, 23)"
        assert len(texts) <= 2 and max(lens) <= 2048, "attn_layers: at most 2 texts of at most 2,048 tokens"
    N, T = len(texts), max(lens)
    size = N * T * (4 * len(hl) * 1024 + 4 * len(ml) * 3584 + 8 * k + 8) + (N * 8 * T * T * 4 * len(al))
    if size > 6e9:
        raise ValueError(f"result would be {size / 1e9:.1f} GB (> 6 GB per pass): fewer layers, texts or tokens")
    m = hf_model(ckpt, eager=bool(al))
    d = out
    os.makedirs(d, exist_ok=True)
    arr = {"input_ids": np.full((N, T), -1, np.int64)}
    if ret.get("token_logprobs"):
        arr["token_logprobs"] = np.zeros((N, T - 1), np.float32)
    if k:
        arr["topk_logprobs"] = np.zeros((N, T, k), np.float32); arr["topk_ids"] = np.zeros((N, T, k), np.int64)
    for l in hl:
        arr[f"hidden_{l}"] = np.lib.format.open_memmap(os.path.join(d, f"hidden_{l}.npy"), "w+", np.float32, (N, T, 1024))
    for l in ml:
        arr[f"mlp_act_{l}"] = np.lib.format.open_memmap(os.path.join(d, f"mlp_act_{l}.npy"), "w+", np.float32, (N, T, 3584))
    for l in al:
        arr[f"attn_{l}"] = np.zeros((N, 8, T, T), np.float32)
    W = m.lm_head.weight
    for i, x in enumerate(ids):
        t = torch.tensor(x, device="cuda")
        hs, ms, last, att = capture(m, t, hl, ml, want_logits=bool(ret.get("token_logprobs") or k), attn_layers=al)
        L = len(x)
        arr["input_ids"][i, :L] = x
        for l in hl:
            arr[f"hidden_{l}"][i, :L] = hs[l].cpu().numpy()
        for l in ml:
            arr[f"mlp_act_{l}"][i, :L] = ms[l].cpu().numpy()
        for l in al:
            arr[f"attn_{l}"][i, :, :L, :L] = att[l]
        if last is not None:
            with torch.no_grad():
                for s in range(0, L, 2048):
                    lp = torch.log_softmax((last[s:s + 2048] @ W.T).float(), -1)
                    if "token_logprobs" in arr:
                        nxt = t[s + 1:s + 2049]
                        g = lp[:len(nxt)].gather(1, nxt[:, None])[:, 0]
                        arr["token_logprobs"][i, s:s + len(nxt)] = g.cpu().numpy()
                    if k:
                        v, j = lp.topk(k, -1)
                        arr["topk_logprobs"][i, s:s + len(v)] = v.cpu().numpy(); arr["topk_ids"][i, s:s + len(v)] = j.cpu().numpy()
    files = {}
    for name, a in arr.items():
        if isinstance(a, np.memmap):
            a.flush()
        else:
            np.save(os.path.join(d, name + ".npy"), a)
        files[name] = list(a.shape)
    json.dump({"lengths": lens, "padding": "right; input_ids padded with -1, other arrays with 0",
               "token_logprobs": "position t = log p(token t+1 | tokens <= t)", "files": files},
              open(os.path.join(d, "meta.json"), "w"), indent=1)
    return {"dir": d + "/", "lengths": lens, "files": files}


def main():
    run, reqf, which, out = sys.argv[1:5]
    req = json.load(open(reqf))
    t0 = time.time()
    try:
        r = (generate if req.get("type") == "generate" else teacher_forced)(run, req, which, out)
        r = {"ok": True, **r, "s": round(time.time() - t0, 1)}
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        r = {"ok": False, "error": f"{type(e).__name__}: {e}"}
    json.dump(r, open(out + ".result.json", "w"))


if __name__ == "__main__":
    main()
