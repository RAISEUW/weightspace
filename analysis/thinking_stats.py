"""Does an edit switch off thinking? For every first sample (the authors' released generations, one per problem), measure
whether the model closed its thinking block (</think>), how many tokens it thought, and how many it wrote after.

    python analysis/thinking_stats.py <authors' exp8_math/eval dir> <tokenizer dir>
"""
import glob, gzip, json, os, sys

import numpy as np
from transformers import AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ws"))
import mathqa as Q  # noqa: E402

ev, tokdir = sys.argv[1], sys.argv[2]
tok = AutoTokenizer.from_pretrained(tokdir)
out = {}
for run in ("unedited", "p10c", "p25c", "p50c"):
    for s in ("math500", "gsm8k"):
        rows = []
        for f in glob.glob(f"{ev}/{run}/parts/{s}_*/{s}_first_samples.jsonl.gz"):
            rows += [json.loads(l) for l in gzip.open(f, "rt")]
        closed, think, after, empty, ok_closed = [], [], [], 0, []
        for r in rows:
            t = r["text"]
            i = t.find("</think>")
            closed.append(i >= 0)
            th = t if i < 0 else t[:i]
            nt = len(tok(th)["input_ids"])
            think.append(nt)
            na = 0 if i < 0 else len(tok(t[i + 8:])["input_ids"])
            after.append(na)
            empty += i >= 0 and nt < 20
            if i >= 0:
                ok_closed.append(Q.is_correct(Q.extract(t)[0], r["answer"]))
        think, after, closed = np.array(think), np.array(after), np.array(closed)
        out[f"{run}/{s}"] = d = {
            "n": len(rows), "closed_think": float(closed.mean()), "think_tokens_median": float(np.median(think)),
            "think_tokens_median_closed": float(np.median(think[closed])) if closed.any() else None,
            "think_under_20_tokens": empty, "think_under_1000_tokens": int((think < 1000).sum()),
            "answer_tokens_median_closed": float(np.median(after[closed])) if closed.any() else None,
            "acc_when_closed": float(np.mean(ok_closed)) if ok_closed else None}
        print(f"{run:9s} {s:8s} n={d['n']:4d} closed={100*d['closed_think']:5.1f}% think_med={d['think_tokens_median']:7.0f} "
              f"think_med(closed)={d['think_tokens_median_closed'] or 0:7.0f} <20tok={empty:3d} <1000tok={d['think_under_1000_tokens']:4d} "
              f"ans_med={d['answer_tokens_median_closed'] or 0:5.0f} acc|closed={100*(d['acc_when_closed'] or 0):5.1f}", flush=True)
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "thinking_stats.json"), "w"), indent=1)
