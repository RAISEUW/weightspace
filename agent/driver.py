"""Driver of the rebuilt Experiment 8 harness: a Claude editor edits Qwen3.5-0.8B's weights with a Python kernel and a
budget of forward passes, following the authors' task message, rules and conversation handling (p50c logs).

    python agent/driver.py <run_dir>              # run_dir/config.json made by agent/new_run.py; resumable

Differences from the authors' harness are listed in agent/README.md.
"""
import json, os, re, select, signal, subprocess, sys, time, traceback

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
PY = sys.executable
RUN = os.path.abspath(sys.argv[1])
CFG = json.load(open(os.path.join(RUN, "config.json")))
BUDGET, MAXRUNS, HANDOFF = CFG["budget"], 300, CFG.get("handoff_every", 40)
WORK = os.path.join(RUN, "work")
PRICE = {"claude-opus-5-5": (4.0, 20.0, 0.20, 5.0), "claude-fable-5-2": (8.0, 40.0, 0.20, 10.0),
         "claude-sonnet-5-5": (2.0, 10.0, 0.10, 2.5)}
EXPLAIN = open(os.path.join(HERE, "prompts", "explain.txt")).read().strip()


def P(*a):
    line = time.strftime("[%Y-%m-%dT%H:%M:%S%z] ") + " ".join(str(x) for x in a)
    print(line, flush=True)
    open(os.path.join(RUN, "driver.log"), "a").write(line + "\n")


def paths(s):
    """the authors' sandbox paths -> this run's folders"""
    return re.sub(r"(?<![\w./])/(model|practice|work)(?=[/\s,.;:)'\"]|$)", lambda m: os.path.join(RUN, m.group(1)), s)


def jload(p, d=None):
    return json.load(open(p)) if os.path.exists(p) else d


def jsave(p, o):
    json.dump(o, open(p + ".tmp", "w"), indent=1)
    os.replace(p + ".tmp", p)


# ---------------------------------------------------------------- kernel
class Kernel:
    def __init__(self):
        self.p = None

    def start(self):
        r, w = os.pipe()
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="", HTTP_PROXY="http://127.0.0.1:9", HTTPS_PROXY="http://127.0.0.1:9",
                   OMP_NUM_THREADS=str(CFG.get("kernel_threads", 16)), KERNEL_THREADS=str(CFG.get("kernel_threads", 16)))
        lim = CFG.get("kernel_mem_gb", 96) * 2**30
        def pre():
            import resource
            resource.setrlimit(resource.RLIMIT_AS, (lim, lim))
            os.setsid()
        self.p = subprocess.Popen([PY, os.path.join(HERE, "kernel.py"), str(w), RUN], stdin=subprocess.PIPE,
                                  stdout=open(os.path.join(RUN, "kernel.log"), "a"), stderr=subprocess.STDOUT,
                                  pass_fds=(w,), env=env, preexec_fn=pre, text=True)
        os.close(w)
        self.r = os.fdopen(r)

    def kill(self):
        if self.p and self.p.poll() is None:
            try:
                os.killpg(self.p.pid, signal.SIGKILL)
            except Exception:  # noqa: BLE001
                pass
        self.p = None

    def run(self, code, timeout=900):
        """(output, restarted_note or None)"""
        if self.p is None or self.p.poll() is not None:
            self.start()
        self.p.stdin.write(json.dumps({"code": code}) + "\n")
        self.p.stdin.flush()
        rd, _, _ = select.select([self.r], [], [], timeout)
        if rd:
            line = self.r.readline()
            if line:
                return json.loads(line)["out"], None
            self.kill()
            return "", "[the kernel died (probably the 96 GB memory limit) and was restarted: its variables are gone]"
        self.kill()
        return "", "[the code ran longer than 900 seconds; the kernel was restarted and its variables are gone]"


def clean_output(s, limit=10000):
    s = s.encode("ascii", "replace").decode("ascii")
    seen, out, hidden = {}, [], 0
    for line in s.split("\n"):
        n = seen.get(line, 0)
        seen[line] = n + 1
        if n >= 2 and line.strip():
            hidden += 1
            continue
        out.append(line)
    s = "\n".join(out)
    if hidden:
        s += f"\n[harness: {hidden} repeated lines not shown (a line is shown at most 2 times per output)]"
    if len(s) > limit:
        s = s[:limit] + "\n[output cut at 10,000 characters]"
    return s


# ---------------------------------------------------------------- forward passes
def summarize(recs):
    g = [r for r in recs if r["correct"] is not None]
    by = {}
    for r in g:
        by.setdefault(r["source"], []).append(r["correct"])
    f = {"wrong_answer": 0, "no_answer_finished": 0, "cut_at_max_tokens": 0}
    for r in g:
        if r["failure"]:
            f[r["failure"]] += 1
    return {"generations": len(recs), "graded": len(g),
            "accuracy": sum(r["correct"] for r in g) / len(g) if g else None,
            "accuracy_by_source": {k: round(sum(v) / len(v), 4) for k, v in sorted(by.items())},
            "mean_tokens": round(sum(r["tokens"] for r in recs) / len(recs)),
            "cut_share": round(sum(r["finish"] == "length" for r in recs) / len(recs), 4),
            "failures": f}


def paired(ro, re_):
    import math
    o = {(r["id"] or r["prompt"], r["sample"]): r for r in ro}
    items = {}
    w2r = r2w = 0
    for r in re_:
        k = (r["id"] or r["prompt"], r["sample"])
        if r["correct"] is None or k not in o:
            continue
        a, b = int(o[k]["correct"]), int(r["correct"])
        w2r += b > a
        r2w += a > b
        items.setdefault(k[0], []).append(b - a)
    d = [sum(v) / len(v) for v in items.values()]
    n = len(d)
    if not n:
        return None
    m = sum(d) / n
    se = math.sqrt(sum((x - m) ** 2 for x in d) / (n - 1) / n) if n > 1 else None
    return {"items": n, "accuracy_difference": round(m, 4), "standard_error": None if se is None else round(se, 4),
            "samples_wrong_to_right": w2r, "samples_right_to_wrong": r2w}


def validate(req):
    t = req.get("type")
    if t not in ("generate", "teacher_forced"):
        raise ValueError('"type" must be "generate" or "teacher_forced"')
    mo = req.get("model")
    if mo not in (("original", "edited", "both") if t == "generate" else ("original", "edited")):
        raise ValueError(f'invalid "model": {mo!r}')
    if t == "generate":
        n = int(req.get("n", 1))
        k = len(req.get("problems", []) or []) + len(req.get("prompts", []) or [])
        cap = 256 if mo == "both" else 512
        if n < 1 or k * n > cap:
            raise ValueError(f"items x n = {k * n}; must be 1..{cap}")


def fwd_once(req, which, out):
    rq = out + ".request.json"
    jsave(rq, req)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(CFG["gpu"]), VLLM_LOGGING_LEVEL="WARNING", PATH=os.path.dirname(PY) + ":" + os.environ.get("PATH", ""))
    with open(os.path.join(RUN, "fwd.log"), "a") as lf:
        lf.write(f"\n==== {out} {which} {time.ctime()}\n")
        lf.flush()
        subprocess.run([PY, os.path.join(HERE, "fwd.py"), RUN, rq, which, out], stdout=lf, stderr=subprocess.STDOUT,
                       env=env, cwd=REPO, timeout=4 * 3600)
    os.remove(rq)
    r = jload(out + ".result.json", {"ok": False, "error": "the forward-pass worker crashed"})
    os.remove(out + ".result.json") if os.path.exists(out + ".result.json") else None
    return r


def forward(req, k):
    validate(req)
    t0 = time.time()
    if req["type"] == "teacher_forced":
        r = fwd_once(req, req["model"], os.path.join(WORK, f"fwd_{k}"))
        if not r["ok"]:
            raise RuntimeError(r["error"])
        r.pop("ok")
        r.pop("s", None)
        return {"ok": True, **r, "s": round(time.time() - t0, 1)}
    if req["model"] in ("original", "edited"):
        out = os.path.join(WORK, f"gen_{k}")
        r = fwd_once(req, req["model"], out)
        if not r["ok"]:
            raise RuntimeError(r["error"])
        recs = [json.loads(l) for l in open(out + ".jsonl")]
        return {**({"activations": r["activations"]} if "activations" in r else {}), "ok": True, "type": "generate",
                "model": req["model"], "file": out + ".jsonl", **summarize(recs), "s": round(time.time() - t0, 1)}
    res = {"ok": True, "type": "generate", "model": "both", "files": {}, "activations": None}
    recs = {}
    for which in ("original", "edited"):
        out = os.path.join(WORK, f"gen_{k}_{which}")
        r = fwd_once(req, which, out)
        if not r["ok"]:
            raise RuntimeError(r["error"])
        res["files"][which] = out + ".jsonl"
        if "activations" in r:
            res["activations"] = res["activations"] or {}
            res["activations"][which] = r["activations"]
        recs[which] = [json.loads(l) for l in open(out + ".jsonl")]
        res[which] = summarize(recs[which])
    res["paired"] = paired(recs["original"], recs["edited"])
    res["s"] = round(time.time() - t0, 1)
    return res


# ---------------------------------------------------------------- the editor
def call_editor(messages, st):
    import anthropic
    client = anthropic.Anthropic(max_retries=0, timeout=3600)
    kw = dict(model=CFG["model"], max_tokens=CFG.get("max_tokens", 64000), system=open(os.path.join(HERE, "prompts", "instructions.md")).read(),
              messages=messages, cache_control={"type": "ephemeral"}, output_config={"effort": CFG.get("effort", "max")})
    if not CFG["model"].startswith("claude-fable"):
        kw["thinking"] = {"type": "adaptive"}
    waits, delay = [], 30
    while True:
        try:
            t0 = time.time()
            with client.messages.stream(**kw) as s:
                msg = s.get_final_message()
            return msg, time.time() - t0, waits
        except (anthropic.RateLimitError, anthropic.InternalServerError, anthropic.APIConnectionError,
                anthropic.APITimeoutError) as e:
            waits.append(f"{type(e).__name__}: {str(e)[:200]}")
            P(f"turn {st['turn']}: API error, retrying in {delay}s: {waits[-1]}")
            if sum(1 for _ in waits) > 200:
                raise
            time.sleep(delay)
            delay = min(delay * 2, 900)
        except anthropic.APIStatusError as e:
            if getattr(e, "status_code", 0) in (529, 503, 502, 504):
                waits.append(f"{e.status_code}")
                time.sleep(delay)
                delay = min(delay * 2, 900)
                continue
            raise


def cost(u):
    pin, pout, pread, pwrite = PRICE.get(CFG["model"], (8.0, 40.0, 0.2, 10.0))
    return (u.get("input_tokens", 0) * pin + u.get("output_tokens", 0) * pout + u.get("cache_read_input_tokens", 0) * pread
            + u.get("cache_creation_input_tokens", 0) * pwrite) / 1e6


BLOCK = re.compile(r"```(python|forward)[^\n]*\n(.*?)```", re.S)


def task_message():
    m = open(os.path.join(HERE, "prompts", "message_p50.txt")).read().strip()
    if BUDGET != 50:      # p10c / p25c differ only in the budget (and the smoke tests use tiny budgets)
        m = m.replace("budget of 50 forward passes", f"budget of {BUDGET} forward passes").replace(
            "all 50 forward passes", f"all {BUDGET} forward passes")
    return paths(m)


def handoff_message(st, last):
    ed = sorted(f[:-4] for f in os.listdir(os.path.join(WORK, "edited")) if f.endswith(".npy")) \
        if os.path.isdir(os.path.join(WORK, "edited")) else []
    notes = open(os.path.join(WORK, "notes.md")).read()[-8000:] if os.path.exists(os.path.join(WORK, "notes.md")) else "(empty)"
    kern = "Your kernel is still running with its variables." if not st.get("kernel_lost") else \
        "Your kernel was restarted, so its variables are gone."
    return (task_message() + f"\n\nTHIS IS A RESTARTED CONVERSATION (to keep it short): {st['turn'] - 1} replies happened "
            f"before. Forward passes used: {st['passes']} of {BUDGET}; Python runs used: {st['runs']} of {MAXRUNS}. {kern} "
            f"Tensors currently edited: {ed}. Files in {WORK}: {sorted(os.listdir(WORK))[:200]}\n\n"
            f"YOUR NOTEBOOK ({WORK}/notes.md, last 8,000 characters):\n```\n{notes}\n```\nContinue from here.\n\nLast result:\n{last}")


def main():
    os.makedirs(os.path.join(RUN, "turns"), exist_ok=True)
    os.makedirs(os.path.join(RUN, "requests"), exist_ok=True)
    stp = os.path.join(RUN, "state.json")
    st = jload(stp) or {"budget": BUDGET, "model": CFG["model"], "effort": CFG.get("effort", "max"), "turn": 1, "runs": 0,
                        "passes": 0, "requests": 0, "done": False, "handoffs": 0, "since_handoff": 0, "nudges": 0,
                        "flags": 0, "flag_streak": 0, "unexplained": 0, "unexplained_streak": 0, "usd": 0.0,
                        "next": None, "new_thread": True}
    convp = os.path.join(RUN, "conversation.json")
    msgs = jload(convp, [])
    K = Kernel()
    if msgs and msgs[-1]["role"] == "user":
        msgs.pop()
    if st["turn"] > 1 and not st["done"]:
        st["kernel_lost"] = True
        st["next"] = "[note: the harness was restarted, so your kernel restarted too and its variables are gone]\n" + (st["next"] or "")
    while not st["done"]:
        if st["usd"] > CFG.get("max_usd", 300):
            P(f"stopping: spend ${st['usd']:.2f} > max_usd"); break
        if st["turn"] > CFG.get("max_turns", 400):
            P("stopping: max_turns"); break
        tdir = os.path.join(RUN, "turns", f"t{st['turn']:03d}")
        os.makedirs(tdir, exist_ok=True)
        if st["new_thread"]:
            prompt = task_message() + "\n\n" + "\n\n".join(paths(open(os.path.join(HERE, "prompts", n)).read().strip())
                                                         for n in CFG.get("notices_at_start", [])) \
                if st["turn"] == 1 else handoff_message(st, st["next"])
            prompt = prompt.strip()
            msgs = []
            st["new_thread"] = False
            st["since_handoff"] = 0
            P(f"turn {st['turn']}: new thread ({len(prompt):,} chars)")
        else:
            prompt = st["next"]
            P(f"turn {st['turn']}: resume ({len(prompt):,} chars)")
        open(os.path.join(tdir, "prompt.txt"), "w").write(prompt)
        msgs.append({"role": "user", "content": prompt})
        msg, wall, waits = call_editor(msgs, st)
        u = msg.usage.model_dump() if msg.usage else {}
        c = cost(u)
        st["usd"] += c
        text = "".join(b.text for b in msg.content if b.type == "text")
        jsave(os.path.join(tdir, "run.json"), {"model": msg.model, "effort": CFG.get("effort", "max"),
                                               "stop_reason": msg.stop_reason, "usage": u, "usd": round(c, 4),
                                               "wall_s": round(wall, 1), "waits": waits,
                                               "thinking_blocks": sum(b.type == "thinking" for b in msg.content)})
        if msg.stop_reason == "refusal":
            st["flags"] += 1
            st["flag_streak"] += 1
            P(f"turn {st['turn']}: refusal ({getattr(msg, 'stop_details', None)}) streak {st['flag_streak']}")
            msgs.pop()
            open(os.path.join(tdir, "answer.md"), "w").write("[refused]")
            st["next"] = open(os.path.join(HERE, "prompts", "refused.txt")).read().strip()
            if st["flag_streak"] >= 3:
                P("three refusals in a row; stopping"); break
            st["turn"] += 1
            jsave(stp, st); jsave(convp, msgs)
            continue
        st["flag_streak"] = 0
        open(os.path.join(tdir, "answer.md"), "w").write(text)
        msgs.append({"role": "assistant", "content": [b.model_dump(exclude_none=True) for b in msg.content]})
        blocks = BLOCK.findall(text)
        expl = BLOCK.sub("", text).strip()
        kind = None
        if len(blocks) > 1:
            st["next"] = (f"Your reply had {len(blocks)} blocks, so none was run (no Python run or forward pass was used). "
                          "Reply with exactly one ```python or ```forward block.")
        elif not blocks:
            if st["passes"] >= BUDGET:
                open(os.path.join(RUN, "final_answer.md"), "w").write(text)
                st["done"] = True
                P(f"final answer: {st['runs']} python runs, {st['passes']} forward passes, {st['nudges']} nudges, "
                  f"{st['handoffs']} handoffs, edited: {sorted(os.listdir(os.path.join(WORK, 'edited'))) if os.path.isdir(os.path.join(WORK, 'edited')) else []}")
            else:
                st["nudges"] += 1
                st["next"] = (f"Your reply had no ```python or ```forward block. The session continues until all {BUDGET} "
                              f"forward passes are used ({BUDGET - st['passes']} left); reply with exactly one block.")
        elif len(expl) < 80 and st["unexplained_streak"] < 2:
            st["unexplained"] += 1
            st["unexplained_streak"] += 1
            st["next"] = ("Your reply had no explanation, so its block was NOT run (no Python run or forward pass was used). "
                          + EXPLAIN + " Send the block again, preceded by your explanation.")
        else:
            st["unexplained_streak"] = 0
            kind, body = blocks[0]
            open(os.path.join(RUN, "explanations.md"), "a").write(f"\n## Turn {st['turn']} ({kind})\n\n{expl}\n")
            rec = {"turn": st["turn"], "kind": kind, "input": body, "t": time.time()}
            if kind == "python":
                if st["runs"] >= MAXRUNS:
                    st["next"] = f"Not run: all {MAXRUNS} Python runs are used."
                else:
                    st["runs"] += 1
                    out, note = K.run(body)
                    st["kernel_lost"] = bool(note)
                    rec["output"] = out
                    st["next"] = (f"OUTPUT of Python run {st['runs']} of at most {MAXRUNS}:\n```\n{clean_output(out)}\n```"
                                  + (f"\n{note}" if note else ""))
            else:
                st["requests"] += 1
                k = st["requests"]
                jsave(stp, st)
                try:
                    req = json.loads(body)
                    jsave(os.path.join(RUN, "requests", f"fwd_{k}.json.claimed"), req)
                    if st["passes"] >= BUDGET:
                        raise RuntimeError(f"all {BUDGET} forward passes are used")
                    res = forward(req, k)
                    st["passes"] += 1
                    jsave(os.path.join(RUN, "requests", f"fwd_{k}.done.json"), res)
                    where = res.get("file") or res.get("dir") or res.get("files")
                    st["next"] = (f"Forward pass {st['passes']} of {BUDGET} done ({req['type']}); results in {where}. "
                                  f"Summary:\n```\n{json.dumps(res, indent=1)}\n```\nForward passes left: {BUDGET - st['passes']}.")
                    rec["result"] = res
                except Exception as e:  # noqa: BLE001
                    err = f"{type(e).__name__}: {e}"
                    rec["error"] = err
                    jsave(os.path.join(RUN, "requests", f"fwd_{k}.failed.json"), {"error": err})
                    st["next"] = f"The forward pass failed and was not counted: {err[:2000]}"
            open(os.path.join(RUN, "tool_calls.jsonl"), "a").write(json.dumps(rec) + "\n")
        if not st["done"] and st["passes"] >= BUDGET and kind == "forward" and "Forward passes left: 0" in st["next"]:
            st["next"] += (f"\nAll {BUDGET} forward passes are used. You can still run Python; when you are done, make your "
                           f"final choice (the tensors in {WORK}/edited are submitted) and reply without a code block.")
        P(f"turn {st['turn']}: {kind or 'no-run'}; runs {st['runs']}, passes {st['passes']}/{BUDGET}, ${st['usd']:.2f}")
        st["turn"] += 1
        st["since_handoff"] += 1
        if st["since_handoff"] >= HANDOFF and not st["done"]:
            st["handoffs"] += 1
            st["new_thread"] = True
        jsave(stp, st)
        jsave(convp, msgs)
    K.kill()
    jsave(stp, st)


if __name__ == "__main__":
    main()
