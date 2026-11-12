"""Persistent Python kernel for the editor (CPU only, no network). Protocol: the driver writes one JSON line
{"code": ...} to stdin; the kernel replies with one JSON line {"out": ..., "error": bool} on fd 3's twin (a pipe
passed as argv[1]). Code runs in one persistent namespace; if the last statement is an expression, its repr is shown.

    python agent/kernel.py <reply_fd> <run_dir>
"""
import ast, contextlib, io, json, os, socket, sys, traceback

REPLY = os.fdopen(int(sys.argv[1]), "w")
RUN = sys.argv[2]
MODEL, PRACTICE, WORK = (os.path.join(RUN, x) for x in ("model", "practice", "work"))
EDITED = os.path.join(WORK, "edited")


def _no_net(*a, **k):
    raise OSError("no internet access in this kernel")


socket.socket.connect = _no_net
socket.socket.connect_ex = _no_net
socket.create_connection = _no_net

import numpy as np  # noqa: E402
import torch  # noqa: E402
from safetensors import safe_open  # noqa: E402
from tokenizers import Tokenizer  # noqa: E402

torch.set_grad_enabled(False)
torch.set_num_threads(int(os.environ.get("KERNEL_THREADS", 16)))
_ST = os.path.join(MODEL, "model.safetensors")
with safe_open(_ST, "pt") as _f:
    NAMES = sorted(_f.keys())
FORBIDDEN = {"model.language_model.norm.weight", "model.language_model.embed_tokens.weight", "lm_head.weight",
             "model.norm.weight", "model.embed_tokens.weight"}
tok = Tokenizer.from_file(os.path.join(MODEL, "tokenizer.json"))


def load_original(name):
    with safe_open(_ST, "pt") as f:
        return f.get_tensor(name).float()


def load(name):
    p = os.path.join(EDITED, name + ".npy")
    return torch.from_numpy(np.load(p)) if os.path.exists(p) else load_original(name)


def save_edit(name, tensor):
    if name not in NAMES:
        raise KeyError(f"unknown tensor {name}")
    if name in FORBIDDEN or "embed_tokens" in name or name.endswith("lm_head.weight") or name.endswith(".norm.weight") and name.count(".") <= 3:
        raise PermissionError(f"save_edit refused: {name} is out of scope (final norm, embedding / output layer)")
    t = torch.as_tensor(tensor).detach().float().cpu()
    o = load_original(name)
    if tuple(t.shape) != tuple(o.shape):
        raise ValueError(f"shape {tuple(t.shape)} != {tuple(o.shape)}")
    if not torch.isfinite(t).all():
        raise ValueError("tensor has NaN or inf")
    oo = float((o * o).sum())
    if oo > 0:
        c = float((t * o).sum()) / oo
        if float((t - c * o).abs().max()) <= 1e-6 * float(o.abs().max()) and abs(c - 1) > 1e-9:
            raise PermissionError("save_edit refused: the edit only multiplies the whole tensor by a constant")
    os.makedirs(EDITED, exist_ok=True)
    np.save(os.path.join(EDITED, name + ".npy"), t.numpy())
    return f"saved edit of {name}"


def drop_edit(name):
    p = os.path.join(EDITED, name + ".npy")
    if os.path.exists(p):
        os.remove(p)
        return f"dropped edit of {name}"
    return f"{name} was not edited"


def edited():
    return sorted(f[:-4] for f in os.listdir(EDITED) if f.endswith(".npy")) if os.path.isdir(EDITED) else []


NS = {"__name__": "__main__", "NAMES": NAMES, "load": load, "load_original": load_original, "save_edit": save_edit,
      "drop_edit": drop_edit, "edited": edited, "tok": tok, "np": np, "torch": torch}
os.chdir(WORK)


def run(code):
    buf = io.StringIO()
    err = False
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        try:
            tree = ast.parse(code, "<cell>", "exec")
            last = tree.body[-1] if tree.body and isinstance(tree.body[-1], ast.Expr) else None
            if last is not None:
                tree.body = tree.body[:-1]
            exec(compile(tree, "<cell>", "exec"), NS)
            if last is not None:
                v = eval(compile(ast.Expression(last.value), "<cell>", "eval"), NS)
                if v is not None:
                    print(repr(v))
        except BaseException:  # noqa: BLE001
            err = True
            tb = traceback.format_exc().splitlines()
            # drop the kernel's own frames
            keep = [l for i, l in enumerate(tb) if "kernel.py" not in l]
            print("\n".join(keep))
    return buf.getvalue(), err


for line in sys.stdin:
    msg = json.loads(line)
    out, err = run(msg["code"])
    REPLY.write(json.dumps({"out": out, "error": err}) + "\n")
    REPLY.flush()
