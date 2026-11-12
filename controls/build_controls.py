"""Controls for the 50-pass edit (p50c): is it the direction, the columns, or any perturbation of that size?

    python controls/build_controls.py <base ckpt dir> <p50c ckpt dir> <out root> [seed]

p50c changes 64 columns of model.layers.{15,19}.mlp.down_proj by (approximately) a rank-one update D = -u (u^T W_S)
(Astra: subtract the loop direction from the 64 output columns per layer that write it most). For each layer we take
D (edited - base, float32) and u = its top left singular vector, then build:
  randdir  : same 64 columns, a random unit direction r instead of u, scaled so ||D_r||_F = ||D||_F
  randcols : Astra's u removed from 64 random other columns of the same matrix, scaled to ||D||_F
Each checkpoint is the base checkpoint with the two tensors replaced (bfloat16). A JSON report records the rank-one
fit of D and the norms.
"""
import json
import os
import shutil
import sys

import torch
from safetensors.torch import load_file, save_file

base_dir, ed_dir, out_root = sys.argv[1:4]
seed = int(sys.argv[4]) if len(sys.argv) > 4 else 0
g = torch.Generator().manual_seed(seed)
B = load_file(os.path.join(base_dir, "model.safetensors"))
E = load_file(os.path.join(ed_dir, "model.safetensors"))
names = [f"model.language_model.layers.{l}.mlp.down_proj.weight" for l in (15, 19)]
report = {"seed": seed, "layers": {}}
variants = {"randdir": {}, "randcols": {}}
for n in names:
    W = B[n].float()
    D = E[n].float() - W
    cols = torch.nonzero(D.abs().sum(0) > 0)[:, 0]
    U, S, _ = torch.linalg.svd(D[:, cols], full_matrices=False)
    u = U[:, 0]
    nD = float(D.norm())
    r = torch.randn(W.shape[0], generator=g)
    r /= r.norm()
    Dr = torch.zeros_like(W)
    Dr[:, cols] = -torch.outer(r, r @ W[:, cols])
    Dr *= nD / Dr.norm()
    variants["randdir"][n] = W + Dr
    taken = set(cols.tolist())
    others = torch.tensor([c for c in range(W.shape[1]) if c not in taken])
    rc = others[torch.randperm(len(others), generator=g)[:len(cols)]]
    Dc = torch.zeros_like(W)
    Dc[:, rc] = -torch.outer(u, u @ W[:, rc])
    Dc *= nD / Dc.norm()
    variants["randcols"][n] = W + Dc
    report["layers"][n] = {"columns": cols.tolist(), "D_frobenius": nD, "W_frobenius": float(W.norm()),
                           "rank1_energy_share": float(S[0] ** 2 / (S ** 2).sum()), "randcols": rc.tolist()}
for v, T in variants.items():
    out = os.path.join(out_root, f"p50c_{v}_s{seed}")
    os.makedirs(out, exist_ok=True)
    for f in os.listdir(base_dir):
        if f != "model.safetensors":
            shutil.copyfile(os.path.join(base_dir, f), os.path.join(out, f))
    M = dict(B)
    for n, t in T.items():
        M[n] = t.to(torch.bfloat16)
    save_file(M, os.path.join(out, "model.safetensors"), {"format": "pt"})
    json.dump({"variant": v, "seed": seed, "from": [base_dir, ed_dir]}, open(os.path.join(out, "edits.json"), "w"), indent=1)
json.dump(report, open(os.path.join(out_root, f"controls_report_s{seed}.json"), "w"), indent=1)
print(json.dumps({n: {k: d[k] for k in ("D_frobenius", "W_frobenius", "rank1_energy_share")}
                  for n, d in report["layers"].items()}, indent=1))
