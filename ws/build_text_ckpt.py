"""Text-only Qwen3.5-0.8B checkpoint, as the authors evaluated it: their config / generation config / tokenizer config /
chat template (third_party/.../base_model/Qwen3.5-0.8B-text, Qwen3_5ForCausalLM) plus the release's tokenizer.json and
its 320 model.language_model.* tensors (names unchanged; the vision tower and the MTP head are dropped).

    python ws/build_text_ckpt.py <release dir (Qwen/Qwen3.5-0.8B @ 2fc0636)> <out dir> [<edit dir> ...]

With edit dirs (a run's edit/: manifest.json + edit.npz), the edited tensors are rebuilt with apply_edit.rebuild and
hash-checked, so the checkpoint is the authors' evaluated model bit for bit.
"""
import glob
import hashlib
import json
import os
import shutil
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import apply_edit as AE  # noqa: E402
import st_numpy as S  # noqa: E402

TEXT_FILES = os.path.join(os.path.dirname(HERE), "third_party/lm_understand_logs/exp8_math/base_model/Qwen3.5-0.8B-text")


def main(release, out, edit_dirs):
    os.makedirs(out, exist_ok=True)
    for f in os.listdir(TEXT_FILES):
        shutil.copyfile(os.path.join(TEXT_FILES, f), os.path.join(out, f))
    shutil.copyfile(os.path.join(release, "tokenizer.json"), os.path.join(out, "tokenizer.json"))
    tensors = {}
    for shard in sorted(glob.glob(os.path.join(release, "*.safetensors"))):
        for name, v in S.load(shard).items():
            if name.startswith("model.language_model."):
                tensors[name] = v
    assert len(tensors) == 320, len(tensors)
    changed = []
    for d in edit_dirs:
        man = json.load(open(os.path.join(d, "manifest.json")))
        z = np.load(os.path.join(d, "edit.npz"))
        for i, t in enumerate(man["tensors"]):
            dt, b = tensors[t["name"]]
            assert hashlib.sha256(np.ascontiguousarray(b).tobytes()).hexdigest() == t["base_sha256"], t["name"]
            e = AE.rebuild(b, t, z, i)
            assert hashlib.sha256(np.ascontiguousarray(e).tobytes()).hexdigest() == t["edited_sha256"], t["name"]
            tensors[t["name"]] = (dt, e)
            changed.append(t["name"])
    S.save(os.path.join(out, "model.safetensors"), tensors, {"format": "pt"})
    json.dump({"release": release, "edits": edit_dirs, "edited": changed}, open(os.path.join(out, "edits.json"), "w"),
              indent=1)
    print(f"{len(tensors)} tensors, {len(changed)} edited (hashes OK) -> {out}")


if __name__ == "__main__":
    main(os.path.abspath(sys.argv[1]), os.path.abspath(sys.argv[2]), [os.path.abspath(d) for d in sys.argv[3:]])
