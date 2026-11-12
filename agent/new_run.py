"""Create a run folder for agent/driver.py.

    python agent/new_run.py <run_dir> --model claude-opus-5-5 --gpu 0 [--effort max] [--budget 50] [--max-usd 300]
"""
import argparse, json, os, shutil

ap = argparse.ArgumentParser()
ap.add_argument("run")
ap.add_argument("--model", required=True)
ap.add_argument("--gpu", required=True)
ap.add_argument("--effort", default="max")
ap.add_argument("--budget", type=int, default=50)
ap.add_argument("--max-usd", type=float, default=300)
ap.add_argument("--base", default="/data/billhowe/weightspace/models/base")
ap.add_argument("--practice", default="/data/billhowe/weightspace/practice35")
a = ap.parse_args()
run = os.path.abspath(a.run)
assert not os.path.exists(os.path.join(run, "config.json")), "run exists"
os.makedirs(os.path.join(run, "work"), exist_ok=True)
os.symlink(a.base, os.path.join(run, "model"))
shutil.copytree(a.practice, os.path.join(run, "practice"))
for f in os.listdir(os.path.join(run, "practice")):
    os.chmod(os.path.join(run, "practice", f), 0o444)
from transformers import AutoTokenizer  # noqa: E402
tok = AutoTokenizer.from_pretrained(a.base)
ex = tok.apply_chat_template([{"role": "user", "content": "<PROBLEM>\nPlease reason step by step, and put your final answer within \\boxed{}."}],
                             tokenize=False, add_generation_prompt=True, enable_thinking=True)
open(os.path.join(run, "work", "eval_prompt_format.txt"), "w").write(
    "The exact evaluation prompt (the model's answer is generated right after it); <PROBLEM> is the problem text:\n" + ex)
json.dump({"model": a.model, "effort": a.effort, "budget": a.budget, "gpu": a.gpu, "base_ckpt": a.base,
           "max_usd": a.max_usd, "handoff_every": 40, "kernel_mem_gb": 96, "kernel_threads": 16, "max_tokens": 64000,
           "notices_at_start": ["001_pass_size.txt", "002_tool_facts.txt"]}, open(os.path.join(run, "config.json"), "w"), indent=1)
print("created", run)
