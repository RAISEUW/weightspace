"""Stand-in for the authors' (unreleased) weightenv.lmo.tasks.mcqa. Only loop_signals is needed by mathqa.py at
import time; the evaluation path does not call it. MMLU-Pro/GPQA extraction (extract) is not implemented yet."""


def loop_signals(text):
    raise NotImplementedError("mcqa.loop_signals is not part of the released code")


def extract(text):
    raise NotImplementedError("mcqa.extract (MMLU-Pro answer extraction) is not part of the released code")
