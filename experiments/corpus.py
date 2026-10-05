"""Load SWE-agent trajectories into a cached, parsed corpus."""
from __future__ import annotations

import glob
import os
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from skc.sweagent import load_traj  # noqa: E402

SUBMISSIONS = {
    "20240402_sweagent_gpt4": "GPT-4 (SWE-agent 0.x)",
    "20240402_sweagent_claude3opus": "Claude 3 Opus (SWE-agent 0.x)",
    "20240728_sweagent_gpt4o": "GPT-4o (SWE-agent 0.x)",
    "20250226_sweagent_claude-3-7-sonnet-20250219": "Claude 3.7 Sonnet (SWE-agent 1.x)",
    "20250526_sweagent_claude-4-sonnet-20250514": "Claude 4 Sonnet (SWE-agent 1.x)",
}
SHORT = {
    "20240402_sweagent_gpt4": "gpt4",
    "20240402_sweagent_claude3opus": "claude3opus",
    "20240728_sweagent_gpt4o": "gpt4o",
    "20250226_sweagent_claude-3-7-sonnet-20250219": "claude37sonnet",
    "20250526_sweagent_claude-4-sonnet-20250514": "claude4sonnet",
}


def _load(path):
    try:
        turns, info = load_traj(path)
    except Exception as e:  # noqa: BLE001
        return path, None, repr(e)
    return path, (turns, {"exit_status": info.get("exit_status")}), None


def build_cache(traj_root: str, cache_dir: str, workers: int = 4):
    os.makedirs(cache_dir, exist_ok=True)
    for sub, short in SHORT.items():
        out = os.path.join(cache_dir, f"{short}.pkl")
        if os.path.exists(out):
            continue
        files = sorted(glob.glob(os.path.join(traj_root, sub, "*.traj")))
        corpus, errors = {}, []
        with ProcessPoolExecutor(workers) as ex:
            for path, res, err in ex.map(_load, files, chunksize=4):
                name = os.path.basename(path)[:-5]
                if res is None:
                    errors.append((name, err))
                else:
                    corpus[name] = res
        with open(out, "wb") as f:
            pickle.dump(corpus, f)
        print(f"{short}: {len(corpus)} parsed, {len(errors)} errors", flush=True)


def load_corpus(cache_dir: str, short: str):
    with open(os.path.join(cache_dir, f"{short}.pkl"), "rb") as f:
        return pickle.load(f)


if __name__ == "__main__":
    build_cache(sys.argv[1], sys.argv[2])
