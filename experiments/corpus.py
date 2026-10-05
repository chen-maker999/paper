"""Load SWE-agent trajectories into a cached, parsed corpus."""
from __future__ import annotations

import glob
import os
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from skc.openhands import load_openhands  # noqa: E402
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


# SWE-bench Verified submissions used to test generalisation across scaffolds and LLMs
VERIFIED = {
    "20250524_openhands_claude_4_sonnet": ("oh-claude4sonnet", "openhands", "Claude 4 Sonnet (OpenHands)"),
    "20250807_openhands_gpt5": ("oh-gpt5", "openhands", "GPT-5 (OpenHands)"),
    "20250716_openhands_kimi_k2": ("oh-kimik2", "openhands", "Kimi K2 (OpenHands)"),
    "20250805-openhands-Qwen3-Coder-480B-A35B-Instruct": ("oh-qwen3coder", "openhands", "Qwen3-Coder-480B (OpenHands)"),
    "20250520_openhands_devstral_small": ("oh-devstral", "openhands", "Devstral-Small (OpenHands)"),
    # 20251127_openhands_claude-opus-4-5 is excluded: its public export records every tool call as
    # "Action: unknown", so actions (and hence state writes and probes) cannot be recovered.
    "20250511_sweagent_lm_32b": ("swe-lm32b", "sweagent", "SWE-agent-LM-32B (SWE-agent)"),
    "20250804_codesweep_sweagent_kimi_k2_instruct": ("swe-kimik2", "sweagent", "Kimi K2 (SWE-agent)"),
}


def _load(path):
    try:
        turns, info = (load_openhands(path) if path.endswith(".json") else load_traj(path))
    except Exception as e:  # noqa: BLE001
        return path, None, repr(e)
    return path, (turns, {"exit_status": info.get("exit_status")}), None


def build_cache(traj_root: str, cache_dir: str, workers: int = 4, registry=None):
    os.makedirs(cache_dir, exist_ok=True)
    registry = registry or SHORT
    for sub, short in registry.items():
        if isinstance(short, tuple):
            short = short[0]
        out = os.path.join(cache_dir, f"{short}.pkl")
        if os.path.exists(out):
            continue
        files = sorted(glob.glob(os.path.join(traj_root, sub, "*.traj")))
        if registry is not SHORT:
            files += sorted(f for f in glob.glob(os.path.join(traj_root, sub, "*__*.json")))
        if not files:
            print(f"{short}: no trajectories found, skipped", flush=True)
            continue
        corpus, errors = {}, []
        with ProcessPoolExecutor(workers) as ex:
            for path, res, err in ex.map(_load, files, chunksize=4):
                name = os.path.splitext(os.path.basename(path))[0].split("::")[0]
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
    if len(sys.argv) > 3 and sys.argv[3] == "verified":
        build_cache(sys.argv[1], sys.argv[2], registry=VERIFIED)
    else:
        build_cache(sys.argv[1], sys.argv[2])
