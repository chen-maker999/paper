"""Characterise SWE-agent trajectories: length, token composition, state
dynamics, and how far back the agent reaches for information it uses."""
from __future__ import annotations

import os
import re
import sys
from collections import Counter

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from corpus import SHORT, VERIFIED, load_corpus  # noqa: E402
from skc.core import count_tokens, norm_line  # noqa: E402
from skc.evaluate import _IDENT, use_probes  # noqa: E402


def redundancy(turns):
    """Share of (non-system) tokens in lines that occur again later in the history."""
    seen, dup, tot = set(), 0, 0
    for t in reversed(turns[1:]):
        for ln in t.text.split("\n"):
            n, k = norm_line(ln), count_tokens(ln)
            tot += k
            if len(n) >= 6 and n in seen:
                dup += k
            seen.add(n)
    return dup / max(tot, 1)


def traj_stats(turns):
    hist = [t for t in turns if t.role in ("assistant", "tool")]
    steps = len(hist) // 2
    tok_task = turns[1].tokens
    tok_agent = sum(t.tokens for t in hist if t.role == "assistant")
    tok_tool = sum(t.tokens for t in hist if t.role == "tool")
    writes = [w for t in turns for w in t.writes]
    per_key = Counter(w.key for w in writes)
    ktypes = Counter(w.ktype for w in writes)
    end = len(turns) - 1
    last = {}
    for w in writes:
        last[w.key] = w
    ages = [(end - w.turn) / 2 for w in last.values() if w.ktype != "task"]
    # reference distance for use-probes: steps back to the most recent / first occurrence
    occ = {}
    near, far = [], []
    for i, t in enumerate(turns):
        if i >= 2 and t.role == "assistant":
            for tok, _ in use_probes(turns, i - 1, turns[0].text):
                near.append((i - occ[tok][-1]) / 2)
                far.append((i - occ[tok][0]) / 2)
        if i >= 1:
            for tok in set(_IDENT.findall(t.text)):
                occ.setdefault(tok, []).append(i)
    return dict(steps=steps, tokens=tok_task + tok_agent + tok_tool, tok_task=tok_task,
                redundancy=redundancy(turns),
                tok_agent=tok_agent, tok_tool=tok_tool, n_keys=len(per_key) - 1,
                n_writes=len(writes) - 1, overwritten=sum(1 for k, c in per_key.items() if c > 1 and k != "task"),
                **{f"w_{k}": ktypes.get(k, 0) for k in ("edit", "view", "search", "cmd")},
                age_med=float(np.median(ages)) if ages else np.nan,
                ref_near=near, ref_far=far)


def main(cache_dir, out_dir):
    rows, refs = [], []
    shorts = list(SHORT.values()) + [v[0] for v in VERIFIED.values()]
    for short in shorts:
        if not os.path.exists(os.path.join(cache_dir, f"{short}.pkl")):
            continue
        corpus = load_corpus(cache_dir, short)
        for name, (turns, info) in corpus.items():
            s = traj_stats(turns)
            for a, b in zip(s.pop("ref_near"), s.pop("ref_far")):
                refs.append(dict(model=short, near=a, far=b))
            rows.append(dict(model=short, instance=name, exit=info.get("exit_status"), **s))
    df = pd.DataFrame(rows)
    rf = pd.DataFrame(refs)
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(os.path.join(out_dir, "corpus_stats_raw.csv"), index=False)
    rf.to_csv(os.path.join(out_dir, "reference_distance_raw.csv.gz"), index=False)
    agg = df.groupby("model").agg(
        trajectories=("instance", "count"),
        steps_median=("steps", "median"), steps_p90=("steps", lambda x: x.quantile(.9)),
        tokens_median=("tokens", "median"), tokens_p90=("tokens", lambda x: x.quantile(.9)),
        tokens_max=("tokens", "max"),
        tool_share=("tok_tool", "sum"), agent_share=("tok_agent", "sum"), task_share=("tok_task", "sum"),
        keys_median=("n_keys", "median"), writes_median=("n_writes", "median"),
        overwritten_median=("overwritten", "median"), age_median=("age_med", "median"),
        redundancy_mean=("redundancy", "mean"))
    tot = agg[["tool_share", "agent_share", "task_share"]].sum(axis=1)
    for c in ["tool_share", "agent_share", "task_share"]:
        agg[c] = agg[c] / tot
    r = rf.groupby("model").agg(use_probes=("near", "count"),
                                near_gt10=("near", lambda x: (x > 10).mean()),
                                far_gt10=("far", lambda x: (x > 10).mean()),
                                far_median=("far", "median"))
    agg = agg.join(r)
    agg.to_csv(os.path.join(out_dir, "corpus_stats.csv"))
    with pd.option_context("display.width", 250, "display.max_columns", 50):
        print(agg.round(3))
    wt = df[[c for c in df.columns if c.startswith("w_")]].sum()
    print((wt / wt.sum()).round(3))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
