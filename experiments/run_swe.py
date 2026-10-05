"""Evaluate compressors on real SWE-agent trajectories with the ideal reader.

For every trajectory we sample up to `--prefixes` decision points (just before
an agent turn) whose uncompressed context exceeds the budget, compress the
prefix, and probe (a) every state key written so far and (b) every identifier
the agent's next action uses that occurred earlier in the context.
Outputs aggregated outcome counts to results/swe_counts.csv.
"""
from __future__ import annotations

import argparse
import os
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from corpus import SHORT, load_corpus  # noqa: E402
from skc.compressors import (BM25Select, ObservationMasking, RandomSelect, RecencyWindow,  # noqa: E402
                             StateKeyedCompaction, TruncateObservations)
from skc.evaluate import evaluate_prefix  # noqa: E402
from skc.synthetic import noisy_extractor  # noqa: E402

AGE_BINS = [-1, 2, 5, 10, 20, 40, 10**9]
AGE_LABELS = ["0-2", "3-5", "6-10", "11-20", "21-40", ">40"]
NW_BINS = [0, 1, 2, 4, 10**9]
NW_LABELS = ["1", "2", "3-4", "5+"]


def methods(suite: str):
    base = [RecencyWindow(), ObservationMasking(), TruncateObservations(), BM25Select(),
            RandomSelect(seed=0), StateKeyedCompaction(),
            StateKeyedCompaction(extend_mode="bm25", name="skc+bm25")]
    if suite == "main":
        return base
    if suite == "ablation":
        return [
            StateKeyedCompaction(),
            StateKeyedCompaction(use_ledger=False, name="skc-no-ledger"),
            StateKeyedCompaction(extend=False, name="skc-no-extend"),
            StateKeyedCompaction(recent=2, name="skc-recent2"),
            StateKeyedCompaction(value_cap=150, name="skc-cap150"),
            StateKeyedCompaction(value_cap=600, name="skc-cap600"),
            StateKeyedCompaction(type_weight={"edit": 1, "cmd": 1, "search": 1, "view": 1},
                                 name="skc-uniform-prior"),
            StateKeyedCompaction(tau=1e9, name="skc-no-recency-prior"),
            StateKeyedCompaction(extractor=noisy_extractor(0.9), name="skc-recall0.9"),
            StateKeyedCompaction(extractor=noisy_extractor(0.7), name="skc-recall0.7"),
            StateKeyedCompaction(extractor=noisy_extractor(0.5), name="skc-recall0.5"),
            StateKeyedCompaction(extend_mode="bm25", extractor=noisy_extractor(0.7),
                                 name="skc+bm25-recall0.7"),
            ObservationMasking(keep_obs=3),
            BM25Select(keep_last=4),
        ]
    raise ValueError(suite)


def _bucket(x, bins, labels):
    return labels[int(np.digitize([x], bins[1:-1], right=True)[0])]


def run_traj(args):
    model, name, turns, budgets, n_prefix, suite = args
    ms = methods(suite)
    if suite == "ablation":
        ms[-2].name = "obs_mask-3"
        ms[-1].name = "bm25-last4"
    hist_tok = np.cumsum([t.tokens if t.role != "system" else 0 for t in turns])
    counts = Counter()
    tokens_used = Counter()
    n_ctx = Counter()
    for B in budgets:
        cands = [i for i in range(3, len(turns) - 1)
                 if turns[i].role == "tool" and turns[i + 1].role == "assistant" and hist_tok[i] > B]
        if not cands:
            continue
        pick = sorted(set(np.linspace(0, len(cands) - 1, min(n_prefix, len(cands))).round().astype(int)))
        for j in pick:
            i = cands[j]
            for m in ms:
                rows, used = evaluate_prefix(turns, i, m, B, with_use=True)
                tokens_used[(m.name, B)] += used
                n_ctx[(m.name, B)] += 1
                for r in rows:
                    if r["probe"] == "state":
                        if r["ktype"] == "task":
                            ab, nb = "task", "task"
                        else:
                            ab = _bucket(r["age"] / 2, AGE_BINS, AGE_LABELS)
                            nb = _bucket(r["n_writes"], NW_BINS, NW_LABELS)
                    else:  # use-probe: distance (in steps) to the latest prior occurrence
                        ab, nb = _bucket(r["age"] / 2, AGE_BINS, AGE_LABELS), "-"
                    counts[(model, m.name, B, r["probe"], r["ktype"], ab, nb, r["outcome"])] += 1
    return counts, tokens_used, n_ctx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", default="results")
    ap.add_argument("--suite", default="main")
    ap.add_argument("--budgets", default="4000,8000,16000")
    ap.add_argument("--prefixes", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    budgets = [int(x) for x in a.budgets.split(",")]
    total, tok, nctx = Counter(), Counter(), Counter()
    for short in SHORT.values():
        corpus = load_corpus(a.cache, short)
        items = sorted(corpus.items())
        if a.limit:
            items = items[: a.limit]
        jobs = [(short, n, t, budgets, a.prefixes, a.suite) for n, (t, _) in items]
        with ProcessPoolExecutor(a.workers) as ex:
            for c, tu, nc in ex.map(run_traj, jobs, chunksize=2):
                total.update(c); tok.update(tu); nctx.update(nc)
        print("done", short, flush=True)
    rows = [dict(model=k[0], method=k[1], budget=k[2], probe=k[3], ktype=k[4], age=k[5],
                 n_writes=k[6], outcome=k[7], count=v) for k, v in total.items()]
    os.makedirs(a.out, exist_ok=True)
    pd.DataFrame(rows).to_csv(os.path.join(a.out, f"swe_counts_{a.suite}.csv"), index=False)
    pd.DataFrame([dict(method=k[0], budget=k[1], contexts=nctx[k], mean_tokens=tok[k] / nctx[k])
                  for k in nctx]).to_csv(os.path.join(a.out, f"swe_tokens_{a.suite}.csv"), index=False)


if __name__ == "__main__":
    main()
