"""Controlled StateTrack experiments.

S1 budget sweep, S2 horizon sweep, S3 stale-rate vs. number of overwrites
(validates Proposition 3), S4 extractor-recall sweep, S5 ledger capacity
(number of keys vs. budget, priority rules).
"""
from __future__ import annotations

import argparse
import os
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from skc.compressors import (BM25Select, ObservationMasking, RandomSelect, RecencyWindow,  # noqa: E402
                             StateKeyedCompaction, TruncateObservations)
from skc.core import ContextIndex, read_key  # noqa: E402
from skc.evaluate import state_probes  # noqa: E402
from skc.synthetic import StateTrackConfig, generate, noisy_extractor  # noqa: E402

SEEDS = 30


def suite(seed):
    return [RecencyWindow(), ObservationMasking(), TruncateObservations(), BM25Select(),
            RandomSelect(seed=seed), StateKeyedCompaction(),
            StateKeyedCompaction(extend_mode="bm25", name="skc+bm25")]


def checkpoints(turns, n=3):
    tools = [i for i, t in enumerate(turns) if t.role == "tool"]
    return [tools[int(len(tools) * (j + 1) / n) - 1] for j in range(n)]


def probe(turns, end, comp, budget, extra=None):
    units = comp(turns[: end + 1], budget)
    cidx = ContextIndex(units)
    hist_turns = [t for t in turns[2: end + 1]]
    kept = {u.src for u in units if u.kind in ("turn", "trunc")}
    rho = sum(1 for t in hist_turns if t.idx in kept) / max(len(hist_turns), 1)
    out = []
    for key, ws in state_probes(turns, end).items():
        out.append(dict(method=comp.name, budget=budget, family=ws[-1].ktype, n_writes=len(ws),
                        age=(end - ws[-1].turn) / 2, rho=rho, outcome=read_key(cidx, ws), **(extra or {})))
    return out


def job_budget(seed):
    turns = generate(StateTrackConfig(steps=150, seed=seed))
    rows = []
    for B in (1000, 2000, 4000, 8000, 16000):
        for comp in suite(seed):
            for end in checkpoints(turns):
                rows += probe(turns, end, comp, B, dict(seed=seed, exp="S1"))
    return rows


def job_horizon(seed):
    rows = []
    for T in (50, 100, 200, 400, 800):
        turns = generate(StateTrackConfig(steps=T, seed=seed))
        for comp in suite(seed):
            rows += probe(turns, len(turns) - 1, comp, 4000, dict(seed=seed, steps=T, exp="S2"))
    return rows


def job_overwrite(seed):
    cfg = StateTrackConfig(steps=200, n_fast=12, p_fast=0.15, seed=seed)
    turns = generate(cfg)
    rows = []
    for B in (2000, 8000, 32000):
        for comp in (RandomSelect(seed=seed), BM25Select(), RecencyWindow(), StateKeyedCompaction()):
            rows += probe(turns, len(turns) - 1, comp, B, dict(seed=seed, exp="S3"))
    return rows


def job_recall(seed):
    turns = generate(StateTrackConfig(steps=150, seed=seed))
    rows = []
    for r in (1.0, 0.95, 0.9, 0.8, 0.7, 0.5):
        for B in (2000, 8000):
            for comp in (StateKeyedCompaction(extractor=noisy_extractor(r, seed), name="skc"),
                         StateKeyedCompaction(extractor=noisy_extractor(r, seed), extend=False,
                                              recent=0, name="skc-ledger-only"),
                         StateKeyedCompaction(extractor=noisy_extractor(r, seed), extend_mode="bm25",
                                              name="skc+bm25")):
                for end in checkpoints(turns):
                    rows += probe(turns, end, comp, B, dict(seed=seed, recall=r, exp="S4"))
    return rows


def job_capacity(seed):
    rows = []
    for K in (20, 50, 100, 200, 400):
        cfg = StateTrackConfig(steps=200, n_pinned=4, n_slow=int(K * 0.7), n_fast=int(K * 0.3),
                               p_slow=0.03 * 20 / K, p_fast=0.25 * 6 / (K * 0.3), seed=seed)
        turns = generate(cfg)
        comps = [StateKeyedCompaction(),
                 StateKeyedCompaction(tau=1e9, type_weight={"slow": 1, "fast": 1, "pinned": 1},
                                      name="skc-no-priority"),
                 StateKeyedCompaction(tau=0.5, name="skc-recency-only"),
                 RecencyWindow(), ObservationMasking()]
        for comp in comps:
            rows += probe(turns, len(turns) - 1, comp, 2000, dict(seed=seed, n_keys=K, exp="S5"))
    return rows


JOBS = {"S1": job_budget, "S2": job_horizon, "S3": job_overwrite, "S4": job_recall, "S5": job_capacity}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results")
    ap.add_argument("--seeds", type=int, default=SEEDS)
    ap.add_argument("--only", default="S1,S2,S3,S4,S5")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    for name in a.only.split(","):
        rows = []
        with ProcessPoolExecutor(4) as ex:
            for r in ex.map(JOBS[name], range(a.seeds)):
                rows += r
        df = pd.DataFrame(rows)
        df.to_csv(os.path.join(a.out, f"synth_{name}.csv.gz"), index=False)
        print(name, len(df), flush=True)


if __name__ == "__main__":
    main()
