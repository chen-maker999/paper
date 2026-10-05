"""Evaluate compressors on real SWE-agent trajectories (state, use and copy probes).

Decision points: up to 6 per trajectory among those whose history exceeds 4k
tokens; a budget B is evaluated on the points whose history exceeds B. The
learned SKC scores items with the utility model trained on the other four LLMs
and on the other half of the SWE-bench instances (see train_utility.py).

Outputs (in --out):
  eval_<suite>_breakdown.csv  outcome counts by model/method/budget/probe/type/age/#writes
  eval_<suite>_pertraj.csv    outcome counts per trajectory (for bootstrap CIs)
  eval_<suite>_tokens.csv     mean context tokens per method and budget
"""
from __future__ import annotations

import os as _os
_os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import os
import pickle
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from common import decision_points, instance_split  # noqa: E402
from corpus import SHORT, VERIFIED, load_corpus  # noqa: E402
from skc.compressors import (BM25Select, ObservationMasking, RandomSelect, RecencyWindow,  # noqa: E402
                             StateKeyedCompaction, TruncateObservations)
from skc.evaluate import evaluate_prefix  # noqa: E402
from skc.synthetic import noisy_extractor  # noqa: E402
from skc.evaluate import copy_probes, use_probes  # noqa: E402
from skc.utility import ChunkBM25, Dedup, OracleUtility, SelfInfoLines, UtilityCompaction  # noqa: E402

AGE_BINS = [-1, 2, 5, 10, 20, 40, 10**9]
AGE_LABELS = ["0-2", "3-5", "6-10", "11-20", "21-40", ">40"]
NW_BINS = [0, 1, 2, 4, 10**9]
NW_LABELS = ["1", "2", "3-4", "5+"]
THRESHOLDS = (0.6, 1.0)  # evidence thresholds for the sensitivity analysis (main: 0.8)
DEV_PER_MODEL = 10  # the first 10 instances (sorted) of every model form the development set
UTIL_DIR = os.environ.get("SKC_UTIL_DIR", "results/utility")
_MODELS = {}


class _LR:
    def __init__(self, lr, mu, sd):
        self.lr, self.mu, self.sd = lr, mu, sd

    def predict_proba(self, X):
        return self.lr.predict_proba((X - self.mu) / self.sd)


def utility_model(kind, model, inst):
    if model not in SHORT.values():  # other scaffolds / benchmarks: model trained on all SWE-agent Lite data
        key = (kind, "all", -1)
        if key not in _MODELS:
            with open(os.path.join(UTIL_DIR, f"{kind}_all.pkl"), "rb") as f:
                _MODELS[key] = pickle.load(f)
        return _MODELS[key]
    key = (kind, model, instance_split(inst))
    if key not in _MODELS:
        with open(os.path.join(UTIL_DIR, f"{kind}_{model}_split{key[2]}.pkl"), "rb") as f:
            obj = pickle.load(f)
        _MODELS[key] = _LR(*obj) if kind == "lr" else obj
    return _MODELS[key]


def methods(suite, model, inst):
    hgb = utility_model("hgb", model, inst)
    if suite == "main":
        return [RecencyWindow(), ObservationMasking(), TruncateObservations(), BM25Select(),
                ChunkBM25(), SelfInfoLines(), RandomSelect(seed=0),
                StateKeyedCompaction(name="skc-basic"), UtilityCompaction(model=hgb, name="skc")]
    if suite == "dev":
        return [RecencyWindow(), ObservationMasking(), BM25Select(), StateKeyedCompaction(name="skc-basic"),
                Dedup(RecencyWindow()), Dedup(ObservationMasking()),
                Dedup(StateKeyedCompaction(name="skc-basic")),
                UtilityCompaction(model=hgb, name="skc"),
                UtilityCompaction(model=hgb, dedup=False, name="skc-nodedup"),
                UtilityCompaction(model=None, name="skc-heur"),
                UtilityCompaction(model=hgb, chunk_lines=10**6, chunk_tokens=10**6, name="skc-turns"),
                UtilityCompaction(model=hgb, lam=1.0, name="skc-lam1")]
    if suite == "frontier":
        hgbc = utility_model("hgbc", model, inst)
        out = [UtilityCompaction(model=hgbc, lam=lam, name=f"skc-count l{lam}") for lam in (0.0, 0.1, 0.3, 1.0, 3.0)]
        out += [UtilityCompaction(model=hgb, lam=lam, name=f"skc-binary l{lam}") for lam in (0.0, 1.0, 3.0)]
        return out
    if suite == "devv2":
        hgb2 = utility_model("hgb2", model, inst)
        return [ChunkBM25(), UtilityCompaction(model=hgb, name="skc"),
                UtilityCompaction(model=hgb2, name="skc-v2"),
                UtilityCompaction(model=hgb2, lam=0.1, name="skc-v2 l0.1"),
                UtilityCompaction(model=hgb2, lam=1.0, name="skc-v2 l1")]
    if suite == "grid":
        out = []
        for cl, ct in ((12, 200), (24, 400), (48, 800)):
            for lam in (0.1, 0.3):
                for R in (2, 4):
                    out.append(UtilityCompaction(model=hgb, chunk_lines=cl, chunk_tokens=ct, lam=lam, recent=R,
                                                 name=f"skc c{cl} l{lam} R{R}"))
        return out + [Dedup(RecencyWindow()), Dedup(ObservationMasking())]
    if suite == "ablation":
        return [
            UtilityCompaction(model=hgb, name="skc"),
            UtilityCompaction(model=hgb, use_ledger=False, name="-ledger"),
            UtilityCompaction(model=hgb, safe=False, name="-safety"),
            UtilityCompaction(model=None, name="-learned (heuristic utility)"),
            UtilityCompaction(model=utility_model("lr", model, inst), name="logreg utility"),
            UtilityCompaction(model=hgb, chunk_lines=10**6, chunk_tokens=10**6, name="-chunking (whole turns)"),
            UtilityCompaction(model=hgb, recent=2, name="recent R=2"),
            UtilityCompaction(model=hgb, lam=0.0, name="lambda=0"),
            UtilityCompaction(model=hgb, lam=0.1, name="lambda=0.1"),
            UtilityCompaction(model=hgb, lam=1.0, name="lambda=1"),
            UtilityCompaction(model=hgb, lam=3.0, name="lambda=3"),
            UtilityCompaction(model=hgb, extractor=noisy_extractor(0.7), name="extractor recall 0.7"),
            UtilityCompaction(model=hgb, extractor=noisy_extractor(0.7), safe=False,
                              name="extractor recall 0.7, -safety"),
            Dedup(RecencyWindow()), Dedup(ObservationMasking()),
            Dedup(StateKeyedCompaction(name="skc-basic")),
            UtilityCompaction(model=hgb, dedup=False, name="-dedup"),
            OracleUtility(name="oracle utility (upper bound)"),
        ]
    raise ValueError(suite)


def _bucket(x, bins, labels):
    return labels[int(np.digitize([x], bins[1:-1], right=True)[0])]


def run_traj(args):
    model, name, turns, budgets, suite = args
    ms = methods(suite, model, name)
    pts, hist_tok = decision_points(turns, n=6, min_tokens=min(budgets))
    br, pt, tok, nctx, lat = Counter(), Counter(), Counter(), Counter(), Counter()
    for B in budgets:
        for i in [p for p in pts if hist_tok[p] > B]:
            for m in ms:
                if isinstance(m, OracleUtility):
                    m.need = ({t for t, _ in use_probes(turns, i, turns[0].text)},
                              {ln for ln, _ in copy_probes(turns, i, turns[0].text)})
                tm = []
                rows, used = evaluate_prefix(turns, i, m, B, with_use=True, timing=tm,
                                             extra_thresholds=THRESHOLDS if suite == "main" else ())
                lat[(m.name, B)] += tm[0]
                tok[(m.name, B)] += used
                nctx[(m.name, B)] += 1
                for r in rows:
                    if r["probe"].startswith("state"):
                        if r["ktype"] == "task":
                            ab, nb = "task", "task"
                        else:
                            ab = _bucket(r["age"] / 2, AGE_BINS, AGE_LABELS)
                            nb = _bucket(r["n_writes"], NW_BINS, NW_LABELS)
                        long = "all"
                    else:
                        ab, nb = _bucket(r["age"] / 2, AGE_BINS, AGE_LABELS), "-"
                        long = "long" if r["age"] / 2 > 2 else "short"
                    br[(model, m.name, B, r["probe"], r["ktype"], ab, nb, r["outcome"])] += 1
                    pt[(model, name, m.name, B, r["probe"], long, r["outcome"])] += 1
    return br, pt, tok, nctx, lat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", default="results")
    ap.add_argument("--suite", default="main")
    ap.add_argument("--budgets", default="4000,8000,16000")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--exclude-dev", action="store_true",
                    help="drop the development instances used for design and hyper-parameter choices")
    ap.add_argument("--per-model", type=int, default=0, help="evaluate at most this many trajectories per model")
    ap.add_argument("--corpora", default="lite", choices=["lite", "verified"])
    a = ap.parse_args()
    dev = set()
    if a.exclude_dev:
        for short in SHORT.values():
            dev |= {n for n, _ in sorted(load_corpus(a.cache, short).items())[:DEV_PER_MODEL]}
        print("excluding", len(dev), "development instances", flush=True)
    budgets = [int(x) for x in a.budgets.split(",")]
    br, pt, tok, nctx, lat = Counter(), Counter(), Counter(), Counter(), Counter()
    corpora = list(SHORT.values()) if a.corpora == "lite" else [v[0] for v in VERIFIED.values()]
    lite_instances = set()
    if a.corpora != "lite":  # cross-scaffold test: drop instances that also occur in SWE-bench Lite
        for short in SHORT.values():
            lite_instances |= set(load_corpus(a.cache, short))
    for short in corpora:
        if not os.path.exists(os.path.join(a.cache, f"{short}.pkl")):
            continue
        corpus = load_corpus(a.cache, short)
        items = [kv for kv in sorted(corpus.items())[: a.limit or None]
                 if kv[0] not in dev and kv[0] not in lite_instances]
        if a.per_model:
            rs = np.random.default_rng(0)
            pick = sorted(rs.choice(len(items), size=min(a.per_model, len(items)), replace=False))
            items = [items[i] for i in pick]
        jobs = [(short, n, t, budgets, a.suite) for n, (t, _) in items]
        with ProcessPoolExecutor(a.workers) as ex:
            for b, p, t, c, la in ex.map(run_traj, jobs, chunksize=1):
                br.update(b); pt.update(p); tok.update(t); nctx.update(c); lat.update(la)
        print("done", short, flush=True)
    os.makedirs(a.out, exist_ok=True)
    tag = a.suite if a.corpora == "lite" else f"{a.suite}_{a.corpora}"
    pd.DataFrame([dict(model=k[0], method=k[1], budget=k[2], probe=k[3], ktype=k[4], age=k[5],
                       n_writes=k[6], outcome=k[7], count=v) for k, v in br.items()]
                 ).to_csv(os.path.join(a.out, f"eval_{tag}_breakdown.csv.gz"), index=False)
    pd.DataFrame([dict(model=k[0], instance=k[1], method=k[2], budget=k[3], probe=k[4], range=k[5],
                       outcome=k[6], count=v) for k, v in pt.items()]
                 ).to_csv(os.path.join(a.out, f"eval_{tag}_pertraj.csv.gz"), index=False)
    tag = a.suite if a.corpora == "lite" else f"{a.suite}_{a.corpora}"
    pd.DataFrame([dict(method=k[0], budget=k[1], contexts=nctx[k], mean_tokens=tok[k] / nctx[k],
                       ms_per_call=1000 * lat[k] / nctx[k])
                  for k in nctx]).to_csv(os.path.join(a.out, f"eval_{tag}_tokens.csv"), index=False)


if __name__ == "__main__":
    main()
