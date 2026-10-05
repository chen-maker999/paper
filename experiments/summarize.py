"""Per-trajectory macro averages with bootstrap CIs from eval_<suite>_pertraj.csv.gz."""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd


def macro(df, probe, rng_filter=None, outcome="correct"):
    d = df[df.probe == probe]
    if rng_filter:
        d = d[d.range == rng_filter]
    g = d.groupby(["model", "instance", "method", "budget", "outcome"])["count"].sum().unstack("outcome").fillna(0)
    for c in ("correct", "stale", "missing"):
        if c not in g:
            g[c] = 0.0
    g["rate"] = g[outcome] / g[["correct", "stale", "missing"]].sum(axis=1)
    return g["rate"].reset_index()


def bootstrap_table(df, probe, rng_filter=None, outcome="correct", n_boot=1000, seed=0):
    """Mean over trajectories (each trajectory weight 1) and 95% bootstrap CI.
    Only trajectories that have probes for every method are used (paired)."""
    r = macro(df, probe, rng_filter, outcome)
    out = []
    rs = np.random.default_rng(seed)
    for B, rb in r.groupby("budget"):
        piv = rb.pivot_table(index=["model", "instance"], columns="method", values="rate").dropna()
        n = len(piv)
        idx = rs.integers(0, n, size=(n_boot, n))
        vals = piv.values
        boots = vals[idx].mean(axis=1)
        for j, m in enumerate(piv.columns):
            out.append(dict(budget=B, method=m, mean=100 * vals[:, j].mean(),
                            lo=100 * np.percentile(boots[:, j], 2.5), hi=100 * np.percentile(boots[:, j], 97.5),
                            n_traj=n))
    return pd.DataFrame(out)


def paired_test(df, probe, a, b, budget, rng_filter=None, outcome="correct", n_boot=5000, seed=0):
    """Bootstrap p-value (two-sided) for mean difference a-b over trajectories."""
    r = macro(df, probe, rng_filter, outcome)
    r = r[r.budget == budget].pivot_table(index=["model", "instance"], columns="method", values="rate")
    d = (r[a] - r[b]).dropna().values
    rs = np.random.default_rng(seed)
    boots = d[rs.integers(0, len(d), size=(n_boot, len(d)))].mean(axis=1)
    p = 2 * min((boots <= 0).mean(), (boots >= 0).mean())
    return 100 * d.mean(), 100 * np.percentile(boots, 2.5), 100 * np.percentile(boots, 97.5), max(p, 1 / n_boot)


if __name__ == "__main__":
    df = pd.read_csv(sys.argv[1])
    pd.set_option("display.width", 220)
    for probe, rf, oc in [("state", "all", "correct"), ("state", "all", "stale"), ("use", "long", "correct"),
                          ("copy", "long", "correct")]:
        t = bootstrap_table(df, probe, rf, oc, n_boot=200)
        print(f"== {probe} ({rf}) {oc}")
        print(t.pivot(index="method", columns="budget", values="mean").round(1))
