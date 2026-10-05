"""Train the utility model of SKC with leave-one-model-out, instance-disjoint folds.

For every (held-out model m, instance split s) we train on the trajectories of the
other four models whose instances are in split 1-s, so every test trajectory is
scored by a model that saw neither its LLM nor its SWE-bench instance.
Labels: an item is positive if it contains an identifier or a verbatim line that
the next action uses (use / copy probes).
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from common import decision_points, instance_split  # noqa: E402
from corpus import SHORT, load_corpus  # noqa: E402
from skc.evaluate import copy_probes, use_probes  # noqa: E402
from skc.utility import FEATURES, build_items, feature_matrix, featurize, label_items  # noqa: E402

NEG_RATE = 0.3


def traj_rows(args):
    model, name, turns, n_points = args
    rng = np.random.default_rng(abs(hash((model, name))) % 2**32)
    pts, _ = decision_points(turns, n=n_points)
    X, y, w = [], [], []
    for i in pts:
        pre = turns[: i + 1]
        pinned, hist, tail, items = build_items(pre)
        if not items:
            continue
        featurize(items, pinned, hist)
        ids = {tok for tok, _ in use_probes(turns, i, turns[0].text)}
        lines = {ln for ln, _ in copy_probes(turns, i, turns[0].text)}
        lab = label_items(items, ids, lines)
        F = feature_matrix(items)
        keep = (lab == 1) | (rng.random(len(lab)) < NEG_RATE)
        X.append(F[keep]); y.append(lab[keep])
        w.append(np.where(lab[keep] == 1, 1.0, 1.0 / NEG_RATE))
    if not X:
        return model, name, None
    return model, name, (np.vstack(X), np.concatenate(y), np.concatenate(w))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", default="results/utility")
    ap.add_argument("--points", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    data = {}
    for short in SHORT.values():
        corpus = load_corpus(a.cache, short)
        items = sorted(corpus.items())[: a.limit or None]
        jobs = [(short, n, t, a.points) for n, (t, _) in items]
        with ProcessPoolExecutor(4) as ex:
            for m, n, r in ex.map(traj_rows, jobs, chunksize=2):
                if r is not None:
                    data[(m, n)] = r
        print("features", short, flush=True)
    models = list(SHORT.values())
    report = []
    for m in models:
        for s in (0, 1):
            tr = [k for k in data if k[0] != m and instance_split(k[1]) != s]
            te = [k for k in data if k[0] == m and instance_split(k[1]) == s]
            Xtr = np.vstack([data[k][0] for k in tr]); ytr = np.concatenate([data[k][1] for k in tr])
            wtr = np.concatenate([data[k][2] for k in tr])
            clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08, max_leaf_nodes=31,
                                                 l2_regularization=1.0, random_state=0)
            clf.fit(Xtr, ytr, sample_weight=wtr)
            with open(os.path.join(a.out, f"hgb_{m}_split{s}.pkl"), "wb") as f:
                pickle.dump(clf, f)
            lr = LogisticRegression(max_iter=2000, C=1.0)
            mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
            lr.fit((Xtr - mu) / sd, ytr, sample_weight=wtr)
            with open(os.path.join(a.out, f"lr_{m}_split{s}.pkl"), "wb") as f:
                pickle.dump((lr, mu, sd), f)
            if te:
                Xte = np.vstack([data[k][0] for k in te]); yte = np.concatenate([data[k][1] for k in te])
                wte = np.concatenate([data[k][2] for k in te])
                fi = {f: j for j, f in enumerate(FEATURES)}
                scores = {
                    "hgb": clf.predict_proba(Xte)[:, 1],
                    "logreg": lr.predict_proba((Xte - mu) / sd)[:, 1],
                    "bm25": Xte[:, fi["bm25"]],
                    "recency": -Xte[:, fi["log_age"]],
                    "refcount": Xte[:, fi["log_refcount"]],
                }
                for k, sc in scores.items():
                    report.append(dict(test_model=m, split=s, scorer=k, n=len(yte), pos=float(yte.mean()),
                                       auc=roc_auc_score(yte, sc, sample_weight=wte),
                                       ap=average_precision_score(yte, sc, sample_weight=wte)))
            print("trained", m, s, flush=True)
    rep = pd.DataFrame(report)
    rep.to_csv(os.path.join(a.out, "auc.csv"), index=False)
    print(rep.groupby("scorer")[["auc", "ap"]].mean().round(3))
    hgb_imp = None
    with open(os.path.join(a.out, "features.txt"), "w") as f:
        f.write("\n".join(FEATURES))
    return hgb_imp


if __name__ == "__main__":
    main()
