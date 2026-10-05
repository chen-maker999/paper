"""Assemble the final result files used in the paper from the individual runs.

final_lite_*      SWE-bench Lite test set: baselines and SKC-basic from `main`, SKC (final, v2
                  utility features) from `skconly`.
final_verified_*  cross-scaffold set: baselines from `main_verified` (OpenHands Devstral from its
                  re-run with the fixed parser; OpenHands Claude Opus 4.5 dropped because its export
                  has no actions), SKC (final) from `skconly_verified`.
"""
import os
import sys

import pandas as pd

R = sys.argv[1] if len(sys.argv) > 1 else "results"


def load(name):
    return pd.read_csv(os.path.join(R, name))


for kind in ("pertraj", "breakdown"):
    m = load(f"eval_main_{kind}.csv.gz")
    s = load(f"eval_skconly_{kind}.csv.gz")
    lite = pd.concat([m[m.method != "skc"], s])
    lite.to_csv(os.path.join(R, f"final_lite_{kind}.csv.gz"), index=False)

    v = load(f"eval_main_verified_{kind}.csv.gz")
    d = load(f"eval_main_verified_devstral_{kind}.csv.gz")
    sv = load(f"eval_skconly_verified_{kind}.csv.gz")
    v = v[~v.model.isin(["oh-opus45", "oh-devstral"]) & (v.method != "skc")]
    sv = sv[sv.model != "oh-devstral"]
    ver = pd.concat([v, d, sv])
    ver.to_csv(os.path.join(R, f"final_verified_{kind}.csv.gz"), index=False)
    print(kind, "lite", lite.method.nunique(), lite.model.nunique(), "verified", ver.method.nunique(),
          ver.model.nunique(), sorted(ver.model.unique()))

t = pd.concat([load("eval_main_tokens.csv").query("method != 'skc'"), load("eval_skconly_tokens.csv")])
t.to_csv(os.path.join(R, "final_lite_tokens.csv"), index=False)
