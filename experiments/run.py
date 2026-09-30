"""批量实验：对每个实例、每个预算比例、每个算法求解并记录指标。

实例目录名约定为 <组名>_s<种子>，例如 vul_1k_s3（组 vul_1k，种子 3）；同组不同种子用于求均值和标准差。
预算取 lambda（切断全部可切断入口—目标对的最小割成本，观察 1）的若干比例。

用法示例：
  python -m experiments.run --data "data/vul_1k_s*" --algs A,B2,C,D --out results/raw_1k.csv
  python -m experiments.run --data "data/vul_10k_s*" --algs A,B2,D --lp-bound --out results/raw_10k.csv
结果逐行追加写入 CSV；加 --resume 可跳过已完成的 (实例, 预算比例, 算法)。
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import re
import time

import numpy as np

from adinterdict.algorithms import greedy, hubcut, ip, mincut
from adinterdict.flow import global_min_cut
from adinterdict.graphio import load_instance
from adinterdict.indexed import IndexedInstance
from adinterdict.lp import lp_relaxation
from adinterdict.reduction import reduce_instance

ALGS = {
    "A": lambda I, red, B, a: greedy.solve(I, B),
    "B2": lambda I, red, B, a: mincut.solve_b2(I, B),
    "C": lambda I, red, B, a: ip.solve(red, B, time_limit=a.ip_time_limit),
    "D": lambda I, red, B, a: hubcut.solve(I, B),
    # 消融实验
    "D-noreduce": lambda I, red, B, a: hubcut.solve(I, B, reduce=False),
    "D-nohub": lambda I, red, B, a: hubcut.solve(I, B, n_hubs=0, use_target_cuts=False),
    "D-noLP": lambda I, red, B, a: hubcut.solve(I, B, use_lp=False),
    "D-noLS": lambda I, red, B, a: hubcut.solve(I, B, use_local_search=False),
    "D-ratio": lambda I, red, B, a: hubcut.solve(I, B, modes=("ratio",)),
}

FIELDS = ["dataset", "group", "seed", "n", "m", "m_deletable", "n_entries", "n_targets",
          "red_n", "red_m", "red_time", "lambda", "budget_frac", "budget", "alg", "time",
          "cost", "risk0", "risk", "risk_ratio", "reach_pair_frac", "mean_dist", "status"]


def parse_name(name):
    m = re.match(r"^(.*)_s(\d+)$", name)
    return (m.group(1), int(m.group(2))) if m else (name, 0)


def done_keys(path):
    if not os.path.exists(path):
        return set()
    with open(path, newline="", encoding="utf-8") as f:
        return {(r["dataset"], r["budget_frac"], r["alg"]) for r in csv.DictReader(f)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", required=True, help="实例目录（可用通配符）")
    ap.add_argument("--algs", default="A,B2,C,D")
    ap.add_argument("--fracs", default="0.05,0.1,0.25,0.5,0.75")
    ap.add_argument("--ip-time-limit", type=float, default=300)
    ap.add_argument("--ip-max-nodes", type=int, default=3000,
                    help="约简后节点数超过该值时跳过 C")
    ap.add_argument("--lp-bound", action="store_true", help="额外记录 LP 下界（alg=LB）")
    ap.add_argument("--out", default="results/raw.csv")
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()

    algs = [x.strip() for x in a.algs.split(",") if x.strip()]
    for x in algs:
        if x not in ALGS:
            raise SystemExit(f"未知算法 {x}，可选：{', '.join(ALGS)}")
    fracs = [float(x) for x in a.fracs.split(",")]
    dirs = sorted({d for pat in a.data for d in glob.glob(pat) if os.path.isdir(d)})
    if not dirs:
        raise SystemExit("没有找到实例目录")
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    skip = done_keys(a.out) if a.resume else set()
    new_file = not os.path.exists(a.out) or not a.resume
    f = open(a.out, "w" if new_file else "a", newline="", encoding="utf-8")
    wr = csv.DictWriter(f, fieldnames=FIELDS)
    if new_file:
        wr.writeheader()

    for d in dirs:
        inst = load_instance(d)
        I = IndexedInstance.from_instance(inst)
        group, seed = parse_name(inst.name)
        t0 = time.perf_counter()
        red, st = reduce_instance(I)
        red_time = time.perf_counter() - t0
        lam, _, _ = global_min_cut(I)
        r0 = I.risk()
        base = dict(dataset=inst.name, group=group, seed=seed, n=I.n, m=I.m,
                    m_deletable=int(I.deletable.sum()), n_entries=len(I.entries),
                    n_targets=len(I.targets), red_n=red.n, red_m=red.m,
                    red_time=round(red_time, 4), risk0=r0)
        base["lambda"] = lam
        print(f"[{inst.name}] n={I.n} m={I.m} -> 约简后 n={red.n} m={red.m}，lambda={lam}，R0={r0:.2f}")
        for frac in fracs:
            B = max(1, int(frac * lam)) if lam and lam != float("inf") else 0
            for alg in algs + (["LB"] if a.lp_bound else []):
                key = (inst.name, str(frac), alg)
                if key in skip:
                    continue
                row = dict(base, budget_frac=frac, budget=B, alg=alg)
                if alg == "C" and red.n > a.ip_max_nodes:
                    continue
                t = time.perf_counter()
                if alg == "LB":
                    lb, _, status = lp_relaxation(red, B)
                    row.update(time=round(time.perf_counter() - t, 4), cost="", risk=lb,
                               risk_ratio=lb / r0 if r0 else 0, reach_pair_frac="",
                               mean_dist="", status=status)
                else:
                    sol = ALGS[alg](I, red, B, a)
                    dt = time.perf_counter() - t
                    assert sol.cost <= B, (alg, sol.cost, B)
                    met = I.metrics(sol.removed)
                    row.update(time=round(dt, 4), cost=sol.cost, risk=met["risk"],
                               risk_ratio=met["risk_ratio"], reach_pair_frac=met["reach_pair_frac"],
                               mean_dist=met["mean_dist"], status=sol.info.get("status", ""))
                wr.writerow(row)
                f.flush()
                print(f"   B={B:<6} ({frac:>4}) {alg:<10} R/R0={row['risk_ratio']:.4f}  "
                      f"time={row['time']:.2f}s {row['status']}")
    f.close()


if __name__ == "__main__":
    main()
