"""汇总 run.py 的原始结果：按 (组, 预算比例, 算法) 计算均值 ± 标准差，并计算与最优/下界的差距。

差距定义（阶段 2 文档 4.3 节）：Gap = (P* - P_alg) / P*，其中 P = R0 - R 为被保护的权重。
  * gap_opt：P* 取 C 的最优解（仅统计 C 状态为 optimal 的实例）；
  * gap_lb ：P* 用 LP 下界换算的上界 R0 - LB 代替，是真实差距的上界（大规模图用）。

用法：python -m experiments.aggregate results/raw_1k.csv results/raw_10k.csv --out results/summary
输出 summary.csv 和 summary.md。
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd


def add_gaps(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["P"] = df["risk0"] - df["risk"]
    key = ["dataset", "budget_frac"]
    opt = df[(df.alg == "C") & (df.status == "optimal")][key + ["P"]].rename(columns={"P": "P_opt"})
    lb = df[df.alg == "LB"][key + ["P"]].rename(columns={"P": "P_ub"})
    df = df.merge(opt, on=key, how="left").merge(lb, on=key, how="left")
    with np.errstate(divide="ignore", invalid="ignore"):
        df["gap_opt"] = np.where(df.P_opt > 0, (df.P_opt - df.P) / df.P_opt,
                                 np.where(df.P_opt == 0, 0.0, np.nan))
        df["gap_lb"] = np.where(df.P_ub > 0, (df.P_ub - df.P) / df.P_ub,
                                np.where(df.P_ub == 0, 0.0, np.nan))
    df.loc[df.alg == "LB", ["gap_opt", "gap_lb"]] = np.nan
    return df


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    df = add_gaps(df)
    df["cost_over_budget"] = pd.to_numeric(df["cost"], errors="coerce") / df["budget"].replace(0, np.nan)
    metrics = ["risk_ratio", "time", "cost_over_budget", "gap_opt", "gap_lb",
               "reach_pair_frac", "mean_dist"]
    for m in metrics:
        df[m] = pd.to_numeric(df[m], errors="coerce")
    g = df.groupby(["group", "budget_frac", "alg"])
    out = g[metrics].agg(["mean", "std"])
    out.columns = [f"{a}_{b}" for a, b in out.columns]
    out["runs"] = g.size()
    out["n_mean"] = g["n"].mean()
    out["red_n_mean"] = g["red_n"].mean()
    return out.reset_index()


def to_markdown(s: pd.DataFrame) -> str:
    def pm(r, m, fmt="{:.3f}"):
        mu, sd = r[f"{m}_mean"], r[f"{m}_std"]
        if pd.isna(mu):
            return "–"
        return fmt.format(mu) + (" ± " + fmt.format(sd) if not pd.isna(sd) else "")

    lines = []
    for group, sub in s.groupby("group"):
        lines.append(f"\n### {group}（平均节点数 {sub.n_mean.iloc[0]:.0f}，约简后 {sub.red_n_mean.iloc[0]:.0f}）\n")
        lines.append("| 预算/λ | 算法 | R/R0 | 与最优差距 | 与下界差距(上界) | 时间 (s) | 运行数 |")
        lines.append("|---|---|---|---|---|---|---|")
        for _, r in sub.sort_values(["budget_frac", "alg"]).iterrows():
            lines.append(f"| {r.budget_frac} | {r.alg} | {pm(r, 'risk_ratio')} | {pm(r, 'gap_opt')} | "
                         f"{pm(r, 'gap_lb')} | {pm(r, 'time', '{:.2f}')} | {int(r.runs)} |")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("raw", nargs="+")
    ap.add_argument("--out", default="results/summary")
    a = ap.parse_args()
    df = pd.concat([pd.read_csv(p) for p in a.raw], ignore_index=True)
    s = summarize(df)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    s.to_csv(a.out + ".csv", index=False)
    with open(a.out + ".md", "w", encoding="utf-8") as f:
        f.write("# 实验结果汇总（均值 ± 标准差，跨种子）\n")
        f.write(to_markdown(s))
    print(f"写出 {a.out}.csv 和 {a.out}.md")


if __name__ == "__main__":
    main()
