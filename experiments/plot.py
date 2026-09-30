"""由 aggregate.py 的 summary.csv 画论文插图（PNG + PDF）。

图 1  risk_vs_budget   ：剩余风险比例 R/R0 随预算/λ 的变化，每个数据组一个子图，误差线为标准差。
图 2  gap_vs_budget    ：与最优解（有 C 时）或与 LP 下界（无 C 时）的差距。
图 3  runtime_vs_size  ：运行时间随图规模的变化（对数坐标），取一个固定的预算比例。
图 4  ablation         ：算法 D 各消融版本的 R/R0（只在结果里有消融数据时绘制）。

颜色：每个算法固定一个颜色（跟随算法，不随排名变化），并配固定的标记和线型作为第二编码，
灰度打印也能区分。LP 下界画成灰色虚线参考线，不占用分类颜色。

用法：python -m experiments.plot results/summary.csv --out figures
"""
from __future__ import annotations

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e6e5e1"
REF = "#8a8984"

# 分类色板（已用 dataviz 校验脚本验证：相邻 CVD ΔE >= 9.1，正常视觉 ΔE >= 19.6）。
STYLE = {
    "D":  dict(color="#2a78d6", marker="o", ls="-",  label="D (本文)"),
    "A":  dict(color="#eb6834", marker="s", ls="--", label="A 贪心"),
    "B2": dict(color="#1baf7a", marker="^", ls="-.", label="B2 目标割"),
    "C":  dict(color="#eda100", marker="D", ls=":",  label="C 整数规划"),
}
ABLATION = {
    "D":          dict(color="#2a78d6", marker="o", ls="-",  label="D 完整"),
    "D-noLP":     dict(color="#eb6834", marker="s", ls="--", label="去掉 LP 补救"),
    "D-nohub":    dict(color="#1baf7a", marker="^", ls="-.", label="去掉割动作"),
    "D-noLS":     dict(color="#eda100", marker="D", ls=":",  label="去掉局部搜索"),
    "D-ratio":    dict(color="#e87ba4", marker="v", ls="--", label="只按性价比"),
    "D-noreduce": dict(color="#4a3aa7", marker="P", ls="-.", label="不做约简"),
}


def _setup():
    for fam in ("Noto Sans CJK SC", "WenQuanYi Zen Hei", "SimHei", "Microsoft YaHei",
                "PingFang SC", "Arial Unicode MS"):
        if any(fam == f.name for f in matplotlib.font_manager.fontManager.ttflist):
            plt.rcParams["font.sans-serif"] = [fam, "DejaVu Sans"]
            break
    plt.rcParams.update({
        "axes.unicode_minus": False, "mathtext.fontset": "dejavusans", "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE, "axes.edgecolor": INK2, "axes.labelcolor": INK2,
        "xtick.color": INK2, "ytick.color": INK2, "text.color": INK, "axes.grid": True,
        "grid.color": GRID, "grid.linewidth": 0.8, "axes.spines.top": False,
        "axes.spines.right": False, "font.size": 9, "axes.titlesize": 10,
        "legend.frameon": False, "lines.linewidth": 2, "lines.markersize": 6,
    })


def _save(fig, out, name):
    os.makedirs(out, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out, f"{name}.{ext}"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("写出", os.path.join(out, name + ".png"))


def _grid(n):
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3.4 * cols, 2.8 * rows), squeeze=False)
    return fig, axes.ravel()


def _lines(ax, sub, metric, styles):
    for alg, st in styles.items():
        d = sub[sub.alg == alg].sort_values("budget_frac")
        d = d[d[f"{metric}_mean"].notna()]
        if d.empty:
            continue
        ax.errorbar(d.budget_frac, d[f"{metric}_mean"], yerr=d[f"{metric}_std"].fillna(0),
                    color=st["color"], marker=st["marker"], ls=st["ls"], label=st["label"],
                    capsize=2, elinewidth=1, markeredgecolor=SURFACE, markeredgewidth=1)


def _legend(fig, axes):
    handles, labels = {}, {}
    for ax in axes:
        for h, l in zip(*ax.get_legend_handles_labels()):
            handles.setdefault(l, h)
    if not handles:
        return
    fig.legend(list(handles.values()), list(handles.keys()), loc="lower center",
               ncol=min(5, len(handles)), bbox_to_anchor=(0.5, 1.0))


def risk_vs_budget(s, out, styles, name, title_metric="R / R0（越低越好）"):
    groups = sorted(s.group.unique())
    fig, axes = _grid(len(groups))
    for ax, g in zip(axes, groups):
        sub = s[s.group == g]
        lb = sub[sub.alg == "LB"].sort_values("budget_frac")
        if not lb.empty:
            ax.plot(lb.budget_frac, lb.risk_ratio_mean, color=REF, ls=(0, (2, 2)), lw=1.2,
                    label="LP 下界")
        _lines(ax, sub, "risk_ratio", styles)
        ax.set_title(f"{g}（n≈{sub.n_mean.iloc[0]:.0f}）")
        ax.set_xlabel("预算 / λ")
        ax.set_ylabel(title_metric)
        ax.set_ylim(bottom=0)
    for ax in axes[len(groups):]:
        ax.set_visible(False)
    _legend(fig, axes[:len(groups)])
    fig.tight_layout()
    _save(fig, out, name)


def gap_vs_budget(s, out):
    groups = sorted(s.group.unique())
    fig, axes = _grid(len(groups))
    for ax, g in zip(axes, groups):
        sub = s[(s.group == g) & (s.alg != "LB") & (s.alg != "C")]
        metric = "gap_opt" if sub.gap_opt_mean.notna().any() else "gap_lb"
        _lines(ax, sub, metric, {k: v for k, v in STYLE.items() if k != "C"})
        ax.set_title(g)
        ax.set_xlabel("预算 / λ")
        ax.set_ylabel("与最优解的差距" if metric == "gap_opt" else "与 LP 下界的差距（上界）")
        ax.set_ylim(bottom=0)
    for ax in axes[len(groups):]:
        ax.set_visible(False)
    _legend(fig, axes[:len(groups)])
    fig.tight_layout()
    _save(fig, out, "gap_vs_budget")


def runtime_vs_size(s, out, frac):
    d = s[(s.budget_frac == frac) & (s.alg != "LB")]
    if d.empty:
        return
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    for alg, st in STYLE.items():
        a = d[d.alg == alg].sort_values("n_mean")
        if a.empty:
            continue
        ax.errorbar(a.n_mean, a.time_mean, yerr=a.time_std.fillna(0), color=st["color"],
                    marker=st["marker"], ls=st["ls"], label=st["label"], capsize=2,
                    elinewidth=1, markeredgecolor=SURFACE, markeredgewidth=1)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("节点数 n")
    ax.set_ylabel("运行时间 (s)")
    ax.set_title(f"运行时间（预算 = {frac}·λ）")
    ax.legend(loc="upper left")
    fig.tight_layout()
    _save(fig, out, "runtime_vs_size")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("summary")
    ap.add_argument("--out", default="figures")
    ap.add_argument("--runtime-frac", type=float, default=None,
                    help="运行时间图使用的预算比例（默认取中间值）")
    a = ap.parse_args()
    _setup()
    s = pd.read_csv(a.summary)
    main_s = s[s.alg.isin(list(STYLE) + ["LB"])]
    if main_s.alg.isin(["A", "B2", "C"]).any():
        risk_vs_budget(main_s, a.out, STYLE, "risk_vs_budget")
        gap_vs_budget(main_s, a.out)
    fracs = sorted(s.budget_frac.unique())
    runtime_vs_size(s, a.out, a.runtime_frac if a.runtime_frac is not None else fracs[len(fracs) // 2])
    abl = s[s.alg.isin(ABLATION)]
    if abl.alg.nunique() > 1:
        risk_vs_budget(abl, a.out, ABLATION, "ablation")


if __name__ == "__main__":
    main()
