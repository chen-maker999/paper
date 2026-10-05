"""Final tables (markdown + LaTeX rows) and figures for the paper.

usage: python experiments/final_report.py [results_dir] [figure_dir]
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import plots as P  # noqa: E402  (style constants and helpers)
from summarize import bootstrap_table, macro, paired_test  # noqa: E402

plt = P.plt
RES = sys.argv[1] if len(sys.argv) > 1 else "results"
FIG = sys.argv[2] if len(sys.argv) > 2 else "paper/figures"
OUT = os.path.join(RES, "final")
os.makedirs(OUT, exist_ok=True)
os.makedirs(FIG, exist_ok=True)

METRICS = [("state", "all", "correct", "State correct"), ("state", "all", "stale", "State stale"),
           ("use", "long", "correct", "Use (long range)"), ("copy", "long", "correct", "Copy (long range)")]
BASELINES = ["window", "obs_mask", "obs_trunc", "bm25", "bm25_chunk", "selfinfo", "random"]
ORDER = ["window", "obs_mask", "obs_trunc", "bm25", "bm25_chunk", "selfinfo", "random", "skc-basic", "skc"]
VERIFIED_LABEL = {"oh-claude4sonnet": "Claude 4 Sonnet (OH)", "oh-gpt5": "GPT-5 (OH)", "oh-kimik2": "Kimi K2 (OH)",
                  "oh-qwen3coder": "Qwen3-Coder-480B (OH)", "oh-devstral": "Devstral-Small (OH)",
                  "swe-lm32b": "SWE-agent-LM-32B (SA)", "swe-kimik2": "Kimi K2 (SA)"}


def write(df, name, floatfmt=".1f", index=True):
    df.to_csv(os.path.join(OUT, name + ".csv"), index=index)
    with open(os.path.join(OUT, name + ".md"), "w") as f:
        f.write(df.to_markdown(floatfmt=floatfmt, index=index))


def ci_tables(df, tag):
    """Mean [95% CI] per method/budget for each metric, plus SKC-vs-best-baseline tests."""
    rows, sig = [], []
    for probe, rf, oc, title in METRICS:
        t = bootstrap_table(df, probe, rf, oc, n_boot=2000)
        t["metric"] = title
        rows.append(t)
        for B in sorted(t.budget.unique()):
            tb = t[(t.budget == B) & t.method.isin(BASELINES)]
            best = tb.sort_values("mean", ascending=(oc == "stale")).iloc[0]
            d, lo, hi, p = paired_test(df, probe, "skc", best.method, B, rf, oc)
            sig.append(dict(metric=title, budget=B, skc=t[(t.budget == B) & (t.method == "skc")]["mean"].iloc[0],
                            best_baseline=P.LABEL.get(best.method, best.method), baseline=best["mean"],
                            diff=d, ci_lo=lo, ci_hi=hi, p=p, n_traj=int(best.n_traj)))
    allt = pd.concat(rows)
    allt.to_csv(os.path.join(OUT, f"{tag}_ci.csv"), index=False)
    allt["cell"] = allt.apply(lambda r: f"{r['mean']:.1f} ±{(r.hi - r.lo) / 2:.1f}", axis=1)
    md = allt.pivot_table(index=["metric", "method"], columns="budget", values="cell", aggfunc="first")
    with open(os.path.join(OUT, f"{tag}_ci.md"), "w") as f:
        f.write(md.to_markdown())
    write(pd.DataFrame(sig), f"{tag}_significance", floatfmt=".3g", index=False)
    return allt


def latex_main(allt, tag, budgets=(4000, 8000, 16000, 32000)):
    """LaTeX rows: method & state(4 budgets) & stale & use & copy, best per column in bold."""
    piv = {}
    for title in [m[3] for m in METRICS]:
        piv[title] = allt[allt.metric == title].pivot(index="method", columns="budget", values="mean")
    lines = []
    for m in ORDER:
        if m not in piv["State correct"].index:
            continue
        cells = []
        for title, better in [("State correct", max), ("State stale", min), ("Use (long range)", max),
                              ("Copy (long range)", max)]:
            for B in budgets:
                col = piv[title][B]
                v = col.get(m, np.nan)
                best = better(col.dropna())
                s = f"{v:.1f}"
                cells.append(f"\\textbf{{{s}}}" if abs(v - best) < 0.05 else s)
        name = P.LABEL[m].replace(" (ours)", "")
        if m.startswith("skc"):
            name += " (ours)"
        lines.append(f"{name} & " + " & ".join(cells) + " \\\\")
    with open(os.path.join(OUT, f"{tag}_latex_rows.tex"), "w") as f:
        f.write("\n".join(lines) + "\n")


def fig_budget(allt, name, title_suffix=""):
    fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.15))
    for ax, (probe, rf, oc, title) in zip(axes, METRICS):
        t = allt[allt.metric == title]
        for m in P.METHODS:
            tm = t[t.method == m].sort_values("budget")
            if tm.empty:
                continue
            ax.errorbar(tm.budget, tm["mean"], yerr=[tm["mean"] - tm.lo, tm.hi - tm["mean"]], color=P.COLOR[m],
                        marker=P.MARK[m], ls=P.LS.get(m, "-"), label=P.LABEL[m], capsize=1.5, lw=1.2, ms=3)
        bs = sorted(t.budget.unique())
        ax.set_xscale("log", base=2)
        ax.set_xticks(bs)
        ax.set_xticklabels([f"{b // 1000}k" for b in bs])
        ax.set_xlabel("Budget (tokens)")
        ax.set_title(title + " (%)", loc="left")
    P.shared_legend(fig, axes[0], ncol=5)
    P.save(fig, name)


def breakdowns(bd, tag):
    st = bd[bd.probe == "state"]
    B = 8000
    r = P.rates(st[st.budget == B], ["method", "ktype"])
    fam = (100 * r["correct"]).unstack("ktype").reindex(columns=["task", "edit", "view", "cmd", "search"])
    stale = (100 * r["stale"]).unstack("ktype").reindex(columns=["edit", "view", "cmd", "search"])
    fam = fam.join(stale, rsuffix="_stale").reindex([m for m in ORDER if m in fam.index])
    fam.index = [P.LABEL[m] for m in fam.index]
    write(fam, f"{tag}_by_family")
    r["n"].unstack("ktype").iloc[0].to_csv(os.path.join(OUT, f"{tag}_by_family_counts.csv"))
    labels = P.MODEL_LABEL if tag == "lite" else VERIFIED_LABEL
    for probe in ("state", "use", "copy"):
        x = bd[(bd.probe == probe) & (bd.budget == B)]
        if probe != "state":
            x = x[~x.age.isin(["0-2"])]
        rr = P.rates(x, ["method", "model"])["correct"].unstack("model") * 100
        rr = rr.reindex([m for m in ORDER if m in rr.index])
        rr = rr[[c for c in labels if c in rr.columns]]
        rr.index = [P.LABEL[m] for m in rr.index]
        rr.columns = [labels[c] for c in rr.columns]
        write(rr, f"{tag}_by_model_{probe}")
    return st


def fig_age_and_stale(bd):
    st = bd[bd.probe == "state"]
    B = 8000
    ages = ["0-2", "3-5", "6-10", "11-20", "21-40", ">40"]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.2))
    for m in P.METHODS:
        s0 = st[(st.budget == B) & (st.ktype != "task") & (st.method == m)]
        if s0.empty:
            continue
        kw = dict(color=P.COLOR[m], marker=P.MARK[m], ls=P.LS.get(m, "-"), label=P.LABEL[m], lw=1.2, ms=3)
        axes[0].plot(range(6), 100 * P.rates(s0, ["age"]).reindex(ages).correct, **kw)
        for ax, probe in ((axes[1], "use"), (axes[2], "copy")):
            u = bd[(bd.probe == probe) & (bd.budget == B) & (bd.method == m)]
            ax.plot(range(6), 100 * P.rates(u, ["age"]).reindex(ages).correct, **kw)
    for ax, t, xl in zip(axes, ["State: correct (%)", "Use: available (%)", "Copy: available (%)"],
                         ["age of the key's latest write (steps)", "steps since last seen", "steps since last seen"]):
        ax.set_xticks(range(6))
        ax.set_xticklabels(ages, fontsize=6)
        ax.set_title(t, loc="left")
        ax.set_xlabel(xl)
    P.shared_legend(fig, axes[0], ncol=5)
    P.save(fig, "real_age")
    nws = ["2", "3-4", "5+"]
    fig, ax = plt.subplots(figsize=(3.4, 2.2))
    s1 = st[(st.budget == B) & (st.ktype != "task")]
    for m in P.METHODS:
        x = s1[s1.method == m]
        if x.empty:
            continue
        ax.plot(range(3), 100 * P.rates(x, ["n_writes"]).reindex(nws).stale, color=P.COLOR[m], marker=P.MARK[m],
                ls=P.LS.get(m, "-"), label=P.LABEL[m], lw=1.2, ms=3)
    ax.set_xticks(range(3))
    ax.set_xticklabels(nws)
    ax.set_xlabel("writes to the key so far")
    ax.set_title("Stale (%) at 8k, by number of writes", loc="left")
    ax.legend(frameon=False, fontsize=5.5, ncol=2)
    fig.tight_layout()
    P.save(fig, "real_stale_nwrites")


def compression(tok):
    """Compression ratio at the evaluated decision points (Lite)."""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    p = os.path.join(OUT, "compression_hist.csv")
    if not os.path.exists(p):
        return
    h = pd.read_csv(p, index_col=0)
    t = tok.pivot(index="method", columns="budget", values="mean_tokens")
    ms = tok.pivot(index="method", columns="budget", values="ms_per_call")
    rows = []
    for B in h.index:
        rows.append(dict(budget=B, decision_points=int(h.loc[B, "count"]), history_mean=h.loc[B, "mean"],
                         history_median=h.loc[B, "median"], skc_tokens=t.loc["skc", B],
                         ratio=h.loc[B, "mean"] / t.loc["skc", B], skc_ms=ms.loc["skc", B]))
    write(pd.DataFrame(rows).set_index("budget"), "lite_compression", floatfmt=".2f")
    write(t.reindex([m for m in ORDER if m in t.index]), "lite_tokens_used", floatfmt=".0f")
    write(ms.reindex([m for m in ORDER if m in ms.index]), "lite_latency_ms", floatfmt=".0f")


def ablation():
    df = pd.read_csv(os.path.join(RES, "eval_ablation_v2_pertraj.csv.gz"))
    out = None
    for probe, rf, oc, t in [("state", "all", "correct", "state"), ("state", "all", "stale", "stale"),
                             ("use", "long", "correct", "use"), ("copy", "long", "correct", "copy")]:
        x = bootstrap_table(df, probe, rf, oc, n_boot=200).pivot(index="method", columns="budget", values="mean")
        x.columns = [f"{t}@{c // 1000}k" for c in x.columns]
        out = x if out is None else out.join(x)
    write(out, "ablation")
    # lambda frontier figure (state vs use / copy), with baselines from the main run on the same budgets
    lam = {"lambda=0": 0.0, "lambda=0.1": 0.1, "skc": 0.3, "lambda=1": 1.0, "lambda=3": 3.0}
    main = pd.read_csv(os.path.join(RES, "final_lite_pertraj.csv.gz"))
    fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.2))
    for col, B in enumerate((4000, 8000)):
        for row, (probe, ylab) in enumerate((("use", "Use (long range)"), ("copy", "Copy (long range)"))):
            ax = axes[col * 2 + row]
            xs = [out.loc[k, f"state@{B // 1000}k"] for k in lam]
            ys = [out.loc[k, f"{probe}@{B // 1000}k"] for k in lam]
            ax.plot(xs, ys, color=P.COLOR["skc"], marker="o", lw=1.4, label="SKC, $\\lambda\\in\\{0,..,3\\}$")
            for k, x, y in zip(lam, xs, ys):
                ax.annotate(f"{lam[k]:g}", (x, y), fontsize=5, xytext=(2, 2), textcoords="offset points")
            for m in BASELINES:
                s = macro(main[main.method == m], "state", "all")
                u = macro(main[main.method == m], probe, "long")
                sx = 100 * s[s.budget == B].rate.mean()
                uy = 100 * u[u.budget == B].rate.mean()
                ax.scatter([sx], [uy], color=P.COLOR[m], marker=P.MARK[m], s=14, label=P.LABEL[m], zorder=3)
            ax.set_xlabel("State correct (%)")
            ax.set_title(f"{ylab}, {B // 1000}k", loc="left", fontsize=7)
    P.shared_legend(fig, axes[0], ncol=4)
    P.save(fig, "lambda_frontier")


def llm():
    readers = {"qwen3.8-flash": "Qwen3.8-Flash", "qwen3.7-flash": "Qwen3.7-Flash",
               "deepseek-v4-flash": "DeepSeek-V4-Flash"}
    meth = {"window": "Recency window", "obs_mask": "Observation masking", "bm25": "BM25 (turns)",
            "llm_summary": "LLM summary", "summary_fill": "LLM summary + recent", "skc2": "SKC (ours)"}
    rows, sig = [], []
    rs = np.random.default_rng(0)
    for key, rname in readers.items():
        p = os.path.join(RES, f"llm_{key}", "llm_eval_rows.csv")
        if not os.path.exists(p):
            continue
        d = pd.read_csv(p)
        d = d[d.method.isin(meth)]
        qa = d.dropna(subset=["qa"]).copy()
        for o in ("correct", "stale", "unknown", "wrong"):
            qa[o] = (qa.qa == o).astype(float)
        for m in meth:
            q = qa[qa.method == m]
            x = d[d.method == m]
            rows.append(dict(reader=rname, method=meth[m], qa_correct=100 * q.correct.mean(),
                             qa_stale=100 * q.stale.mean(), qa_unknown=100 * q.unknown.mean(),
                             qa_wrong=100 * q.wrong.mean(), path_hit=100 * x.path_hit.mean(),
                             ident_f1=100 * x.ident_f1.mean(), ctx_tokens=x.ctx_tokens.mean(), n_qa=len(q)))
        for col, frame, val in (("qa_correct", qa, "correct"), ("path_hit", d, "path_hit"), ("ident_f1", d, "ident_f1")):
            pv = frame.pivot_table(index="sample", columns="method", values=val)
            for b in ("window", "obs_mask", "bm25", "llm_summary", "summary_fill"):
                if b not in pv:
                    continue
                diff = (pv["skc2"] - pv[b]).dropna().values
                bs = diff[rs.integers(0, len(diff), (5000, len(diff)))].mean(1)
                sig.append(dict(reader=rname, metric=col, versus=meth[b], diff=100 * diff.mean(),
                                ci_lo=100 * np.percentile(bs, 2.5), ci_hi=100 * np.percentile(bs, 97.5),
                                p=min(1.0, max(2 * min((bs <= 0).mean(), (bs >= 0).mean()), 1 / 5000)), n=len(diff)))
    t = pd.DataFrame(rows)
    write(t, "llm_eval", index=False)
    write(pd.DataFrame(sig), "llm_significance", floatfmt=".3g", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.3))
    cols = {"Recency window": P.COLOR["window"], "Observation masking": P.COLOR["obs_mask"],
            "BM25 (turns)": P.COLOR["bm25"], "LLM summary": "#52514e", "LLM summary + recent": "#9085e9",
            "SKC (ours)": P.COLOR["skc"]}
    rnames = list(t.reader.unique())
    w = 0.16
    for ax, col, title in ((axes[0], "qa_correct", "State QA: correct (%)"), (axes[1], "ident_f1",
                                                                            "Next action: identifier F1")):
        ms = [m for m in cols if m in set(t.method)]
        for j, m in enumerate(ms):
            vals = [t[(t.reader == r) & (t.method == m)][col].iloc[0] for r in rnames]
            ax.bar(np.arange(len(rnames)) + (j - (len(ms) - 1) / 2) * w, vals, width=w * 0.9, color=cols[m], label=m)
        ax.set_xticks(range(len(rnames)))
        ax.set_xticklabels(rnames, fontsize=6.5)
        ax.set_title(title, loc="left")
        ax.grid(axis="x", visible=False)
    P.shared_legend(fig, axes[0], ncol=5)
    P.save(fig, "llm_eval")


def utility_auc():
    a = pd.read_csv(os.path.join(RES, "utility", "auc_hgb2.csv"))
    b = pd.read_csv(os.path.join(RES, "utility", "auc_hgb.csv"))
    b = b[b.scorer == "logreg"]
    a = pd.concat([a, b])
    t = a.groupby("scorer")[["auc", "ap"]].mean().sort_values("auc", ascending=False)
    t["positive_rate"] = a.groupby("scorer")["pos"].mean()
    write(t, "utility_auc", floatfmt=".3f")


def corpus():
    c = pd.read_csv(os.path.join(RES, "corpus_stats.csv"), index_col=0)
    c = c.drop(index=[i for i in c.index if i == "oh-opus45"], errors="ignore")
    cols = ["trajectories", "steps_median", "steps_p90", "tokens_median", "tokens_p90", "tool_share",
            "redundancy_mean", "far_gt10", "keys_median", "overwritten_median"]
    write(c[cols], "corpus", floatfmt=".2f")


if __name__ == "__main__":
    lite = pd.read_csv(os.path.join(RES, "final_lite_pertraj.csv.gz"))
    ver = pd.read_csv(os.path.join(RES, "final_verified_pertraj.csv.gz"))
    a = ci_tables(lite, "lite"); latex_main(a, "lite"); fig_budget(a, "real_budget")
    v = ci_tables(ver, "verified"); latex_main(v, "verified"); fig_budget(v, "verified_budget")
    bl = pd.read_csv(os.path.join(RES, "final_lite_breakdown.csv.gz"))
    bv = pd.read_csv(os.path.join(RES, "final_verified_breakdown.csv.gz"))
    breakdowns(bl, "lite"); breakdowns(bv, "verified"); fig_age_and_stale(bl)
    compression(pd.read_csv(os.path.join(RES, "final_lite_tokens.csv")))
    ablation(); llm(); utility_auc(); corpus()
    print("done")


def corpus_fig():
    c = pd.read_csv(os.path.join(RES, "corpus_stats.csv"), index_col=0).drop(index="oh-opus45", errors="ignore")
    labels = {**P.MODEL_LABEL, **VERIFIED_LABEL}
    order = list(P.MODEL_LABEL) + list(VERIFIED_LABEL)
    c = c.reindex([o for o in order if o in c.index])
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9), gridspec_kw={"width_ratios": [1.15, 1]})
    y = np.arange(len(c))
    left = np.zeros(len(c))
    for col, lab, colr in [("tool_share", "tool outputs", "#2a78d6"), ("agent_share", "agent turns", "#eb6834"),
                           ("task_share", "task", "#1baf7a")]:
        axes[0].barh(y, 100 * c[col], left=left, color=colr, label=lab, height=0.62, edgecolor="white", linewidth=1)
        left += 100 * c[col].values
    axes[0].scatter(100 * c["redundancy_mean"], y, color="#0b0b0b", marker="|", s=60, zorder=4,
                    label="repeated lines (redundancy)")
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([labels[i] for i in c.index], fontsize=6)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("share of context tokens (%)")
    axes[0].legend(frameon=False, ncol=2, loc="lower center", bbox_to_anchor=(0.45, 1.0), fontsize=6)
    axes[0].grid(axis="y", visible=False)
    rf = pd.read_csv(os.path.join(RES, "reference_distance_raw.csv.gz"))
    palette = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
    for i, m in enumerate(["gpt4", "claude3opus", "claude4sonnet", "oh-gpt5", "oh-qwen3coder", "swe-lm32b"]):
        x = np.sort(rf[rf.model == m].far.values)
        if len(x):
            axes[1].plot(x, 1 - np.arange(len(x)) / len(x), color=palette[i], label=labels[m], lw=1.3)
    axes[1].set_xscale("symlog", linthresh=1)
    axes[1].set_xlabel("steps since the identifier first appeared")
    axes[1].set_title("P(distance > x), identifiers used in actions", loc="left", fontsize=7)
    axes[1].legend(frameon=False, fontsize=5.5)
    fig.tight_layout()
    P.save(fig, "corpus")


if __name__ == "__main__":
    corpus_fig()
    P.synthetic()
