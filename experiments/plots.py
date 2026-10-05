"""Tables (markdown/LaTeX) and figures from the experiment outputs."""
from __future__ import annotations

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

RES = sys.argv[1] if len(sys.argv) > 1 else "results"
FIG = sys.argv[2] if len(sys.argv) > 2 else "paper/figures"
os.makedirs(FIG, exist_ok=True)

# Fixed method -> colour/marker mapping (validated categorical order; colour follows entity).
METHODS = ["skc", "skc+bm25", "window", "obs_mask", "obs_trunc", "bm25", "random"]
LABEL = {"skc": "SKC (ours)", "skc+bm25": "SKC+BM25 (ours)", "window": "Recency window", "obs_mask": "Observation masking",
         "obs_trunc": "Truncate observations", "bm25": "BM25 selection", "random": "Random selection"}
COLOR = {"skc": "#2a78d6", "skc+bm25": "#008300", "window": "#eb6834", "obs_mask": "#1baf7a", "obs_trunc": "#eda100",
         "bm25": "#e87ba4", "random": "#4a3aa7"}
MARK = {"skc": "o", "skc+bm25": "P", "window": "s", "obs_mask": "^", "obs_trunc": "D", "bm25": "v", "random": "x"}
MODELS = ["gpt4", "claude3opus", "gpt4o", "claude37sonnet", "claude4sonnet"]
MODEL_LABEL = {"gpt4": "GPT-4", "claude3opus": "Claude 3 Opus", "gpt4o": "GPT-4o",
               "claude37sonnet": "Claude 3.7 Sonnet", "claude4sonnet": "Claude 4 Sonnet"}
INK, MUTED = "#0b0b0b", "#52514e"

plt.rcParams.update({"font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
                     "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True,
                     "grid.color": "#e6e5e0", "grid.linewidth": 0.6, "lines.linewidth": 1.6,
                     "lines.markersize": 4, "figure.dpi": 200, "savefig.bbox": "tight",
                     "pdf.fonttype": 42})


def save(fig, name):
    fig.savefig(os.path.join(FIG, name + ".pdf"))
    fig.savefig(os.path.join(FIG, name + ".png"))
    plt.close(fig)


def rates(df, by):
    p = df.groupby(by + ["outcome"])["count"].sum().unstack("outcome").fillna(0)
    for c in ("correct", "stale", "missing"):
        if c not in p:
            p[c] = 0
    n = p[["correct", "stale", "missing"]].sum(axis=1)
    out = p[["correct", "stale", "missing"]].div(n, axis=0)
    out["n"] = n
    return out


def write_table(df, name, floatfmt=".1f"):
    df.to_csv(os.path.join(RES, name + ".csv"))
    with open(os.path.join(RES, name + ".md"), "w") as f:
        f.write(df.to_markdown(floatfmt=floatfmt))


# ---------------------------------------------------------------- real data
def real():
    d = pd.read_csv(os.path.join(RES, "swe_counts_main.csv"))
    st = d[d.probe == "state"]
    use = d[d.probe == "use"]

    # Table: per model x method at each budget, correct% and stale% (state), use%, long-range use%
    rows = []
    for B in sorted(st.budget.unique()):
        r = rates(st[st.budget == B], ["model", "method"])
        u = rates(use[use.budget == B], ["model", "method"])
        ul = rates(use[(use.budget == B) & (~use.age.isin(["0-2"]))], ["model", "method"])
        for (m, meth), row in r.iterrows():
            rows.append(dict(budget=B, model=m, method=meth, state_correct=100 * row.correct,
                             state_stale=100 * row.stale, state_n=row.n,
                             use_avail=100 * u.loc[(m, meth), "correct"],
                             use_long_avail=100 * ul.loc[(m, meth), "correct"],
                             use_n=u.loc[(m, meth), "n"]))
    tab = pd.DataFrame(rows)
    tab.to_csv(os.path.join(RES, "table_main_long.csv"), index=False)

    # pooled over models (micro) and macro over models
    pooled = []
    for B in sorted(st.budget.unique()):
        r = rates(st[st.budget == B], ["method"])
        u = rates(use[use.budget == B], ["method"])
        ul = rates(use[(use.budget == B) & (~use.age.isin(["0-2"]))], ["method"])
        mac = tab[tab.budget == B].groupby("method")[["state_correct", "state_stale"]].mean()
        for meth in METHODS:
            pooled.append(dict(budget=B, method=LABEL[meth], state_correct=100 * r.loc[meth, "correct"],
                               state_stale=100 * r.loc[meth, "stale"],
                               state_correct_macro=mac.loc[meth, "state_correct"],
                               use_avail=100 * u.loc[meth, "correct"],
                               use_long_avail=100 * ul.loc[meth, "correct"]))
    write_table(pd.DataFrame(pooled).set_index(["budget", "method"]), "table_main")

    # Table: per key family at 8k
    B = 8000 if 8000 in st.budget.unique() else st.budget.max()
    r = rates(st[st.budget == B], ["method", "ktype"])
    fam = (100 * r["correct"]).unstack("ktype")[["task", "edit", "view", "cmd", "search"]]
    stale = (100 * r["stale"]).unstack("ktype")[["edit", "view", "cmd", "search"]]
    fam = fam.join(stale, rsuffix="_stale").loc[METHODS]
    fam.index = [LABEL[m] for m in fam.index]
    write_table(fam, "table_by_family")
    nfam = r["n"].unstack("ktype").iloc[0]
    nfam.to_csv(os.path.join(RES, "table_by_family_counts.csv"))

    # Table: per model at 8k
    t8 = tab[tab.budget == B].pivot(index="method", columns="model", values="state_correct")[MODELS].loc[METHODS]
    t8.index = [LABEL[m] for m in t8.index]
    t8.columns = [MODEL_LABEL[c] for c in t8.columns]
    write_table(t8, "table_by_model")
    u8 = tab[tab.budget == B].pivot(index="method", columns="model", values="use_long_avail")[MODELS].loc[METHODS]
    u8.index = [LABEL[m] for m in u8.index]
    u8.columns = [MODEL_LABEL[c] for c in u8.columns]
    write_table(u8, "table_by_model_use_long")

    # Figure: correct & stale vs budget (pooled)
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.2))
    budgets = sorted(st.budget.unique())
    for meth in METHODS:
        r = [rates(st[(st.budget == b) & (st.method == meth)], ["method"]).iloc[0] for b in budgets]
        u = [rates(use[(use.budget == b) & (use.method == meth) & (~use.age.isin(["0-2"]))], ["method"]).iloc[0]
             for b in budgets]
        kw = dict(color=COLOR[meth], marker=MARK[meth], label=LABEL[meth])
        axes[0].plot(budgets, [100 * x.correct for x in r], **kw)
        axes[1].plot(budgets, [100 * x.stale for x in r], **kw)
        axes[2].plot(budgets, [100 * x.correct for x in u], **kw)
    for ax, t in zip(axes, ["State probes: correct (%)", "State probes: stale (%)",
                            "Use probes (>2 steps back): available (%)"]):
        ax.set_xscale("log", base=2)
        ax.set_xticks(budgets)
        ax.set_xticklabels([f"{b // 1000}k" for b in budgets])
        ax.set_xlabel("Budget (tokens)")
        ax.set_title(t, loc="left")
    axes[0].legend(frameon=False, loc="lower right", ncol=1)
    fig.tight_layout()
    save(fig, "real_budget")

    # Figure: correct by age bucket at B
    ages = ["0-2", "3-5", "6-10", "11-20", "21-40", ">40"]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.3))
    s = st[(st.budget == B) & (st.ktype != "task")]
    for meth in METHODS:
        r = rates(s[s.method == meth], ["age"]).reindex(ages)
        axes[0].plot(range(len(ages)), 100 * r.correct, color=COLOR[meth], marker=MARK[meth], label=LABEL[meth])
        u = rates(use[(use.budget == B) & (use.method == meth)], ["age"]).reindex(ages)
        axes[1].plot(range(len(ages)), 100 * u.correct, color=COLOR[meth], marker=MARK[meth], label=LABEL[meth])
    for ax, t in zip(axes, [f"State probes at {B // 1000}k: correct (%)", f"Use probes at {B // 1000}k: available (%)"]):
        ax.set_xticks(range(len(ages)))
        ax.set_xticklabels(ages)
        ax.set_title(t, loc="left")
    axes[0].set_xlabel("Age of the key's latest write (agent steps)")
    axes[1].set_xlabel("Steps since the identifier last appeared")
    axes[0].legend(frameon=False, loc="lower left")
    fig.tight_layout()
    save(fig, "real_age")

    # Figure: stale rate vs number of writes (real data, at B)
    nws = ["2", "3-4", "5+"]
    fig, ax = plt.subplots(figsize=(3.4, 2.2))
    s = st[(st.budget == B) & (st.ktype != "task")]
    for meth in METHODS:
        r = rates(s[s.method == meth], ["n_writes"]).reindex(nws)
        ax.plot(range(len(nws)), 100 * r.stale, color=COLOR[meth], marker=MARK[meth], label=LABEL[meth])
    ax.set_xticks(range(len(nws)))
    ax.set_xticklabels(nws)
    ax.set_xlabel("Writes to the key so far")
    ax.set_title(f"Real trajectories, {B // 1000}k: stale (%)", loc="left")
    ax.legend(frameon=False, fontsize=6)
    fig.tight_layout()
    save(fig, "real_stale_nwrites")


def real_ablation():
    p = os.path.join(RES, "swe_counts_ablation.csv")
    if not os.path.exists(p):
        return
    d = pd.read_csv(p)
    rows = []
    for B in sorted(d.budget.unique()):
        st = d[(d.probe == "state") & (d.budget == B)]
        us = d[(d.probe == "use") & (d.budget == B) & (~d.age.isin(["0-2"]))]
        r, u = rates(st, ["method"]), rates(us, ["method"])
        for m in r.index:
            rows.append(dict(budget=B, variant=m, state_correct=100 * r.loc[m, "correct"],
                             state_stale=100 * r.loc[m, "stale"], use_long_avail=100 * u.loc[m, "correct"]))
    write_table(pd.DataFrame(rows).set_index(["budget", "variant"]), "table_ablation")


# ---------------------------------------------------------------- synthetic
def srates(df, by):
    g = df.groupby(by)["outcome"].value_counts(normalize=True).unstack().fillna(0)
    for c in ("correct", "stale", "missing"):
        if c not in g:
            g[c] = 0.0
    return g


def synthetic():
    S1 = pd.read_csv(os.path.join(RES, "synth_S1.csv.gz"))
    fams = ["pinned", "slow", "fast"]
    fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.0), sharey=False)
    budgets = sorted(S1.budget.unique())
    for j, fam in enumerate(fams):
        for meth in METHODS:
            r = srates(S1[(S1.family == fam) & (S1.method == meth)], ["budget"]).reindex(budgets)
            axes[j].plot(budgets, 100 * r.correct, color=COLOR[meth], marker=MARK[meth], label=LABEL[meth])
        axes[j].set_title(f"{fam} keys: correct (%)", loc="left")
    for meth in METHODS:
        r = srates(S1[S1.method == meth], ["budget"]).reindex(budgets)
        axes[3].plot(budgets, 100 * r.stale, color=COLOR[meth], marker=MARK[meth], label=LABEL[meth])
    axes[3].set_title("all keys: stale (%)", loc="left")
    for ax in axes:
        ax.set_xscale("log", base=2)
        ax.set_xticks(budgets)
        ax.set_xticklabels([f"{b // 1000}k" for b in budgets])
        ax.set_xlabel("Budget")
    axes[0].legend(frameon=False, fontsize=5.5, loc="lower right")
    fig.tight_layout()
    save(fig, "synth_budget")
    t = srates(S1, ["budget", "method"])[["correct", "stale", "missing"]] * 100
    write_table(t, "table_synth_budget")

    S2 = pd.read_csv(os.path.join(RES, "synth_S2.csv.gz"))
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.2))
    Ts = sorted(S2.steps.unique())
    for meth in METHODS:
        r = srates(S2[S2.method == meth], ["steps"]).reindex(Ts)
        axes[0].plot(Ts, 100 * r.correct, color=COLOR[meth], marker=MARK[meth], label=LABEL[meth])
        axes[1].plot(Ts, 100 * r.stale, color=COLOR[meth], marker=MARK[meth], label=LABEL[meth])
    for ax, tt in zip(axes, ["correct (%)", "stale (%)"]):
        ax.set_xscale("log", base=2)
        ax.set_xticks(Ts)
        ax.set_xticklabels(Ts)
        ax.set_xlabel("Trajectory length (steps), budget 4k")
        ax.set_title(tt, loc="left")
    axes[0].legend(frameon=False, fontsize=6)
    fig.tight_layout()
    save(fig, "synth_horizon")

    # S3: stale vs number of writes, compared with Proposition 3
    S3 = pd.read_csv(os.path.join(RES, "synth_S3.csv.gz"))
    S3 = S3[S3.family != "pinned"]
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.1), sharey=True)
    for ax, B in zip(axes, sorted(S3.budget.unique())):
        for meth in ["random", "bm25", "window", "skc"]:
            s = S3[(S3.budget == B) & (S3.method == meth)]
            s = s.assign(m=s.n_writes.clip(upper=12))
            r = srates(s, ["m"])
            ax.plot(r.index, 100 * r.stale, color=COLOR[meth], marker=MARK[meth], label=LABEL[meth], lw=1.2)
        s = S3[(S3.budget == B) & (S3.method == "random")]
        rho = s.rho.mean()
        mm = np.arange(1, 13)
        ax.plot(mm, 100 * (1 - rho) * (1 - (1 - rho) ** (mm - 1)), color=MUTED, ls="--", lw=1.0,
                label=r"Prop. 3: $(1-\rho)(1-(1-\rho)^{m-1})$")
        ax.set_title(f"budget {B // 1000}k (random keeps $\\rho$={rho:.2f})", loc="left")
        ax.set_xlabel("writes to the key ($m$)")
    axes[0].set_ylabel("stale (%)")
    axes[0].legend(frameon=False, fontsize=5.5)
    fig.tight_layout()
    save(fig, "synth_stale_theory")

    # S4: extractor recall
    S4 = pd.read_csv(os.path.join(RES, "synth_S4.csv.gz"))
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.1), sharey=True)
    for ax, B in zip(axes, sorted(S4.budget.unique())):
        r = srates(S4[S4.budget == B], ["recall"]).sort_index()
        ax.plot(r.index, 100 * r.correct, color=COLOR["skc"], marker="o", label="correct")
        ax.plot(r.index, 100 * r.stale, color="#e34948", marker="s", label="stale")
        ax.plot(r.index, 100 * r.missing, color=MUTED, marker="^", label="missing")
        ax.set_xlabel("extractor recall $r$")
        ax.set_title(f"SKC, budget {B // 1000}k", loc="left")
        ax.invert_xaxis()
    axes[0].set_ylabel("% of probes")
    axes[0].legend(frameon=False)
    fig.tight_layout()
    save(fig, "synth_recall")
    write_table(srates(S4, ["budget", "recall"])[["correct", "stale", "missing"]] * 100, "table_synth_recall")

    # S5: capacity
    S5 = pd.read_csv(os.path.join(RES, "synth_S5.csv.gz"))
    fig, ax = plt.subplots(figsize=(3.4, 2.2))
    variants = [("skc", "SKC (knapsack priority)", COLOR["skc"], "o", "-"),
                ("skc-recency-only", "SKC, recency-only priority", COLOR["skc"], "s", "--"),
                ("skc-no-priority", "SKC, no priority (FIFO)", COLOR["skc"], "^", ":"),
                ("window", LABEL["window"], COLOR["window"], "s", "-"),
                ("obs_mask", LABEL["obs_mask"], COLOR["obs_mask"], "^", "-")]
    Ks = sorted(S5.n_keys.unique())
    for m, lab, c, mk, ls in variants:
        r = srates(S5[S5.method == m], ["n_keys"]).reindex(Ks)
        ax.plot(Ks, 100 * r.correct, color=c, marker=mk, ls=ls, label=lab)
    ax.set_xscale("log", base=2)
    ax.set_xticks(Ks)
    ax.set_xticklabels(Ks)
    ax.set_xlabel("number of state keys (budget 2k)")
    ax.set_title("correct (%)", loc="left")
    ax.legend(frameon=False, fontsize=5.5)
    fig.tight_layout()
    save(fig, "synth_capacity")
    write_table(srates(S5, ["n_keys", "method"])[["correct", "stale", "missing"]] * 100, "table_synth_capacity")


def corpus():
    p = os.path.join(RES, "corpus_stats.csv")
    if not os.path.exists(p):
        return
    c = pd.read_csv(p, index_col=0).loc[MODELS]
    rf = pd.read_csv(os.path.join(RES, "reference_distance_raw.csv.gz"))
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.1))
    y = np.arange(len(MODELS))
    left = np.zeros(len(MODELS))
    for col, lab, colr in [("tool_share", "tool outputs", "#2a78d6"), ("agent_share", "agent turns", "#eb6834"),
                           ("task_share", "task", "#1baf7a")]:
        axes[0].barh(y, 100 * c[col], left=left, color=colr, label=lab, height=0.6, edgecolor="white", linewidth=1)
        left += 100 * c[col].values
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([MODEL_LABEL[m] for m in MODELS])
    axes[0].invert_yaxis()
    axes[0].set_xlabel("share of context tokens (%)")
    axes[0].legend(frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    axes[0].grid(axis="y", visible=False)
    cols = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
    for m, colr in zip(MODELS, cols):
        x = np.sort(rf[rf.model == m].far.values)
        axes[1].plot(x, 1 - np.arange(len(x)) / len(x), color=colr, label=MODEL_LABEL[m], lw=1.3)
    axes[1].set_xscale("symlog", linthresh=1)
    axes[1].set_xlabel("steps since identifier first appeared")
    axes[1].set_title("P(distance > x) for identifiers used in actions", loc="left")
    axes[1].legend(frameon=False, fontsize=6)
    fig.tight_layout()
    save(fig, "corpus")


if __name__ == "__main__":
    for f in (corpus, real, real_ablation, synthetic):
        try:
            f()
            print("ok", f.__name__)
        except FileNotFoundError as e:
            print("skip", f.__name__, e)
