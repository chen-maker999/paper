"""算法 D（本文方法）：Reduce-and-HubCut。

D1 精确约简（reduction.py）。
D2 枢纽割贪心：每轮从动作集合中选“单位成本收益”最大、付得起的动作：
   (a) 单边动作：全部可删边，收益由支配树一次算出（引理 2）；
   (b) 目标割：对每个目标 t，切断“当前能到达 t 的入口”与 t 的最小割；
   (c) 枢纽割：按 score(h) = w(S~>h) * v(h~>T) 取前 M 个枢纽 h，对每个 h 生成三个割：
       S_h -> T（切断经过 h 的那一簇入口与全部目标），h -> T，S_h -> h。
   成本只计新增的边，所以不同动作共用的边不重复计费。
D2' LP 引导补救：若所有动作收益都为 0（典型情形：每个入口都有多条冗余路径），求当前状态下
   (IP-T) 的 LP 松弛，按 x_e 从大到小逐条加入付得起的边，直到风险开始下降，把这组边作为一个动作。
D3 局部搜索：删掉已选但冗余的边（删掉后风险不变），用省下的预算继续执行 D2，直到没有改进。

另外同时运行两种选择规则（按单位成本收益 / 按收益），取风险更低的解。
这是背包型贪心的常用修正：只按性价比选择，可能错过“贵但收益大”的动作。
"""
from __future__ import annotations

import numpy as np
from scipy.sparse.csgraph import breadth_first_order

from ..dominators import single_edge_gains
from ..flow import min_cut
from ..indexed import IndexedInstance
from ..lp import lp_relaxation
from ..reduction import reduce_instance
from .base import Solution
from .greedy import pick_best_edge

EPS = 1e-9


def _hubs(inst: IndexedInstance, mask, tr, n_hubs):
    fwd = inst.csr(mask)
    reach_w = np.zeros(inst.n)
    for s in inst.entries.tolist():
        order = breadth_first_order(fwd, s, directed=True, return_predecessors=False)
        reach_w[order] += inst.w[s]
    tv = np.zeros(inst.n)
    for t, nodes in tr.items():
        tv[nodes] += inst.v[t]
    score = reach_w * tv
    score[inst.entries] = 0
    score[inst.targets] = 0
    cand = np.flatnonzero(score > 0)
    if len(cand) == 0:
        return []
    order = np.lexsort((cand, -score[cand]))
    return cand[order[:n_hubs]].tolist()


def cut_actions(inst: IndexedInstance, mask, tr, n_hubs=10, use_target_cuts=True):
    """生成当前状态下的割动作（边编号集合的列表，已去重）。"""
    acts = {}

    def add(kind, cut):
        if cut:
            acts.setdefault(frozenset(cut), kind)

    targets = inst.targets.tolist()
    if use_target_cuts:
        for t, nodes in tr.items():
            ents = [s for s in nodes.tolist() if inst.w[s] > 0 and s != t]
            if ents:
                _, cut, _ = min_cut(inst, ents, [t], mask=mask)
                add("target", cut)
    if n_hubs > 0:
        for h in _hubs(inst, mask, tr, n_hubs):
            up = inst.reach([h], mask=mask, reverse=True)
            s_h = [s for s in inst.entries.tolist() if up[s]]
            _, cut, _ = min_cut(inst, s_h, targets, mask=mask)
            add("hub_S_T", cut)
            _, cut, _ = min_cut(inst, [h], targets, mask=mask)
            add("hub_h_T", cut)
            _, cut, _ = min_cut(inst, s_h, [h], mask=mask)
            add("hub_S_h", cut)
    return [(kind, set(c)) for c, kind in acts.items()]


def lp_action(inst: IndexedInstance, F, remaining, r_cur, max_edges=200):
    """D2'：LP 引导的补救动作。返回 (边集合, 成本, 收益)；找不到则返回 None。"""
    _, x, status = lp_relaxation(inst, remaining, removed=F)
    if status != "optimal":
        return None
    idx = np.flatnonzero(x > 1e-6)
    order = idx[np.lexsort((idx, inst.cost[idx], -x[idx]))]
    chosen, spent = set(), 0
    for e in order.tolist()[:max_edges]:
        c = int(inst.cost[e])
        if spent + c > remaining:
            continue
        chosen.add(e)
        spent += c
        g = r_cur - inst.risk(F | chosen)
        if g > EPS:
            return chosen, spent, g
    return None


def hubcut_greedy(inst: IndexedInstance, budget, removed=None, mode="ratio",
                  n_hubs=10, use_target_cuts=True, use_lp=True, log=None):
    F = set(removed or ())
    spent = inst.cost_of(F)
    while True:
        remaining = budget - spent
        if remaining <= 0:
            break
        mask = inst.alive_mask(F)
        tr = inst.target_reach(mask=mask)
        r_cur = float(sum(inst.v[t] * inst.w[nodes].sum() for t, nodes in tr.items()))
        if r_cur <= EPS:
            break

        best = None  # (key, gain, -cost, kind, 边集合)

        def consider(key, gain, cost, kind, edges):
            nonlocal best
            cand = (key, gain, -cost)
            if gain > EPS and (best is None or cand > best[:3]):
                best = (key, gain, -cost, kind, edges)

        gains = single_edge_gains(inst, mask=mask)
        e = pick_best_edge(inst, gains, F, remaining, mode)
        if e is not None:
            c = int(inst.cost[e])
            consider(gains[e] / c if mode == "ratio" else gains[e], gains[e], c, "edge", {e})

        for kind, cut in cut_actions(inst, mask, tr, n_hubs, use_target_cuts):
            extra = cut - F
            c = inst.cost_of(extra)
            if c == 0 or c > remaining:
                continue
            ub = r_cur / c if mode == "ratio" else r_cur  # 收益的上界，用来剪枝
            if best is not None and ub < best[0]:
                continue
            g = r_cur - inst.risk(F | extra)
            consider(g / c if mode == "ratio" else g, g, c, kind, extra)

        if best is None and use_lp:
            act = lp_action(inst, F, remaining, r_cur)
            if act is not None:
                edges, c, g = act
                best = (g / c, g, -c, "lp", edges)
        if best is None:
            break
        F |= best[4]
        spent += -best[2]
        if log is not None:
            log.append((best[3], len(best[4]), -best[2], best[1]))
    return F


def local_search(inst: IndexedInstance, budget, F, max_iter=10, **kw):
    F = set(F)
    for _ in range(max_iter):
        r = inst.risk(F)
        dropped = False
        for e in sorted(F, key=lambda e: (-int(inst.cost[e]), e)):
            if inst.risk(F - {e}) <= r + EPS:
                F.discard(e)
                dropped = True
        if not dropped:
            break
        F2 = hubcut_greedy(inst, budget, F, **kw)
        if inst.risk(F2) < r - EPS:
            F = F2
        else:
            F = F2  # 风险不变、成本不高于原解
            break
    return F


def solve(inst: IndexedInstance, budget: int, reduce=True, n_hubs=10, use_target_cuts=True,
          modes=("ratio", "gain"), use_local_search=True, use_lp=True) -> Solution:
    if reduce:
        work, stats = reduce_instance(inst)
    else:
        work, stats = inst, {}
    best_F, best_key, best_mode = set(), None, None
    for mode in modes:
        kw = dict(mode=mode, n_hubs=n_hubs, use_target_cuts=use_target_cuts, use_lp=use_lp)
        F = hubcut_greedy(work, budget, **kw)
        if use_local_search:
            F = local_search(work, budget, F, **kw)
        key = (work.risk(F), work.cost_of(F))
        if best_key is None or key < best_key:
            best_F, best_key, best_mode = F, key, mode
    info = {"mode": best_mode, "reduced_n": work.n, "reduced_m": work.m}
    info.update({f"red_{k}": v for k, v in stats.items()})
    return Solution(removed=work.to_original(best_F), cost=work.cost_of(best_F), info=info)
