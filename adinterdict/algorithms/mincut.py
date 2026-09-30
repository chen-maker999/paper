"""算法 B：基于最小割的基线。

B1：全局最小割（观察 1）。B >= lambda 时就是最优解；否则不适用，返回空解。
B2：按目标求最小割，按“单位成本的保护价值”排序后贪心加入，剩余预算交给贪心（算法 A）补充。
"""
from __future__ import annotations

from ..flow import global_min_cut, min_cut
from ..indexed import IndexedInstance
from .base import Solution
from .greedy import greedy_extend


def solve_b1(inst: IndexedInstance, budget: int) -> Solution:
    lam, cut, _ = global_min_cut(inst)
    if cut is None or lam > budget:
        return Solution(removed=set(), cost=0, info={"lambda": lam, "applicable": False})
    return Solution(removed=inst.to_original(cut), cost=int(lam),
                    info={"lambda": lam, "applicable": True})


def solve_b2(inst: IndexedInstance, budget: int, fill_greedy=True) -> Solution:
    tr = inst.target_reach()
    actions = []
    for t, nodes in tr.items():
        entries = [s for s in nodes.tolist() if inst.w[s] > 0 and s != t]
        if not entries:
            continue
        value, cut, used = min_cut(inst, entries, [t])
        if not cut:
            continue
        protect = inst.v[t] * inst.w[used].sum()
        actions.append((protect / value, protect, t, cut))
    actions.sort(key=lambda a: (-a[0], -a[1], a[2]))

    F = set()
    spent = 0
    chosen = []
    for _, _, t, cut in actions:
        extra = cut - F
        c = inst.cost_of(extra)
        if spent + c <= budget:
            F |= extra
            spent += c
            chosen.append(t)
    n_cut_edges = len(F)
    if fill_greedy:
        F = greedy_extend(inst, budget, removed=F)
    return Solution(removed=inst.to_original(F), cost=inst.cost_of(F),
                    info={"targets_cut": len(chosen), "cut_edges": n_cut_edges})
