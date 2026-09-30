"""算法 A：贪心删边（基线），用支配树一次算出全部单边收益（引理 2）。

每轮选择“单位成本收益”最大、且付得起的边；所有候选边的收益都为 0 时停止。
"""
from __future__ import annotations

import numpy as np

from ..dominators import single_edge_gains
from ..indexed import IndexedInstance
from .base import Solution


def pick_best_edge(inst: IndexedInstance, gains, removed, remaining, mode="ratio"):
    """返回收益为正、付得起、未删除的最佳边；没有则返回 None。"""
    cand = (gains > 0) & inst.deletable & (inst.cost <= remaining)
    if removed:
        cand[np.fromiter(removed, dtype=np.int64)] = False
    idx = np.flatnonzero(cand)
    if len(idx) == 0:
        return None
    if mode == "ratio":
        key = gains[idx] / inst.cost[idx]
    else:
        key = gains[idx]
    # 主键最大；并列时收益更大者优先；再并列取编号最小（np.lexsort 最后一个键为主键）。
    order = np.lexsort((idx, -gains[idx], -key))
    return int(idx[order[0]])


def greedy_extend(inst: IndexedInstance, budget, removed=None, mode="ratio"):
    """从已有删边集合出发继续贪心，返回新的删边集合（本实例的边编号）。"""
    F = set(removed or ())
    spent = inst.cost_of(F)
    while True:
        remaining = budget - spent
        if remaining <= 0:
            break
        gains = single_edge_gains(inst, F)
        e = pick_best_edge(inst, gains, F, remaining, mode)
        if e is None:
            break
        F.add(e)
        spent += int(inst.cost[e])
    return F


def solve(inst: IndexedInstance, budget: int, mode="ratio") -> Solution:
    F = greedy_extend(inst, budget, mode=mode)
    return Solution(removed=inst.to_original(F), cost=inst.cost_of(F),
                    info={"rounds": len(F)})
