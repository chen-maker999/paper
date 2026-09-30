"""测试用的随机小实例与暴力枚举。"""
from __future__ import annotations

import itertools
import random

import numpy as np

from adinterdict.indexed import IndexedInstance


def random_instance(seed, n=9, p=0.25, p_fixed=0.35, n_entries=3, n_targets=2,
                    n_twins=2, max_deletable=11):
    rng = random.Random(seed)
    edges = {}
    for u in range(n):
        for v in range(n):
            if u != v and rng.random() < p:
                edges[(u, v)] = rng.random() < p_fixed
    nodes = list(range(n))
    rng.shuffle(nodes)
    entries = nodes[:n_entries]
    targets = nodes[n_entries:n_entries + n_targets]
    # 人为制造一些“孪生”节点（出边全不可删且出邻居相同），用于覆盖约简规则 R3。
    others = [u for u in range(n) if u not in targets]
    for _ in range(n_twins):
        a, b = rng.sample(others, 2)
        outs = [v for (u, v) in edges if u == a and v != b]
        for key in [k for k in edges if k[0] in (a, b)]:
            del edges[key]
        for v in outs:
            edges[(a, v)] = True
            if v != b:
                edges[(b, v)] = True
    # 控制可删边数量，保证暴力枚举可行。
    del_keys = [k for k, fixed in edges.items() if not fixed]
    rng.shuffle(del_keys)
    for k in del_keys[max_deletable:]:
        edges[k] = True
    src, dst, cost, dele = [], [], [], []
    for (u, v), fixed in sorted(edges.items()):
        src.append(u)
        dst.append(v)
        dele.append(not fixed)
        cost.append(rng.randint(1, 4))
    w = np.zeros(n)
    val = np.zeros(n)
    for s in entries:
        w[s] = rng.choice([0.5, 1.0, 2.0])
    for t in targets:
        val[t] = rng.choice([1.0, 3.0, 5.0])
    return IndexedInstance(n, src, dst, cost, dele, w, val, name=f"rand{seed}")


def brute_force(inst: IndexedInstance, budget):
    """枚举全部可删边子集，返回 (最优风险, 最优解)。"""
    del_edges = np.flatnonzero(inst.deletable).tolist()
    best = (inst.risk(), set())
    for k in range(1, len(del_edges) + 1):
        for F in itertools.combinations(del_edges, k):
            if inst.cost_of(F) <= budget:
                r = inst.risk(set(F))
                if r < best[0] - 1e-9:
                    best = (r, set(F))
    return best
