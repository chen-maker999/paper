"""最小割（观察 1、算法 B、算法 D 的割动作）。

使用 scipy 的 Dinic 最大流（C 实现，要求整数容量）。
不可删边的容量取一个大数 BIG（大于全部可删边成本之和），
割值 >= BIG 说明源汇之间存在只由不可删边组成的路径，即无法切断。
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import breadth_first_order, maximum_flow

from .indexed import IndexedInstance


def uncuttable_sources(inst: IndexedInstance, sources, sinks, mask=None) -> np.ndarray:
    """sources 中只经不可删边就能到达 sinks 的节点（布尔数组，长度 n）。"""
    if mask is None:
        mask = np.ones(inst.m, dtype=bool)
    fixed = mask & ~inst.deletable
    return inst.reach(sinks, mask=fixed, reverse=True)


def min_cut(inst: IndexedInstance, sources, sinks, removed=(), mask=None,
            drop_uncuttable=True):
    """在 G-F 上求 sources 到 sinks 的最小割。

    返回 (割值, 割边编号集合, 实际使用的源点列表)。
    drop_uncuttable=True 时先去掉无法切断的源点（它们无论如何都保护不了）。
    若没有可切断的源点，返回 (0, set(), [])。
    """
    if mask is None:
        mask = inst.alive_mask(removed)
    sources = np.asarray(list(sources), dtype=np.int64)
    sinks = np.asarray(list(sinks), dtype=np.int64)
    sink_set = np.zeros(inst.n, dtype=bool)
    sink_set[sinks] = True
    sources = sources[~sink_set[sources]]
    if drop_uncuttable and len(sources):
        bad = uncuttable_sources(inst, sources, sinks, mask)
        sources = sources[~bad[sources]]
    if len(sources) == 0 or len(sinks) == 0:
        return 0, set(), []

    n = inst.n
    sigma, tau = n, n + 1
    big = int(inst.cost[inst.deletable].sum()) + 1
    alive = np.flatnonzero(mask)
    cap = np.where(inst.deletable[alive], inst.cost[alive], big)
    rows = np.concatenate([inst.src[alive], np.full(len(sources), sigma), sinks])
    cols = np.concatenate([inst.dst[alive], sources, np.full(len(sinks), tau)])
    caps = np.concatenate([cap, np.full(len(sources), big), np.full(len(sinks), big)])
    if big * max(len(sources), len(sinks), 1) >= 2**31 - 1:
        raise OverflowError("容量超出 int32 范围，请缩小成本或实例规模")
    C = sp.csr_matrix((caps.astype(np.int32), (rows, cols)), shape=(n + 2, n + 2))
    res = maximum_flow(C, sigma, tau, method="dinic")
    value = int(res.flow_value)
    if value >= big:
        return float("inf"), None, sources.tolist()

    # 残量网络上从 sigma 出发可达的点集 X，割边为 X -> 非 X 的原图边。
    R = (C - res.flow).tocsr()
    R.data[R.data < 0] = 0
    R.eliminate_zeros()
    order = breadth_first_order(R, sigma, directed=True, return_predecessors=False)
    in_x = np.zeros(n + 2, dtype=bool)
    in_x[order] = True
    cut_mask = in_x[inst.src[alive]] & ~in_x[inst.dst[alive]]
    cut = set(alive[cut_mask].tolist())
    assert int(inst.cost[list(cut)].sum()) == value if cut else value == 0
    return value, cut, sources.tolist()


def global_min_cut(inst: IndexedInstance, removed=()):
    """观察 1：切断全部（可切断的）入口—目标对的最小成本 lambda 及对应割。"""
    return min_cut(inst, inst.entries, inst.targets, removed=removed)
