"""(IP-T) 的 LP 松弛（x 也松弛到 [0,1]），用 scipy HiGHS 求解。

用途：
  1. 下界：LP 最优值 <= CMTI 最优值，大规模图上 IP 解不动时用它报告“与下界的差距”；
  2. 算法 D 的 LP 引导补救：贪心停滞（所有动作收益为 0）时，按 LP 解中的 x_e 排序选边。
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.optimize import linprog

from .indexed import IndexedInstance


def lp_relaxation(inst: IndexedInstance, budget: float, removed=(), mask=None):
    """返回 (LP 最优值, x 数组(长度 m，未参与的边为 0), 状态字符串)。"""
    if mask is None:
        mask = inst.alive_mask(removed)
    tr = inst.target_reach(mask=mask)
    cand = np.flatnonzero(mask & inst.deletable)
    xcol = -np.ones(inst.m, dtype=np.int64)
    xcol[cand] = np.arange(len(cand))
    ncol = len(cand)
    obj = [np.zeros(len(cand))]
    const = 0.0
    rows, cols, vals, rhs = [], [], [], []
    nrow = 0
    alive = np.flatnonzero(mask)
    for t, nodes in tr.items():
        rcol = -np.ones(inst.n, dtype=np.int64)
        others = nodes[nodes != t]
        rcol[others] = ncol + np.arange(len(others))
        ncol += len(others)
        o = np.zeros(len(others))
        o[:] = inst.v[t] * inst.w[others]
        obj.append(o)
        const += inst.v[t] * inst.w[t]
        in_t = np.zeros(inst.n, dtype=bool)
        in_t[nodes] = True
        es = alive[in_t[inst.dst[alive]] & (inst.src[alive] != t)]
        u, v = inst.src[es], inst.dst[es]
        k = len(es)
        r = nrow + np.arange(k)
        nrow += k
        # 约束： -rho_u + rho_v - x_e <= 0 ；若 v == t，则 rho_v = 1 移到右端。
        rows.append(r); cols.append(rcol[u]); vals.append(-np.ones(k))
        vt = v != t
        rows.append(r[vt]); cols.append(rcol[v[vt]]); vals.append(np.ones(vt.sum()))
        de = inst.deletable[es]
        rows.append(r[de]); cols.append(xcol[es[de]]); vals.append(-np.ones(de.sum()))
        rhs.append(np.where(vt, 0.0, -1.0))
    # 预算约束
    rows.append(np.array([nrow])); cols.append(np.array([0]))  # 占位，下面整体替换
    vals.append(np.array([0.0]))
    rows[-1] = np.full(len(cand), nrow)
    cols[-1] = np.arange(len(cand))
    vals[-1] = inst.cost[cand].astype(float)
    rhs.append(np.array([float(budget)]))
    nrow += 1
    c = np.concatenate(obj)
    A = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                      shape=(nrow, ncol))
    res = linprog(c, A_ub=A, b_ub=np.concatenate(rhs), bounds=(0, 1), method="highs")
    x = np.zeros(inst.m)
    if res.status != 0:
        return float("nan"), x, res.message
    x[cand] = res.x[:len(cand)]
    return float(res.fun + const), x, "optimal"
