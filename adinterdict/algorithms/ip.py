"""算法 C：整数规划 (IP-T) 精确求解（阶段 2 文档 3.3 节），PuLP + CBC。

只为能到达 t 的节点建 rho^t_u 变量（其余节点的 rho 恒为 0），规模更小。
建议先做精确约简（reduction.reduce_instance），再在约简后的实例上求解。
"""
from __future__ import annotations

import time

import numpy as np
import pulp

from ..indexed import IndexedInstance
from .base import Solution


def build_model(inst: IndexedInstance, budget: int):
    prob = pulp.LpProblem("IP_T", pulp.LpMinimize)
    del_edges = np.flatnonzero(inst.deletable).tolist()
    x = {e: pulp.LpVariable(f"x_{e}", cat="Binary") for e in del_edges}
    tr = inst.target_reach()
    obj = []
    for t, nodes in tr.items():
        in_t = np.zeros(inst.n, dtype=bool)
        in_t[nodes] = True
        rho = {u: pulp.LpVariable(f"r_{t}_{u}", lowBound=0, upBound=1)
               for u in nodes.tolist() if u != t}

        def r(u, _rho=rho, _t=t):
            return 1 if u == _t else _rho[u]

        for e in np.flatnonzero(in_t[inst.dst]).tolist():
            u, v = int(inst.src[e]), int(inst.dst[e])
            if u == t:
                continue  # rho_t = 1，约束恒成立
            if inst.deletable[e]:
                prob += rho[u] >= r(v) - x[e]
            else:
                prob += rho[u] >= r(v)
        for s in nodes.tolist():
            if inst.w[s] > 0:
                obj.append(inst.v[t] * inst.w[s] * r(s))
    prob += pulp.lpSum(obj)
    prob += pulp.lpSum(int(inst.cost[e]) * x[e] for e in del_edges) <= budget
    return prob, x


def solve(inst: IndexedInstance, budget: int, time_limit=300, warm_start=None,
          threads=1) -> Solution:
    t0 = time.perf_counter()
    prob, x = build_model(inst, budget)
    if warm_start:
        for e, var in x.items():
            var.setInitialValue(1 if e in warm_start else 0)
    solver = pulp.PULP_CBC_CMD(msg=False, timeLimit=time_limit, threads=threads,
                               warmStart=bool(warm_start))
    build_time = time.perf_counter() - t0
    prob.solve(solver)
    F = {e for e, var in x.items() if var.value() is not None and var.value() > 0.5}
    status = {pulp.LpSolutionOptimal: "optimal",
              pulp.LpSolutionIntegerFeasible: "feasible"}.get(prob.sol_status, pulp.LpStatus[prob.status])
    return Solution(removed=inst.to_original(F), cost=inst.cost_of(F),
                    info={"status": status, "ip_objective": pulp.value(prob.objective),
                          "n_binary": len(x), "n_constraints": len(prob.constraints),
                          "build_time": build_time})
