"""算法 D 的第一部分 (D1)：保持最优值不变的精确约简。

规则（阶段 2 文档 4.4 节）：
  R1 相关性剪枝：只保留“从某入口可达、且能到达某目标”的节点。
  R2 不可删 SCC 缩点：只用不可删边求强连通分量并缩点（分量内节点在任何 F 下都互相可达）。
  R3 不可删孪生节点合并：非目标节点 u1, u2 的出边都不可删、且出邻居集合相同，则可达集永远相同，合并。
     （阶段 2 只要求入口；这里推广到任意非目标节点，证明相同。）
  R4 不可删度 1 收缩（推广了阶段 2 的“链压缩”）：
     非入口、非目标节点 u 只有一条入边 p->u 且不可删 => 把 u 并入 p（u 的出边改为从 p 出发）；
     或只有一条出边 u->q 且不可删 => 把 u 并入 q（u 的入边改为指向 q）。

合并后同一对节点之间出现多条边时，把它们合并成一条，成本相加：
这些边是两点之间互为替代的“一跳”连接，只有全部删掉才能切断，只删一部分没有任何效果。
每条约简后的边记录它对应的原始边编号（edge_orig），用于把解映射回原图。
"""
from __future__ import annotations

import math
from collections import defaultdict

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components

from .indexed import IndexedInstance

INF = math.inf


class _DictGraph:
    def __init__(self):
        self.E = {}  # (u, v) -> [cost(int 或 INF), [原始边编号...]]
        self.succ = defaultdict(set)
        self.pred = defaultdict(set)
        self.nodes = set()
        self.w = {}
        self.v = {}
        self.members = {}

    def add_edge(self, u, v, cost, orig):
        if u == v:
            return
        key = (u, v)
        if key in self.E:
            rec = self.E[key]
            rec[0] = rec[0] + cost
            rec[1].extend(orig)
        else:
            self.E[key] = [cost, list(orig)]
            self.succ[u].add(v)
            self.pred[v].add(u)

    def pop_edge(self, u, v):
        rec = self.E.pop((u, v))
        self.succ[u].discard(v)
        self.pred[v].discard(u)
        return rec

    def drop_node(self, u):
        assert not self.succ[u] and not self.pred[u]
        self.nodes.discard(u)
        for d in (self.succ, self.pred, self.w, self.v, self.members):
            d.pop(u, None)

    def merge(self, group):
        """把 group 中的节点合并成一个节点（编号取最小者），返回代表节点。"""
        group = set(group)
        rep = min(group)
        touched = []
        for x in group:
            for y in list(self.succ[x]):
                touched.append((x, y, self.pop_edge(x, y)))
            for y in list(self.pred[x]):
                touched.append((y, x, self.pop_edge(y, x)))
        w = sum(self.w[x] for x in group)
        v = sum(self.v[x] for x in group)
        mem = [m for x in sorted(group) for m in self.members[x]]
        for x in group:
            if x != rep:
                self.drop_node(x)
        self.w[rep], self.v[rep], self.members[rep] = w, v, mem
        for a, b, (c, o) in touched:
            a = rep if a in group else a
            b = rep if b in group else b
            self.add_edge(a, b, c, o)
        return rep

    def absorb(self, u, into, direction):
        """R4：把节点 u 并入 into。direction='out' 表示把 u 的出边移给 into，'in' 表示入边。"""
        if direction == "out":
            self.pop_edge(into, u)
            for x in list(self.succ[u]):
                c, o = self.pop_edge(u, x)
                self.add_edge(into, x, c, o)
        else:
            self.pop_edge(u, into)
            for x in list(self.pred[u]):
                c, o = self.pop_edge(x, u)
                self.add_edge(x, into, c, o)
        self.members[into] = self.members[into] + self.members[u]
        self.drop_node(u)

    def size(self):
        return len(self.nodes), len(self.E)


def _rule_scc(g: _DictGraph) -> bool:
    nodes = sorted(g.nodes)
    idx = {u: i for i, u in enumerate(nodes)}
    rows, cols = [], []
    for (u, v), (c, _) in g.E.items():
        if c == INF:
            rows.append(idx[u])
            cols.append(idx[v])
    if not rows:
        return False
    M = sp.csr_matrix((np.ones(len(rows), dtype=np.int8), (rows, cols)), shape=(len(nodes), len(nodes)))
    k, labels = connected_components(M, directed=True, connection="strong")
    if k == len(nodes):
        return False
    groups = defaultdict(list)
    for i, lab in enumerate(labels.tolist()):
        groups[lab].append(nodes[i])
    changed = False
    for grp in groups.values():
        if len(grp) > 1:
            g.merge(grp)
            changed = True
    return changed


def _rule_twins(g: _DictGraph) -> bool:
    groups = defaultdict(list)
    for u in g.nodes:
        if g.v[u] > 0 or not g.succ[u]:
            continue
        if all(g.E[(u, x)][0] == INF for x in g.succ[u]):
            groups[frozenset(g.succ[u])].append(u)
    changed = False
    for grp in groups.values():
        if len(grp) > 1:
            g.merge(grp)
            changed = True
    return changed


def _rule_degree_one(g: _DictGraph) -> bool:
    changed = False
    queue = list(g.nodes)
    while queue:
        u = queue.pop()
        if u not in g.nodes or g.w[u] > 0 or g.v[u] > 0:
            continue
        if len(g.pred[u]) == 1:
            (p,) = g.pred[u]
            if g.E[(p, u)][0] == INF:
                nbrs = list(g.succ[u])
                g.absorb(u, p, "out")
                queue.append(p)
                queue.extend(nbrs)
                changed = True
                continue
        if len(g.succ[u]) == 1:
            (q,) = g.succ[u]
            if g.E[(u, q)][0] == INF:
                nbrs = list(g.pred[u])
                g.absorb(u, q, "in")
                queue.append(q)
                queue.extend(nbrs)
                changed = True
    return changed


def reduce_instance(inst: IndexedInstance, rules=("R1", "R2", "R3", "R4"), max_iter=20):
    """返回 (约简后的 IndexedInstance, 统计信息)。约简后实例的 edge_orig 指向 inst 的原始边编号。"""
    stats = {"n0": inst.n, "m0": inst.m}
    base_orig = inst.edge_orig if inst.edge_orig is not None else [[e] for e in range(inst.m)]

    if "R1" in rules:
        keep = inst.reach(inst.entries) & inst.reach(inst.targets, reverse=True)
    else:
        keep = np.ones(inst.n, dtype=bool)

    g = _DictGraph()
    for u in np.flatnonzero(keep).tolist():
        g.nodes.add(u)
        g.w[u] = float(inst.w[u])
        g.v[u] = float(inst.v[u])
        lab = inst.node_labels[u]
        g.members[u] = list(lab) if isinstance(lab, list) else [lab]
    for e in range(inst.m):
        u, v = int(inst.src[e]), int(inst.dst[e])
        if keep[u] and keep[v]:
            c = int(inst.cost[e]) if inst.deletable[e] else INF
            g.add_edge(u, v, c, base_orig[e])
    stats["n_R1"], stats["m_R1"] = g.size()

    for it in range(max_iter):
        changed = False
        if "R2" in rules:
            changed |= _rule_scc(g)
        if "R3" in rules:
            while _rule_twins(g):
                changed = True
        if "R4" in rules:
            changed |= _rule_degree_one(g)
        if not changed:
            break
    stats["iterations"] = it + 1
    stats["n1"], stats["m1"] = g.size()

    nodes = sorted(g.nodes)
    idx = {u: i for i, u in enumerate(nodes)}
    src, dst, cost, dele, orig = [], [], [], [], []
    for (u, v), (c, o) in sorted(g.E.items()):
        src.append(idx[u])
        dst.append(idx[v])
        dele.append(c != INF)
        cost.append(int(c) if c != INF else 0)
        orig.append(sorted(o))
    red = IndexedInstance(
        len(nodes), src, dst, cost, dele,
        [g.w[u] for u in nodes], [g.v[u] for u in nodes],
        node_labels=[g.members[u] for u in nodes], edge_orig=orig,
        name=inst.name + "/reduced",
    )
    return red, stats
