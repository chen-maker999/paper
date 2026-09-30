"""整数下标表示的实例，以及残余风险 R(F) 等指标的快速计算。

所有算法都在 IndexedInstance 上工作：节点编号 0..n-1，边编号 0..m-1，
删边集合 F 用边编号的集合表示。BFS 使用 scipy 的 C 实现。
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import breadth_first_order, shortest_path

from .model import INF, Instance


class IndexedInstance:
    def __init__(self, n, src, dst, cost, deletable, w, v,
                 node_labels=None, edge_orig=None, name=""):
        self.n = int(n)
        self.src = np.asarray(src, dtype=np.int64)
        self.dst = np.asarray(dst, dtype=np.int64)
        self.m = len(self.src)
        self.deletable = np.asarray(deletable, dtype=bool)
        # 不可删边的成本记为 0，只作占位；是否可删以 deletable 为准。
        self.cost = np.where(self.deletable, np.asarray(cost, dtype=np.int64), 0)
        self.w = np.asarray(w, dtype=float)
        self.v = np.asarray(v, dtype=float)
        self.entries = np.flatnonzero(self.w > 0)
        self.targets = np.flatnonzero(self.v > 0)
        self.node_labels = node_labels if node_labels is not None else list(range(self.n))
        # edge_orig[e]：该边对应的原始实例边编号列表（约简后的实例才需要）。
        self.edge_orig = edge_orig
        self.name = name
        self._succ = None
        self._pred = None

    # ------------------------------------------------------------------ 构造
    @classmethod
    def from_instance(cls, inst: Instance) -> "IndexedInstance":
        G = inst.G
        labels = list(G.nodes)
        idx = {u: i for i, u in enumerate(labels)}
        src, dst, cost, dele = [], [], [], []
        edge_labels = []
        for u, v, c in G.edges(data="cost"):
            src.append(idx[u])
            dst.append(idx[v])
            dele.append(c != INF)
            cost.append(int(c) if c != INF else 0)
            edge_labels.append((u, v))
        w = np.zeros(len(labels))
        val = np.zeros(len(labels))
        for s, ws in inst.entries.items():
            w[idx[s]] = ws
        for t, vt in inst.targets.items():
            val[idx[t]] = vt
        obj = cls(len(labels), src, dst, cost, dele, w, val, node_labels=labels, name=inst.name)
        obj.edge_labels = edge_labels
        return obj

    # ---------------------------------------------------------------- 邻接表
    @property
    def succ(self):
        """succ[u] = [(v, e), ...]"""
        if self._succ is None:
            self._build_adj()
        return self._succ

    @property
    def pred(self):
        """pred[v] = [(u, e), ...]"""
        if self._pred is None:
            self._build_adj()
        return self._pred

    def _build_adj(self):
        succ = [[] for _ in range(self.n)]
        pred = [[] for _ in range(self.n)]
        for e, (u, v) in enumerate(zip(self.src.tolist(), self.dst.tolist())):
            succ[u].append((v, e))
            pred[v].append((u, e))
        self._succ, self._pred = succ, pred

    # -------------------------------------------------------------- 基本工具
    def alive_mask(self, removed=()) -> np.ndarray:
        mask = np.ones(self.m, dtype=bool)
        if removed:
            mask[np.fromiter(removed, dtype=np.int64)] = False
        return mask

    def csr(self, mask=None, reverse=False) -> sp.csr_matrix:
        if mask is None:
            mask = np.ones(self.m, dtype=bool)
        r, c = (self.dst[mask], self.src[mask]) if reverse else (self.src[mask], self.dst[mask])
        data = np.ones(len(r), dtype=np.int8)
        return sp.csr_matrix((data, (r, c)), shape=(self.n, self.n))

    def cost_of(self, removed) -> int:
        removed = list(removed)
        if not removed:
            return 0
        arr = np.asarray(removed, dtype=np.int64)
        if not self.deletable[arr].all():
            raise ValueError("删边集合中含有不可删边")
        return int(self.cost[arr].sum())

    def reach(self, starts, mask=None, reverse=False) -> np.ndarray:
        """从多个起点出发的可达节点（布尔数组）。reverse=True 时沿反向边走。"""
        starts = np.asarray(list(starts), dtype=np.int64)
        out = np.zeros(self.n, dtype=bool)
        if len(starts) == 0:
            return out
        if mask is None:
            mask = np.ones(self.m, dtype=bool)
        r, c = (self.dst[mask], self.src[mask]) if reverse else (self.src[mask], self.dst[mask])
        # 加一个虚拟起点 n，连向所有起点，做一次 BFS。
        r = np.concatenate([r, np.full(len(starts), self.n)])
        c = np.concatenate([c, starts])
        g = sp.csr_matrix((np.ones(len(r), dtype=np.int8), (r, c)), shape=(self.n + 1, self.n + 1))
        order = breadth_first_order(g, self.n, directed=True, return_predecessors=False)
        out[order[order < self.n]] = True
        return out

    def target_reach(self, removed=(), mask=None) -> dict:
        """对每个目标 t，返回能到达 t 的节点下标数组（在 G-F 中）。"""
        if mask is None:
            mask = self.alive_mask(removed)
        rev = self.csr(mask, reverse=True)
        return {int(t): breadth_first_order(rev, int(t), directed=True, return_predecessors=False)
                for t in self.targets}

    # ------------------------------------------------------------ 目标函数
    def risk(self, removed=(), mask=None) -> float:
        """残余风险 R(F) = sum_{s,t} w_s v_t 1[s 可达 t]（定义 2）。"""
        tr = self.target_reach(removed, mask)
        return float(sum(self.v[t] * self.w[nodes].sum() for t, nodes in tr.items()))

    def metrics(self, removed=()) -> dict:
        """阻断效果指标：风险、风险比例、可达对比例、可达对的平均最短路长度。"""
        mask = self.alive_mask(removed)
        r0 = self.risk()
        r = self.risk(mask=mask)
        rev = self.csr(mask, reverse=True)
        if len(self.targets):
            dist = shortest_path(rev, directed=True, unweighted=True, indices=self.targets)
            d = dist[:, self.entries]
            finite = np.isfinite(d)
            n_pairs = d.size
            reach_frac = finite.sum() / n_pairs if n_pairs else 0.0
            mean_dist = float(d[finite].mean()) if finite.any() else float("nan")
        else:
            reach_frac, mean_dist = 0.0, float("nan")
        return {
            "risk": r,
            "risk0": r0,
            "risk_ratio": r / r0 if r0 > 0 else 0.0,
            "reach_pair_frac": float(reach_frac),
            "mean_dist": mean_dist,
        }

    # ------------------------------------------------------------ 约简映射
    def to_original(self, removed) -> set:
        """把本实例上的删边集合映射回原始实例的边编号。"""
        if self.edge_orig is None:
            return set(removed)
        out = set()
        for e in removed:
            out.update(self.edge_orig[e])
        return out
