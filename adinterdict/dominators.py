"""支配树与单边收益（引理 2）。

在反向图 H（以目标 t 为根）上求支配树。原图边 e=(u,v) 在 H 中是 v->u。
阶段 2 的引理 2 通过“把边细分成节点”来刻画必经边；这里用一个等价、且不需要细分的判据：

  H 中的边 v->u 支配 u  <=>  idom(u) = v，且 u 在 H 中的其他（可达）前驱 p 都被 u 支配。

（若存在一个不被 u 支配的前驱 p，则有一条避开 u 的路径 t ~> p，再走 p->u，就绕开了边 v->u；
 反之若所有其他前驱都被 u 支配，则任何到 u 的路径最后进入 u 的那条边只能是 v->u。）
再由“边支配 s 当且仅当边支配 u 且 u 支配 s”，删除 e 能切断的入口恰好是 u 在支配树中的子树里的入口。

因此每个目标只需一次支配树计算，就能得到所有边的收益：
  gain(e) = sum_t v_t * [e 对 t 是必经边] * W_t(u)，W_t(u) 为 u 的支配子树内的入口权重和。
"""
from __future__ import annotations

import numpy as np

from .indexed import IndexedInstance


def immediate_dominators(succ, pred, root, n):
    """Cooper–Harvey–Kennedy 迭代算法。

    succ/pred: 每个节点的后继/前驱列表（只含节点编号）。
    返回 (idom 数组, 后序编号数组)；不可达节点的 idom 为 -1。
    """
    post = np.full(n, -1, dtype=np.int64)
    order = []
    visited = np.zeros(n, dtype=bool)
    visited[root] = True
    stack = [(root, iter(succ[root]))]
    while stack:
        node, it = stack[-1]
        advanced = False
        for nxt in it:
            if not visited[nxt]:
                visited[nxt] = True
                stack.append((nxt, iter(succ[nxt])))
                advanced = True
                break
        if not advanced:
            stack.pop()
            post[node] = len(order)
            order.append(node)
    idom = np.full(n, -1, dtype=np.int64)
    idom[root] = root
    rpo = order[::-1][1:]
    post_l = post.tolist()
    idom_l = idom.tolist()

    changed = True
    while changed:
        changed = False
        for b in rpo:
            new = -1
            for p in pred[b]:
                if idom_l[p] == -1:
                    continue
                if new == -1:
                    new = p
                    continue
                f1, f2 = p, new
                while f1 != f2:
                    while post_l[f1] < post_l[f2]:
                        f1 = idom_l[f1]
                    while post_l[f2] < post_l[f1]:
                        f2 = idom_l[f2]
                new = f1
            if idom_l[b] != new:
                idom_l[b] = new
                changed = True
    return np.asarray(idom_l, dtype=np.int64), post


def single_edge_gains(inst: IndexedInstance, removed=(), mask=None, node_filter=None):
    """删除每条边（单独删除）能让风险下降多少：返回长度为 m 的数组。

    node_filter: 可选的节点布尔掩码。传入“从入口可达”的节点集合可以加速，结果不变
    （入口到目标的路径上的节点都从入口可达）。
    """
    if mask is None:
        mask = inst.alive_mask(removed)
    if node_filter is None:
        node_filter = inst.reach(inst.entries, mask=mask)
    alive = np.flatnonzero(mask & node_filter[inst.src] & node_filter[inst.dst])
    n = inst.n
    # H = 反向图：H 中 v->u 对应原图 u->v。
    h_succ = [[] for _ in range(n)]  # h_succ[v] = 原图前驱 u
    h_pred = [[] for _ in range(n)]  # h_pred[u] = 原图后继 v
    edge_of = {}
    for e, u, v in zip(alive.tolist(), inst.src[alive].tolist(), inst.dst[alive].tolist()):
        h_succ[v].append(u)
        h_pred[u].append(v)
        edge_of[(u, v)] = e

    gains = np.zeros(inst.m)
    w = inst.w
    for t in inst.targets.tolist():
        if not node_filter[t]:
            continue
        idom, _ = immediate_dominators(h_succ, h_pred, t, n)
        reach_nodes = np.flatnonzero(idom >= 0)
        # 支配树的 DFS 进出时间，用于 O(1) 判断“x 是否被 u 支配”。
        children = {}
        for x in reach_nodes.tolist():
            if x != t:
                children.setdefault(int(idom[x]), []).append(x)
        tin = np.zeros(n, dtype=np.int64)
        tout = np.zeros(n, dtype=np.int64)
        sub_w = np.zeros(n)
        clock = 0
        stack = [(t, 0)]
        while stack:
            x, state = stack.pop()
            if state == 0:
                tin[x] = clock
                clock += 1
                stack.append((x, 1))
                for c in children.get(x, ()):
                    stack.append((c, 0))
            else:
                tout[x] = clock
                s = w[x]
                for c in children.get(x, ()):
                    s += sub_w[c]
                sub_w[x] = s
        vt = inst.v[t]
        idom_l = idom.tolist()
        tin_l, tout_l = tin.tolist(), tout.tolist()
        for u in reach_nodes.tolist():
            if u == t or sub_w[u] <= 0:
                continue
            d = idom_l[u]
            ok = True
            lo, hi = tin_l[u], tout_l[u]
            for p in h_pred[u]:
                if p == d or idom_l[p] == -1:
                    continue
                if not (lo <= tin_l[p] < hi):
                    ok = False
                    break
            if ok:
                # idom(u) 不是 u 的直接前驱时 ok 必为 False，这里只是保险。
                e = edge_of.get((u, d))
                if e is not None and inst.deletable[e]:
                    gains[e] += vt * sub_w[u]
    return gains
