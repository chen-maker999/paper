"""通用合成图（与 AD 语义无关），用于流程冒烟测试和理论构造的实验演示。

注意：论文的主实验应使用你自己转换好的数据集（graphio 格式）；这里的图只用于
  (1) 验证实验流程能跑通、测量算法的运行时间随规模的变化；
  (2) 定理 2 的归约构造——每个入口都有两条路径，单边贪心收益全为 0，用来展示贪心的“失明”。

用法：
  python -m experiments.synthetic layered --n 1000 --seeds 0-4 --out data/layered
  python -m experiments.synthetic clique --n 40 --p 0.2 --seeds 0-4 --out data/clique
"""
from __future__ import annotations

import argparse
import os
import random

import networkx as nx

from adinterdict.graphio import save_instance
from adinterdict.model import INF, Instance


def clique_reduction(H: nx.Graph, name="clique") -> Instance:
    """定理 2 的构造：H 的每个顶点 a 对应节点 a 和可删边 (a, t)（成本 1）；
    H 的每条边 {a,b} 对应入口 s_ab 和两条不可删边 (s_ab, a)、(s_ab, b)。
    预算为 k 时，最优保护量 = H 中 k 个顶点最多导出的边数（Densest k-Subgraph）。"""
    G = nx.DiGraph()
    t = "t"
    for a in H.nodes:
        G.add_edge(f"v{a}", t, cost=1)
    entries = {}
    for a, b in H.edges:
        s = f"s{a}_{b}"
        G.add_edge(s, f"v{a}", cost=INF)
        G.add_edge(s, f"v{b}", cost=INF)
        entries[s] = 1.0
    return Instance(G, entries, {t: 1.0}, name=name,
                    meta={"family": "clique_reduction", "H_nodes": H.number_of_nodes(),
                          "H_edges": H.number_of_edges()})


def layered(n=1000, seed=0, n_layers=6, out_deg=2.5, p_skip=0.1, p_back=0.03,
            p_fixed=0.3, entry_frac=0.1, n_targets=5, name=None) -> Instance:
    """分层随机有向图：第 0 层抽取入口，最后一层为目标。
    边主要连向下一层，少量跨层前向边和回边；一部分边不可删；成本取自 {1,2,3,5}。
    层宽从前往后递减，形成“很多入口、少量目标”的漏斗形状。"""
    rng = random.Random(seed)
    weights = [2 ** (n_layers - i) for i in range(n_layers)]
    sizes = [max(1, round(n * w / sum(weights))) for w in weights]
    sizes[-1] = n_targets
    layers, idx = [], 0
    for sz in sizes:
        layers.append([f"n{idx + j}" for j in range(sz)])
        idx += sz
    G = nx.DiGraph()
    for layer in layers:
        G.add_nodes_from(layer)

    def add(u, v):
        if u == v or G.has_edge(u, v):
            return
        c = INF if rng.random() < p_fixed else rng.choice([1, 2, 3, 5])
        G.add_edge(u, v, cost=c)

    for i, layer in enumerate(layers[:-1]):
        for u in layer:
            k = max(1, int(rng.expovariate(1 / out_deg)))
            for _ in range(k):
                r = rng.random()
                if r < p_skip and i + 2 < len(layers):
                    j = rng.randint(i + 2, len(layers) - 1)
                elif r < p_skip + p_back and i > 0:
                    j = rng.randint(0, i - 1)
                else:
                    j = i + 1
                add(u, rng.choice(layers[j]))
    # 目标之间也有少量互相可达（高价值资产常互相控制）。
    tg = layers[-1]
    for u in tg:
        if rng.random() < 0.3:
            add(u, rng.choice(tg))
    k = max(1, round(entry_frac * len(layers[0])))
    entries = {s: round(rng.uniform(0.5, 1.5), 2) for s in rng.sample(layers[0], k)}
    targets = {t: float(rng.choice([4, 6, 8, 10])) for t in tg}
    return Instance(G, entries, targets, name=name or f"layered_{n}_s{seed}",
                    meta={"family": "layered", "n": n, "seed": seed})


def _parse_seeds(s: str):
    if "-" in s:
        a, b = s.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in s.split(",")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("family", choices=["layered", "clique"])
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--p", type=float, default=0.2, help="clique 构造中 H=G(n,p) 的边概率")
    ap.add_argument("--seeds", default="0-4")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    for seed in _parse_seeds(a.seeds):
        if a.family == "layered":
            inst = layered(a.n, seed)
            tag = f"layered_{a.n}_s{seed}"
        else:
            H = nx.gnp_random_graph(a.n, a.p, seed=seed)
            tag = f"clique_{a.n}_s{seed}"
            inst = clique_reduction(H, name=tag)
        save_instance(inst, os.path.join(a.out, tag))
        print(tag, inst.G.number_of_nodes(), inst.G.number_of_edges())


if __name__ == "__main__":
    main()
