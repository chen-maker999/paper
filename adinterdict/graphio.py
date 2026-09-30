"""中性图格式的读写（与数据来源无关）。

一个实例 = 一个目录，包含两个 CSV 文件：

edges.csv   表头: src,dst,cost,deletable
            src/dst 为节点名（任意字符串）；deletable 为 1/0；
            cost 为正整数（deletable=0 时忽略，可留空）。
            同一对 (src,dst) 出现多行时合并为一条边：成本相加，任一行不可删则整条不可删。
nodes.csv   表头: node,entry_weight,target_value
            只需列出入口（entry_weight>0）和目标（target_value>0）；其余节点可省略。
            同一节点不能既是入口又是目标。

可选：meta.json，任意键值（如数据来源、生成参数），原样存入 Instance.meta。
"""
from __future__ import annotations

import csv
import json
import os

import networkx as nx

from .model import INF, Instance


def _truthy(x: str) -> bool:
    return str(x).strip().lower() in ("1", "true", "yes", "y", "t")


def load_instance(path: str, name: str | None = None) -> Instance:
    G = nx.DiGraph()
    with open(os.path.join(path, "edges.csv"), newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            u, v = row["src"].strip(), row["dst"].strip()
            if u == v:
                continue
            dele = _truthy(row.get("deletable", "0"))
            c = int(float(row["cost"])) if dele else INF
            if dele and c <= 0:
                raise ValueError(f"可删边 {u}->{v} 的成本必须为正整数")
            if G.has_edge(u, v):
                G[u][v]["cost"] = G[u][v]["cost"] + c
                G[u][v]["multiplicity"] += 1
            else:
                G.add_edge(u, v, cost=c, multiplicity=1)
    entries, targets = {}, {}
    with open(os.path.join(path, "nodes.csv"), newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            node = row["node"].strip()
            w = float(row.get("entry_weight") or 0)
            val = float(row.get("target_value") or 0)
            if node not in G:
                G.add_node(node)
            if w > 0:
                entries[node] = w
            if val > 0:
                targets[node] = val
    meta = {}
    mp = os.path.join(path, "meta.json")
    if os.path.exists(mp):
        with open(mp, encoding="utf-8") as f:
            meta = json.load(f)
    inst = Instance(G, entries, targets, name=name or os.path.basename(os.path.normpath(path)), meta=meta)
    inst.validate()
    return inst


def save_instance(inst: Instance, path: str) -> None:
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, "edges.csv"), "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["src", "dst", "cost", "deletable"])
        for u, v, c in inst.G.edges(data="cost"):
            if c == INF:
                wr.writerow([u, v, "", 0])
            else:
                wr.writerow([u, v, int(c), 1])
    with open(os.path.join(path, "nodes.csv"), "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["node", "entry_weight", "target_value"])
        for s, w in inst.entries.items():
            wr.writerow([s, w, 0])
        for t, val in inst.targets.items():
            wr.writerow([t, 0, val])
    if inst.meta:
        with open(os.path.join(path, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(inst.meta, f, ensure_ascii=False, indent=2)
