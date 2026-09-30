"""AD 攻击图的数据模型：节点/边类型、删除成本规则、问题实例。

对应阶段 2 文档第 3.1 节（定义 1、表 1）。
边方向与 BloodHound 一致：u -> v 表示“控制 u 之后就能控制 v”。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import networkx as nx

INF = math.inf

# 表 1：边类型 -> 默认删除成本。这些数值只是实验假设（阶段 3 会做敏感性分析）。
DEFAULT_EDGE_COST = {
    "MemberOf": 5,
    "AdminTo": 3,
    "HasSession": 1,
    "CanRDP": 2,
    "ExecuteDCOM": 2,
    "CanPSRemote": 2,
    "GenericAll": 2,
    "GenericWrite": 2,
    "WriteDacl": 2,
    "WriteOwner": 2,
    "Owns": 2,
    "ForceChangePassword": 2,
    "AddMember": 2,
    "AddSelf": 2,
    "AllExtendedRights": 2,
    "ReadLAPSPassword": 2,
    "ReadGMSAPassword": 2,
    "AllowedToDelegate": 2,
    "AllowedToAct": 2,
    "AddAllowedToAct": 2,
    "WriteSPN": 2,
    "AddKeyCredentialLink": 2,
    # Combined GetChanges + GetChangesAll; one ACL revocation breaks the pair.
    "DCSync": 2,
    "Contains": INF,
    "GPLink": INF,
}
OTHER_EDGE_COST = 2

# 结构性关系：永远不可删。
STRUCTURAL_EDGES = {"Contains", "GPLink"}

# 成员关系不可删的“默认组”（按 wellknown 标签识别）。
# 注意：指向 Domain Admins 等特权组的成员关系*可以*删——低层组被嵌套进特权组正是要修复的误配置；
# 合法 Tier-0 管理员的成员关系由规则 3（Tier-0 起点的边不可删）保护。
FIXED_MEMBERSHIP_GROUPS = {
    "DOMAIN_USERS",
    "DOMAIN_COMPUTERS",
    "DOMAIN_CONTROLLERS",
    "EVERYONE",
    "AUTHENTICATED_USERS",
    "BUILTIN_USERS",
}

NODE_TYPES = ("User", "Computer", "Group", "OU", "GPO", "Domain")


def relation_cost(G: nx.DiGraph, u, v, etype: str) -> float:
    """一条 BloodHound 关系的删除成本。

    规则（生成器和 BloodHound 读取器共用，保证两边一致）：
      1. Contains / GPLink 等结构性关系不可删；
      2. 指向默认组（Domain Users 等）以及指向对象主组（PrimaryGroupSID）的 MemberOf 不可删；
      3. 以 Tier-0 对象为起点的边不可删（Tier-0 内部关系，不在防御者可调整的范围内）；
      4. 其余按表 1 取默认成本。
    """
    if etype in STRUCTURAL_EDGES:
        return INF
    if etype == "MemberOf":
        if G.nodes[v].get("wellknown") in FIXED_MEMBERSHIP_GROUPS:
            return INF
        if G.nodes[u].get("primary_group") == v:
            return INF
    if G.nodes[u].get("tier0", False):
        return INF
    return DEFAULT_EDGE_COST.get(etype, OTHER_EDGE_COST)


def build_graph(nodes: dict, relations) -> nx.DiGraph:
    """由节点属性表和关系列表构建攻击图。

    nodes: {node_id: {"name":..., "ntype":..., 可选 "tier0", "wellknown", "primary_group"}}
    relations: 可迭代的 (u, v, etype)
    同一对 (u, v) 之间的多条关系合并成一条边：要切断 u -> v 必须删掉全部关系，
    所以成本相加（有一条不可删则整条边不可删）。
    """
    G = nx.DiGraph()
    for nid, attrs in nodes.items():
        G.add_node(nid, **attrs)
    for u, v, etype in relations:
        if u == v or u not in G or v not in G:
            continue
        c = relation_cost(G, u, v, etype)
        if G.has_edge(u, v):
            d = G[u][v]
            if etype in d["etypes"]:
                continue
            d["etypes"].append(etype)
            d["cost"] = d["cost"] + c
        else:
            G.add_edge(u, v, etypes=[etype], cost=c)
    return G


@dataclass
class Instance:
    """CMTI 问题实例 (G, c, S, T, w, v)。预算 B 在求解时单独给出。"""

    G: nx.DiGraph
    entries: dict  # 入口 s -> 权重 w_s
    targets: dict  # 目标 t -> 价值 v_t
    name: str = ""
    meta: dict = field(default_factory=dict)

    def validate(self) -> None:
        overlap = set(self.entries) & set(self.targets)
        if overlap:
            raise ValueError(f"入口与目标不能重叠: {sorted(overlap)[:5]}")
        for s, w in self.entries.items():
            if s not in self.G or w <= 0:
                raise ValueError(f"非法入口 {s!r} (w={w})")
        for t, v in self.targets.items():
            if t not in self.G or v <= 0:
                raise ValueError(f"非法目标 {t!r} (v={v})")
        for u, v, c in self.G.edges(data="cost"):
            if c != INF and (c <= 0 or int(c) != c):
                raise ValueError(f"边 {u}->{v} 的成本必须是正整数或 INF，实际为 {c}")
