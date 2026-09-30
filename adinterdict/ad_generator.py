"""Deterministic, offline AD-style graph generator for algorithm experiments.

The generator creates abstract users, groups, computers, and Tier-0 assets.  It
does not contact a directory, create accounts, or emit credentials.  The
result is the same neutral :class:`~adinterdict.model.Instance` used by every
other experiment in this repository.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

import networkx as nx

from .model import Instance, build_graph


@dataclass(frozen=True)
class ADGeneratorConfig:
    users: int = 100
    groups: int = 20
    computers: int = 40
    targets: int = 3
    entry_fraction: float = 0.10
    membership_fraction: float = 0.08
    admin_fraction: float = 0.04
    session_fraction: float = 0.04
    remote_fraction: float = 0.03
    group_edge_fraction: float = 0.15
    seed: int = 0

    def validate(self) -> None:
        for field in ("users", "groups", "computers", "targets"):
            if getattr(self, field) < 0:
                raise ValueError(f"{field} 不能为负数")
        if self.groups == 0 or self.computers == 0 or self.targets == 0:
            raise ValueError("groups、computers、targets 至少为 1")
        for field in ("entry_fraction", "membership_fraction", "admin_fraction",
                      "session_fraction", "remote_fraction", "group_edge_fraction"):
            value = getattr(self, field)
            if not 0 <= value <= 1:
                raise ValueError(f"{field} 必须位于 [0, 1]")


def generate_ad_graph(config: ADGeneratorConfig | None = None, *, name: str | None = None) -> Instance:
    """Generate a reproducible abstract AD-style attack graph.

    Every entry has a short backbone to a target through a group, so generated
    instances are useful for smoke tests even when the random edge rates are
    small.  Additional relationships are sampled independently from the
    configured rates.
    """
    cfg = config or ADGeneratorConfig()
    cfg.validate()
    rng = random.Random(cfg.seed)
    users = [f"user-{i:05d}" for i in range(cfg.users)]
    groups = [f"group-{i:04d}" for i in range(cfg.groups)]
    computers = [f"computer-{i:05d}" for i in range(cfg.computers)]
    targets = [f"tier0-{i:03d}" for i in range(cfg.targets)]
    all_nodes = users + groups + computers + targets
    attrs = {u: {"name": u, "ntype": "User"} for u in users}
    attrs.update({g: {"name": g, "ntype": "Group"} for g in groups})
    attrs.update({c: {"name": c, "ntype": "Computer"} for c in computers})
    attrs.update({t: {"name": t, "ntype": "Group", "tier0": True} for t in targets})
    # A well-known default group is present to exercise fixed-membership rules.
    attrs[groups[0]]["wellknown"] = "DOMAIN_USERS"
    relations: list[tuple[str, str, str]] = []

    def add(u: str, v: str, rel: str) -> None:
        if u != v:
            relations.append((u, v, rel))

    # Backbone: entries -> ordinary group -> target.  The target-side relation
    # is intentionally a normal permission edge and receives the configured
    # model cost; no real directory semantics are required to run algorithms.
    for i, user in enumerate(users):
        group = groups[i % len(groups)]
        target = targets[i % len(targets)]
        add(user, group, "MemberOf")
        add(group, target, "GenericAll")

    for user in users:
        for group in groups:
            if rng.random() < cfg.membership_fraction:
                add(user, group, "MemberOf")
    for left in groups:
        for right in groups:
            if left != right and rng.random() < cfg.group_edge_fraction:
                add(left, right, "MemberOf")
    for user in users:
        for computer in computers:
            if rng.random() < cfg.admin_fraction:
                add(user, computer, "AdminTo")
            if rng.random() < cfg.remote_fraction:
                add(user, computer, "CanRDP")
    for computer in computers:
        for user in users:
            if rng.random() < cfg.session_fraction:
                add(computer, user, "HasSession")

    entry_count = max(1, round(cfg.entry_fraction * len(users))) if users else 0
    entries = {u: 1.0 for u in rng.sample(users, min(entry_count, len(users)))}
    target_values = {t: float(5 + 5 * (i % 3)) for i, t in enumerate(targets)}
    graph = build_graph(attrs, relations)
    # Keep isolated generated nodes in the graph as well.
    graph.add_nodes_from((n, attrs[n]) for n in all_nodes if n not in graph)
    inst = Instance(graph, entries, target_values,
                    name=name or f"ad_s{cfg.seed}",
                    meta={"family": "abstract_ad", "seed": cfg.seed,
                          "config": cfg.__dict__.copy()})
    inst.validate()
    return inst


def generate(config: ADGeneratorConfig | None = None, *, name: str | None = None) -> Instance:
    """Short alias for :func:`generate_ad_graph`."""
    return generate_ad_graph(config, name=name)
