"""Deterministic, offline AD-style graph generator for algorithm experiments.

The generated graph is an abstract benchmark fixture. It does not contact a
directory, create accounts, or emit credentials. The generator keeps expected
degree bounded as the graph grows, uses branch-local group hierarchies, and
includes structural/default relations so the reduction and hub algorithms see
the mixed constraints they are designed for.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass

from .model import Instance, build_graph


@dataclass(frozen=True)
class ADGeneratorConfig:
    users: int = 100
    groups: int = 20
    computers: int = 40
    targets: int = 3
    entry_fraction: float = 0.10
    membership_degree: float = 1.5
    admin_degree: float = 1.5
    session_degree: float = 1.0
    remote_degree: float = 0.8
    parent_degree: float = 1.7
    gate_count: int = 3
    cross_branch_fraction: float = 0.02
    ou_count: int = 4
    seed: int = 0

    # Legacy density knobs are accepted for callers of the first generator.
    # When supplied, they are converted to an expected degree.
    membership_fraction: float | None = None
    admin_fraction: float | None = None
    session_fraction: float | None = None
    remote_fraction: float | None = None
    group_edge_fraction: float | None = None

    def validate(self) -> None:
        for field in ("users", "groups", "computers", "targets", "ou_count"):
            if getattr(self, field) < 0:
                raise ValueError(f"{field} 不能为负数")
        if self.groups == 0 or self.computers == 0 or self.targets == 0:
            raise ValueError("groups、computers、targets 至少为 1")
        if not 0 <= self.entry_fraction <= 1:
            raise ValueError("entry_fraction 必须位于 [0, 1]")
        for field in ("membership_degree", "admin_degree", "session_degree",
                      "remote_degree", "parent_degree", "cross_branch_fraction"):
            if getattr(self, field) < 0:
                raise ValueError(f"{field} 不能为负数")
        if self.ou_count == 0:
            raise ValueError("ou_count 至少为 1")
        if self.gate_count < 2:
            raise ValueError("gate_count 至少为 2")
        for field in ("membership_fraction", "admin_fraction", "session_fraction",
                      "remote_fraction", "group_edge_fraction"):
            value = getattr(self, field)
            if value is not None and not 0 <= value <= 1:
                raise ValueError(f"{field} 必须位于 [0, 1]")


def _poisson(rng: random.Random, mean: float) -> int:
    """Small dependency-free Poisson sampler used for bounded degree."""
    if mean <= 0:
        return 0
    limit = math.exp(-mean)
    product = 1.0
    count = 0
    while product > limit:
        count += 1
        product *= rng.random()
    return count - 1


def _degree(cfg: ADGeneratorConfig, legacy: float | None, pool_size: int,
            default: float) -> float:
    return legacy * pool_size if legacy is not None else default


def generate_ad_graph(config: ADGeneratorConfig | None = None, *, name: str | None = None) -> Instance:
    """Generate a reproducible abstract AD-style attack graph.

    Group edges point only toward a higher hierarchy level within a branch,
    and each branch has a small gate set leading to one target. Random edges
    are sampled per source node, so edge count grows approximately linearly.
    """
    cfg = config or ADGeneratorConfig()
    cfg.validate()
    rng = random.Random(cfg.seed)

    domain = "domain-000"
    ous = [f"ou-{i:03d}" for i in range(max(1, cfg.ou_count))]
    default_group = "group-default-users"
    users = [f"user-{i:05d}" for i in range(cfg.users)]
    groups = [f"group-{i:04d}" for i in range(cfg.groups)]
    computers = [f"computer-{i:05d}" for i in range(cfg.computers)]
    targets = [f"tier0-{i:03d}" for i in range(cfg.targets)]

    attrs = {domain: {"name": domain, "ntype": "Domain"}}
    attrs.update({ou: {"name": ou, "ntype": "OU"} for ou in ous})
    attrs.update({u: {"name": u, "ntype": "User"} for u in users})
    attrs.update({g: {"name": g, "ntype": "Group"} for g in groups})
    attrs[default_group] = {"name": default_group, "ntype": "Group",
                            "wellknown": "DOMAIN_USERS"}
    attrs.update({c: {"name": c, "ntype": "Computer"} for c in computers})
    attrs.update({t: {"name": t, "ntype": "Group", "tier0": True} for t in targets})

    relations: list[tuple[str, str, str]] = []

    def add(u: str, v: str, rel: str) -> None:
        if u != v:
            relations.append((u, v, rel))

    # Structural containment is fixed and provides a sparse hierarchy.
    add(domain, default_group, "Contains")
    for i, ou in enumerate(ous):
        add(domain, ou, "Contains")
        if i:
            add(ous[i - 1], ou, "Contains")
    for node in [*users, *groups, *computers, *targets]:
        add(rng.choice(ous), node, "Contains")

    branches = max(1, cfg.targets)
    group_branch = {g: i % branches for i, g in enumerate(groups)}
    # Keep several groups at the top of each branch so a target has redundant
    # gate paths rather than one universal incoming edge.
    levels = {g: (i // branches) // 2 for i, g in enumerate(groups)}
    branch_groups = {b: [g for g in groups if group_branch[g] == b] for b in range(branches)}

    # Default membership is deliberately fixed by model.relation_cost.
    for user in users:
        add(user, default_group, "MemberOf")

    # Every user starts in one branch. Additional memberships stay sparse.
    membership_degree = _degree(cfg, cfg.membership_fraction, max(1, len(groups)), cfg.membership_degree)
    user_branch = {u: i % branches for i, u in enumerate(users)}
    for user in users:
        choices = branch_groups[user_branch[user]] or groups
        count = max(1, _poisson(rng, membership_degree))
        for group in rng.sample(choices, min(count, len(choices))):
            add(user, group, "MemberOf")

    # Hierarchical nesting: only higher levels are eligible parents.
    parent_degree = _degree(cfg, cfg.group_edge_fraction, max(1, len(groups)), cfg.parent_degree)
    for group in groups:
        candidates = [g for g in groups if levels[g] > levels[group]
                      and (group_branch[g] == group_branch[group]
                           or rng.random() < cfg.cross_branch_fraction)]
        if candidates:
            count = min(len(candidates), _poisson(rng, parent_degree))
            for parent in rng.sample(candidates, count):
                add(group, parent, "MemberOf")

    # Several highest-level groups in each branch control that branch's target.
    for branch in range(branches):
        branch_levels = branch_groups[branch]
        if branch_levels:
            top = max(levels[g] for g in branch_levels)
            gates = [g for g in branch_levels if levels[g] == top]
            if len(gates) < cfg.gate_count:
                lower = sorted((g for g in branch_levels if g not in gates),
                               key=lambda g: levels[g], reverse=True)
                gates.extend(lower[:cfg.gate_count - len(gates)])
            for gate in gates[:cfg.gate_count]:
                add(gate, targets[branch % len(targets)], "GenericAll")

    # Tier-0 internal relations are fixed because their source is Tier-0.
    for i in range(1, len(targets)):
        add(targets[i], targets[0], "MemberOf")

    # Per-source sampling keeps user/computer relations sparse at every scale.
    admin_degree = _degree(cfg, cfg.admin_fraction, max(1, len(computers)), cfg.admin_degree)
    remote_degree = _degree(cfg, cfg.remote_fraction, max(1, len(computers)), cfg.remote_degree)
    for user in users:
        for computer in rng.sample(computers, min(_poisson(rng, admin_degree), len(computers))):
            add(user, computer, "AdminTo")
        for computer in rng.sample(computers, min(_poisson(rng, remote_degree), len(computers))):
            add(user, computer, "CanRDP")

    session_degree = _degree(cfg, cfg.session_fraction, max(1, len(users)), cfg.session_degree)
    for computer in computers:
        for user in rng.sample(users, min(_poisson(rng, session_degree), len(users))):
            add(computer, user, "HasSession")

    entry_count = max(1, round(cfg.entry_fraction * len(users))) if users else 0
    entries = {u: 1.0 for u in rng.sample(users, min(entry_count, len(users)))}
    target_values = {t: float(5 + 5 * (i % 3)) for i, t in enumerate(targets)}
    graph = build_graph(attrs, relations)
    inst = Instance(graph, entries, target_values,
                    name=name or f"ad_u{cfg.users}_g{cfg.groups}_c{cfg.computers}_t{cfg.targets}_s{cfg.seed}",
                    meta={"family": "abstract_ad", "seed": cfg.seed,
                          "config": cfg.__dict__.copy()})
    inst.validate()
    return inst


def generate(config: ADGeneratorConfig | None = None, *, name: str | None = None) -> Instance:
    """Short alias for :func:`generate_ad_graph`."""
    return generate_ad_graph(config, name=name)
