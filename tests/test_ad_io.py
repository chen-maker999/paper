from __future__ import annotations

import json

import pytest

from adinterdict.ad_generator import ADGeneratorConfig, generate_ad_graph
from adinterdict.adio import load_ad_export
from adinterdict.graphio import load_instance, save_instance
from adinterdict.indexed import IndexedInstance


def test_ad_generator_is_reproducible_and_feasible(tmp_path):
    cfg = ADGeneratorConfig(users=20, groups=5, computers=8, targets=2, seed=7)
    first = generate_ad_graph(cfg)
    second = generate_ad_graph(cfg)
    assert set(first.G.edges) == set(second.G.edges)
    assert first.entries == second.entries
    assert first.targets == second.targets
    assert set(first.entries).isdisjoint(first.targets)
    assert all("cost" in data for _, _, data in first.G.edges(data=True))
    save_instance(first, tmp_path / "generated")
    loaded = load_instance(tmp_path / "generated")
    assert IndexedInstance.from_instance(loaded).risk() == pytest.approx(
        IndexedInstance.from_instance(first).risk()
    )


def test_load_normalized_combined_export(tmp_path):
    payload = {
        "nodes": [
            {"id": "u", "type": "User", "entry_weight": 2},
            {"id": "g", "type": "Group"},
            {"id": "t", "type": "Group", "tier0": True, "target_value": 9},
        ],
        "edges": [
            {"source": "u", "target": "g", "type": "MemberOf"},
            {"source": "g", "target": "t", "type": "GenericAll"},
        ],
    }
    path = tmp_path / "export.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    inst = load_ad_export(path)
    assert inst.entries == {"u": 2.0}
    assert inst.targets == {"t": 9.0}
    assert inst.G.has_edge("u", "g")
    assert inst.G["g"]["t"]["cost"] == 2


def test_load_embedded_relationship_and_overrides(tmp_path):
    payload = {"data": [{
        "ObjectIdentifier": "u",
        "Type": "User",
        "Properties": {"name": "user", "MemberOf": ["g"]},
    }, {"ObjectIdentifier": "g", "Type": "Group"}]}
    path = tmp_path / "users.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    inst = load_ad_export(path, entry_weight={"u": 1}, target_value={"g": 3})
    assert inst.G.has_edge("u", "g")
    assert inst.entries["u"] == 1
    assert inst.targets["g"] == 3


def test_embedded_members_and_computer_admin_lists_are_reversed(tmp_path):
    payload = {"data": [
        {"ObjectIdentifier": "g", "Type": "Group",
         "Properties": {"Members": ["u"]}},
        {"ObjectIdentifier": "c", "Type": "Computer",
         "Properties": {"LocalAdmins": ["u"], "RDPUsers": ["u"]}},
        {"ObjectIdentifier": "u", "Type": "User"},
    ]}
    path = tmp_path / "relations.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    inst = load_ad_export(path)
    assert inst.G.has_edge("u", "g")
    assert inst.G.has_edge("u", "c")
    assert set(inst.G["u"]["c"]["etypes"]) == {"AdminTo", "CanRDP"}


def test_jsonl_and_unknown_relationships_are_retained(tmp_path):
    path = tmp_path / "export.jsonl"
    path.write_text("\n".join([
        json.dumps({"id": "u", "type": "User"}),
        json.dumps({"id": "t", "type": "Group", "target_value": 1}),
        json.dumps({"source": "u", "target": "t", "type": "NewRelation"}),
    ]), encoding="utf-8")
    stats = {}
    inst = load_ad_export(path, entry_weight={"u": 1}, stats=stats)
    assert inst.G.has_edge("u", "t")
    assert "NewRelation" in inst.G["u"]["t"]["etypes"]
    assert stats["unknown_relation_types"] == {"NewRelation": 1}


def test_generator_is_sparse_and_has_fixed_structure():
    inst = generate_ad_graph(ADGeneratorConfig(users=600, groups=60, computers=300, targets=3, seed=2))
    assert inst.name.startswith("ad_u600_g60_c300_t3_s2")
    assert inst.G.number_of_edges() < 20 * inst.G.number_of_nodes()
    assert any(data["cost"] == float("inf") for _, _, data in inst.G.edges(data=True))
    assert any(data.get("tier0") for _, data in inst.G.nodes(data=True))
