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


def test_adsynth_neo4j_jsonl_shape(tmp_path):
    path = tmp_path / "graph.json"
    records = [
        {"id": "1", "labels": ["Base", "User"], "properties": {"name": "u"}, "type": "node"},
        {"id": "2", "labels": ["Base", "Group"], "properties": {"name": "g", "highvalue": True}, "type": "node"},
        {"type": "relationship", "label": "MemberOf", "start": {"id": "1"}, "end": {"id": "2"}},
    ]
    path.write_text("\n".join(json.dumps(record) for record in records), encoding="utf-8")
    inst = load_ad_export(path, entry_weight={"1": 1}, target_value={"2": 2})
    assert inst.G.has_edge("1", "2")
    assert inst.G.nodes["1"]["ntype"] == "User"
    assert inst.G.nodes["2"]["tier0"] is False


def test_explicit_tier0_and_wellknown_annotations(tmp_path):
    path = tmp_path / "graph.json"
    records = [
        {"id": "u", "type": "User", "highvalue": True},
        {"id": "du", "type": "Group"},
        {"id": "t", "type": "Group"},
    ]
    payload = {"nodes": records, "edges": [
        {"source": "u", "target": "du", "type": "MemberOf"},
        {"source": "t", "target": "u", "type": "GenericAll"},
    ]}
    path.write_text(json.dumps(payload), encoding="utf-8")
    inst = load_ad_export(path, tier0=["t"], wellknown={"du": "DOMAIN_USERS"})
    assert inst.G.nodes["u"]["tier0"] is False
    assert inst.G.nodes["t"]["tier0"] is True
    assert inst.G["u"]["du"]["cost"] == float("inf")
    assert inst.G["t"]["u"]["cost"] == float("inf")


def test_dcsync_requires_both_replication_rights(tmp_path):
    path = tmp_path / "replication.json"
    payload = {"nodes": [
        {"id": "u", "type": "User"},
        {"id": "d", "type": "Domain"},
    ], "edges": [
        {"source": "u", "target": "d", "type": "GetChanges"},
        {"source": "u", "target": "d", "type": "GetChangesAll"},
    ]}
    path.write_text(json.dumps(payload), encoding="utf-8")
    inst = load_ad_export(path, entry_weight={"u": 1}, target_value={"d": 1})
    assert inst.G["u"]["d"]["etypes"] == ["DCSync"]
    assert inst.G["u"]["d"]["cost"] == 2

    payload["edges"] = payload["edges"][:1]
    path.write_text(json.dumps(payload), encoding="utf-8")
    incomplete = load_ad_export(path, entry_weight={"u": 1}, target_value={"d": 1})
    assert not incomplete.G.has_edge("u", "d")
    assert incomplete.meta["import_stats"]["incomplete_replication_pairs"] == 1


def test_mapping_file_converter_format(tmp_path):
    from experiments.convert_ad import _mapping
    path = tmp_path / "ids.txt"
    path.write_text("# comment\nu\nu2,2.5\nu3\t3\n", encoding="utf-8")
    assert _mapping(str(path), 1) == {"u": 1.0, "u2": 2.5, "u3": 3.0}
