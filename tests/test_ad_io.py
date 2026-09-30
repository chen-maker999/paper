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
