"""Generate and annotate the six published ADSynth presets offline.

    python -m experiments.prepare_adsynth --adsynth /path/to/ADSynth --out data

The external repository is pinned by commit in each instance's provenance.
Each run uses a fresh subprocess because ADSynth has module-level graph state.
Only the classic ``generate`` path is invoked; no Neo4j/server is required.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import re
import subprocess
import sys
from pathlib import Path

from adinterdict.adio import load_ad_export
from adinterdict.graphio import load_instance, save_instance


PRESETS = ("vul_1k", "vul_5k", "vul_10k", "secure_1k", "secure_5k", "secure_10k")
DEFAULT_GROUP_NAMES = {
    "DOMAIN USERS": "DOMAIN_USERS", "DOMAIN COMPUTERS": "DOMAIN_COMPUTERS",
    "DOMAIN CONTROLLERS": "DOMAIN_CONTROLLERS", "EVERYONE": "EVERYONE",
    "AUTHENTICATED USERS": "AUTHENTICATED_USERS", "USERS": "BUILTIN_USERS",
}
TARGET_NAMES = {"DOMAIN ADMINS", "ENTERPRISE ADMINS", "ADMINISTRATORS"}
SELECTION_RULES = {
    "entries": "10% of enabled, non-admin User nodes whose DN contains OU=T2 ENABLED USERS; weight 1",
    "targets": "Domain + Domain Admins + Enterprise Admins + Administrators; value 10 each",
    "tier0": "targets; DN component OU=T0 ... or OU=DOMAIN CONTROLLERS; explicit input list, highvalue ignored",
    "wellknown": DEFAULT_GROUP_NAMES,
    "replication": "GetChanges AND GetChangesAll on same principal/domain -> DCSync, cost 2",
}

# UUIDs and timestamps are seeded independently of ADSynth's topology RNG.
# PYTHONHASHSEED is set on the fresh subprocess for upstream set iteration.
RUNNER = r'''
import json, random, sys, uuid
from pathlib import Path
root, config, output, seed = sys.argv[1:]
sys.path.insert(0, root)
uuid_rng = random.Random("adsynth-uuid-" + seed)
uuid.uuid4 = lambda: uuid.UUID(int=uuid_rng.getrandbits(128), version=4)
from adsynth.ADSynth import MainMenu
menu = MainMenu()
menu.parameters = json.loads(Path(config).read_text())
menu.current_time = 1790726400  # 2026-09-30 00:00:00 UTC
menu.do_generate("")
from adsynth.DATABASE import NODES, EDGES
with Path(output).open("w", encoding="utf-8") as f:
    for node in NODES:
        node["type"] = "node"
        f.write(json.dumps(node, separators=(",", ":")) + "\n")
    for edge in EDGES:
        f.write(json.dumps(edge, separators=(",", ":")) + "\n")
'''


def _bool(value) -> bool:
    if isinstance(value, list):
        return bool(value) and _bool(value[0])
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes"}
    return bool(value)


def annotations(path: Path, seed: int) -> dict:
    """Materialize explicit ID lists using the documented selection policy."""
    nodes = [obj for line in path.read_text().splitlines()
             if (obj := json.loads(line)).get("type") == "node"]
    candidates, targets, tier0, wellknown = [], {}, [], {}
    for node in nodes:
        ident = str(node["id"])
        p = node["properties"]
        labels = node.get("labels", [])
        short_name = str(p.get("name", "")).split("@", 1)[0].upper()
        dn = str(p.get("distinguishedname", "")).upper()
        if "User" in labels and "OU=T2 ENABLED USERS," in dn and _bool(p.get("enabled")) and not _bool(p.get("admincount")):
            candidates.append(ident)
        target = "Domain" in labels or ("Group" in labels and short_name in TARGET_NAMES)
        if target:
            targets[ident] = 10.0
        if target or re.search(r"(?:^|,)OU=T0 [^,]+(?:,|$)", dn) or "OU=DOMAIN CONTROLLERS," in dn:
            tier0.append(ident)
        if "Group" in labels and short_name in DEFAULT_GROUP_NAMES:
            wellknown[ident] = DEFAULT_GROUP_NAMES[short_name]
    if len(targets) != 4 or not candidates:
        raise ValueError(f"unexpected ADSynth selection: {len(targets)} targets, {len(candidates)} entry candidates")
    rng = random.Random(seed)
    entries = {ident: 1.0 for ident in sorted(rng.sample(sorted(candidates), max(1, round(len(candidates) * 0.1))))}
    missing_groups = set(DEFAULT_GROUP_NAMES.values()) - set(wellknown.values())
    if missing_groups:
        raise ValueError(f"missing default groups: {sorted(missing_groups)}")
    return {"entries": entries, "targets": targets, "tier0": sorted(tier0), "wellknown": wellknown,
            "entry_candidates": len(candidates)}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare_one(root: Path, out: Path, preset: str, seed: int, commit: str) -> dict:
    name = f"{preset}_s{seed}"
    dest = out / name
    dest.mkdir(parents=True, exist_ok=True)
    audit = dest / "annotations"
    audit.mkdir(exist_ok=True)
    source = out / "_sources" / name
    source.mkdir(parents=True, exist_ok=True)
    config = json.loads((root / "adsynth" / "experiment_params" / f"{preset}.json").read_text())
    config["seed"] = seed
    config_path = audit / "config.json"
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    raw = source / "graph.jsonl"
    env = {**os.environ, "PYTHONHASHSEED": "0"}
    with (source / "generation.log").open("w") as log:
        subprocess.run([sys.executable, "-c", RUNNER, str(root), str(config_path), str(raw), str(seed)],
                       cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    selected = annotations(raw, seed)
    for field in ("entries", "targets", "wellknown"):
        (audit / f"{field}.json").write_text(json.dumps(selected[field], indent=2) + "\n")
    (audit / "tier0.txt").write_text("\n".join(selected["tier0"]) + "\n")
    inst = load_ad_export(raw, name=name, entry_weight=selected["entries"], target_value=selected["targets"],
                          tier0=selected["tier0"], wellknown=selected["wellknown"])
    # Check the cost policy on every membership, rather than a sampled subset.
    fixed_default = 0
    for u, v, attrs in inst.G.edges(data=True):
        if "MemberOf" in attrs["etypes"] and v in selected["wellknown"]:
            assert attrs["cost"] == float("inf"), (u, v)
            fixed_default += 1
        if u in selected["tier0"]:
            assert attrs["cost"] == float("inf"), (u, v)
    inst.meta.update({"source": f"_sources/{name}/graph.jsonl", "preset": preset, "seed": seed,
                      "adsynth_repository": "https://github.com/AUCyberLab/ADSynth",
                      "adsynth_commit": commit, "raw_sha256": _sha(raw), "selection_rules": SELECTION_RULES,
                      "entry_candidates": selected["entry_candidates"], "fixed_default_memberships": fixed_default})
    save_instance(inst, dest)
    reloaded = load_instance(dest)
    assert set(reloaded.G) == set(inst.G)
    assert reloaded.entries == inst.entries and reloaded.targets == inst.targets
    assert all(reloaded.G[u][v]["cost"] == attrs["cost"] for u, v, attrs in inst.G.edges(data=True))
    return {"dataset": name, "preset": preset, "seed": seed, "n": len(inst.G), "m": inst.G.number_of_edges(),
            "relations": inst.meta["import_stats"]["relation_records"], "entries": len(inst.entries),
            "targets": len(inst.targets), "tier0": len(selected["tier0"]), "wellknown": len(selected["wellknown"]),
            "fixed_default_memberships": fixed_default,
            "fixed_edges": sum(attrs["cost"] == float("inf") for _, _, attrs in inst.G.edges(data=True)),
            "edges_sha256": _sha(dest / "edges.csv"), "nodes_sha256": _sha(dest / "nodes.csv")}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adsynth", required=True, type=Path)
    ap.add_argument("--out", default="data", type=Path)
    ap.add_argument("--presets", default=",".join(PRESETS))
    ap.add_argument("--seeds", default="1,2,3,4,5", help="positive seeds; upstream seed 0 is not deterministic")
    a = ap.parse_args()
    root, out = a.adsynth.resolve(), a.out.resolve()
    presets, seeds = a.presets.split(","), [int(x) for x in a.seeds.split(",")]
    if any(p not in PRESETS for p in presets) or any(seed <= 0 for seed in seeds):
        ap.error("use one of the six documented presets and positive seeds")
    commit = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    rows = []
    for preset in presets:
        for seed in seeds:
            row = prepare_one(root, out, preset, seed, commit)
            rows.append(row)
            print(f"{row['dataset']}: n={row['n']} m={row['m']} entries={row['entries']} targets={row['targets']} "
                  f"tier0={row['tier0']} default_memberships={row['fixed_default_memberships']}", flush=True)
            with (out / "manifest.csv").open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(row))
                writer.writeheader()
                writer.writerows(rows)


if __name__ == "__main__":
    main()
