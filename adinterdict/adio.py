"""Offline readers for normalized and BloodHound-style AD graph exports.

The reader only consumes files already exported by another tool. It never
connects to a domain, handles credentials, or performs directory changes.
Unknown relationship types are retained with the model's fallback cost and
reported in ``Instance.meta['import_stats']`` instead of being silently lost.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterable

from .model import NODE_TYPES, Instance, build_graph


class ADImportError(ValueError):
    """Raised when an exported graph is malformed or ambiguous."""


_RELATION_ALIASES = {
    "memberof": "MemberOf", "member_of": "MemberOf", "members": "MemberOf",
    "adminto": "AdminTo", "admin_to": "AdminTo", "localadmin": "AdminTo",
    "localadmins": "AdminTo", "admins": "AdminTo",
    "hassession": "HasSession", "has_session": "HasSession", "sessions": "HasSession",
    "canrdp": "CanRDP", "rdp": "CanRDP", "hasrdp": "CanRDP", "rdpusers": "CanRDP",
    "remoteinteractivelogonright": "CanRDP",
    "executedcom": "ExecuteDCOM", "dcom": "ExecuteDCOM",
    "canpsremote": "CanPSRemote", "psremote": "CanPSRemote",
    "genericall": "GenericAll", "genericwrite": "GenericWrite",
    "writedacl": "WriteDacl", "writeowner": "WriteOwner", "owns": "Owns",
    "forcechangepassword": "ForceChangePassword", "addmember": "AddMember",
    "addself": "AddSelf", "allextendedrights": "AllExtendedRights",
    "readlapsPassword".lower(): "ReadLAPSPassword", "readgmsapassword": "ReadGMSAPassword",
    "allowedtodelegate": "AllowedToDelegate", "allowedtoact": "AllowedToAct",
    "addallowedtoact": "AddAllowedToAct", "writesPN".lower(): "WriteSPN",
    "addkeycredentiallink": "AddKeyCredentialLink", "sqladmin": "SQLAdmin",
    "contains": "Contains", "gplink": "GPLink",
}
_METADATA_KEYS = {
    "id", "objectid", "objectidentifier", "object_identifier", "name", "type", "kind",
    "properties", "source", "target", "sourcenodeid", "targetnodeid", "source_id",
    "target_id", "src", "dst", "from", "to", "relationship", "relationshiptype",
    "rel", "entry", "entry_weight", "target_value", "target", "tier0", "wellknown",
    "primary_group", "primarygroupsid",
}


def _key(value: Any) -> str:
    return "".join(ch.lower() for ch in str(value) if ch.isalnum() or ch == "_")


def _relation_type(value: Any, *, preserve_unknown: bool = False) -> str | None:
    if value is None:
        return None
    text = str(value)
    result = _RELATION_ALIASES.get(_key(text))
    if result:
        return result
    if text in _RELATION_ALIASES.values():
        return text
    return text if preserve_unknown and text.strip() else None


def _looks_like_relation_key(key: str) -> bool:
    k = _key(key)
    if k in _RELATION_ALIASES:
        return True
    return k.startswith(("allow", "admin", "add", "can", "has", "is", "read",
                         "write", "execute", "force", "own", "sql", "remote"))


def _props(record: dict[str, Any]) -> dict[str, Any]:
    value = record.get("properties", record.get("Properties", {}))
    return value if isinstance(value, dict) else {}


def _id(record: dict[str, Any]) -> str | None:
    for key in ("id", "Id", "objectid", "ObjectIdentifier", "object_identifier",
                "node_id", "NodeId", "name", "Name"):
        if record.get(key) is not None:
            return str(record[key])
    p = _props(record)
    for key in ("id", "objectid", "ObjectIdentifier", "name"):
        if p.get(key) is not None:
            return str(p[key])
    return None


def _node_type(record: dict[str, Any]) -> str:
    value = record.get("type", record.get("Type", record.get("kind", record.get("Kind"))))
    p = _props(record)
    text = str(value or p.get("type") or p.get("objecttype") or "Other").lower()
    for known in NODE_TYPES:
        if text == known.lower() or text.endswith(known.lower()):
            return known
    return "Other"


def _endpoint(record: dict[str, Any], names: tuple[str, ...]) -> str | None:
    for key in names:
        if record.get(key) is not None:
            value = record[key]
            return _id(value) if isinstance(value, dict) else str(value)
    p = _props(record)
    for key in names:
        if p.get(key) is not None:
            return str(p[key])
    return None


def _raw_relation(record: dict[str, Any]) -> Any:
    for key in ("type", "Relationship", "relationship", "RelationshipType", "Kind", "rel"):
        if record.get(key) is not None:
            return record[key]
    return None


def _records(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [x for x in payload if isinstance(x, dict)]
    if not isinstance(payload, dict):
        raise ADImportError("JSON 顶层必须是对象或对象数组")
    for field in ("nodes", "edges", "relationships", "data"):
        value = payload.get(field)
        if isinstance(value, list):
            return [x for x in value if isinstance(x, dict)]
    return [payload]


def _split_payload(payload: Any) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if isinstance(payload, dict) and any(k in payload for k in ("nodes", "edges", "relationships")):
        nodes = payload.get("nodes", [])
        edges = payload.get("edges", payload.get("relationships", []))
        return ([x for x in nodes if isinstance(x, dict)] if isinstance(nodes, list) else [],
                [x for x in edges if isinstance(x, dict)] if isinstance(edges, list) else [])
    nodes, edges = [], []
    for record in _records(payload):
        src = _endpoint(record, ("source", "SourceNodeId", "source_id", "src", "from"))
        dst = _endpoint(record, ("target", "TargetNodeId", "target_id", "dst", "to"))
        (edges if src and dst and _raw_relation(record) is not None else nodes).append(record)
    return nodes, edges


def _iter_relation_values(value: Any) -> Iterable[str]:
    values = value if isinstance(value, list) else [value]
    for item in values:
        if isinstance(item, dict):
            ident = _id(item)
            if ident:
                yield ident
        elif item is not None and str(item).strip():
            yield str(item)


def _embedded_direction(key: str, relation: str, owner_type: str) -> bool:
    """Return whether an embedded list points back to its owning object."""
    k = _key(key)
    if k in {"members", "localadmin", "localadmins", "admins", "rdpusers",
             "remoteinteractivelogonright"}:
        return True
    if relation in {"AdminTo", "CanRDP"} and owner_type == "Computer":
        return True
    if relation == "HasSession" and owner_type == "User":
        return True
    return False


def _load_payloads(path: str | os.PathLike[str]) -> list[tuple[str, Any]]:
    p = Path(path)
    files = sorted(p.glob("*.json")) + sorted(p.glob("*.jsonl")) if p.is_dir() else [p]
    if not files:
        raise ADImportError(f"没有找到 JSON 文件: {p}")
    out: list[tuple[str, Any]] = []
    for file in files:
        try:
            text = file.read_text(encoding="utf-8")
        except OSError as exc:
            raise ADImportError(f"无法读取 {file}: {exc}") from exc
        try:
            out.append((file.name, json.loads(text)))
            continue
        except json.JSONDecodeError:
            pass
        # ADSynth/BloodHound exports are often JSON Lines despite a .json suffix.
        for line_no, line in enumerate(text.splitlines(), 1):
            if not line.strip():
                continue
            try:
                out.append((f"{file.name}:{line_no}", json.loads(line)))
            except json.JSONDecodeError as exc:
                raise ADImportError(f"无法解析 {file}:{line_no}: {exc}") from exc
    return out


def load_ad_export(path: str | os.PathLike[str], *, name: str | None = None,
                   entry_weight: dict[str, float] | None = None,
                   target_value: dict[str, float] | None = None,
                   stats: dict[str, Any] | None = None) -> Instance:
    """Load a normalized or BloodHound-style offline export.

    Explicit entry/target mappings override annotations in the files. Unknown
    relationship names are retained with fallback cost 2 and listed in import
    statistics. Embedded ``Members`` and computer-side admin/RDP lists are
    reversed to match the graph convention ``subject -> controlled object``.
    """
    payloads = _load_payloads(path)
    nodes: dict[str, dict[str, Any]] = {}
    relations: list[tuple[str, str, str]] = []
    entries: dict[str, float] = {}
    targets: dict[str, float] = {}
    report: dict[str, Any] = {"files": len(payloads), "node_records": 0,
                              "relation_records": 0, "skipped_records": 0,
                              "unknown_relation_types": {},
                              "skipped_relation_types": {}}

    def record_unknown(value: Any) -> None:
        key = str(value)
        unknown = report["unknown_relation_types"]
        unknown[key] = unknown.get(key, 0) + 1

    def add_relation(u: str, v: str, rel: str) -> None:
        relations.append((u, v, rel))
        report["relation_records"] += 1

    def record_skipped(value: Any) -> None:
        key = str(value)
        skipped = report["skipped_relation_types"]
        skipped[key] = skipped.get(key, 0) + 1

    for _, payload in payloads:
        node_records, edge_records = _split_payload(payload)
        if isinstance(payload, dict):
            for item in payload.get("entries", []):
                if isinstance(item, dict) and _id(item) is not None and float(item.get("weight", item.get("entry_weight", 1))) > 0:
                    entries[str(_id(item))] = float(item.get("weight", item.get("entry_weight", 1)))
            for item in payload.get("targets", []):
                if isinstance(item, dict) and _id(item) is not None and float(item.get("value", item.get("target_value", 1))) > 0:
                    targets[str(_id(item))] = float(item.get("value", item.get("target_value", 1)))

        for rec in edge_records:
            src = _endpoint(rec, ("source", "SourceNodeId", "source_id", "src", "from"))
            dst = _endpoint(rec, ("target", "TargetNodeId", "target_id", "dst", "to"))
            raw = _raw_relation(rec)
            rel = _relation_type(raw, preserve_unknown=True)
            if src and dst and rel:
                add_relation(src, dst, rel)
                if _key(raw) not in _RELATION_ALIASES and str(raw) not in _RELATION_ALIASES.values():
                    record_unknown(raw)
            else:
                report["skipped_records"] += 1

        for rec in node_records:
            ident = _id(rec)
            if not ident:
                report["skipped_records"] += 1
                continue
            report["node_records"] += 1
            p = _props(rec)
            owner_type = _node_type(rec)
            attrs = {
                "name": str(rec.get("name", rec.get("Name", p.get("name", ident)))),
                "ntype": owner_type,
                "tier0": bool(rec.get("tier0", p.get("tier0", False))),
                "wellknown": str(rec.get("wellknown", p.get("wellknown", ""))).upper(),
            }
            primary = rec.get("primary_group", p.get("primary_group", p.get("primarygroupsid")))
            if primary:
                attrs["primary_group"] = str(primary)
            nodes.setdefault(ident, {}).update(attrs)
            ew = rec.get("entry_weight", p.get("entry_weight", rec.get("entry")))
            tv = rec.get("target_value", p.get("target_value", rec.get("target")))
            if ew is not None and float(ew) > 0:
                entries[ident] = float(ew)
            if tv is not None and float(tv) > 0:
                targets[ident] = float(tv)

            for key, value in {**p, **rec}.items():
                if _key(key) in _METADATA_KEYS or key in {"Properties", "properties"}:
                    continue
                rel = _relation_type(key)
                if rel is None and _looks_like_relation_key(key):
                    rel = _relation_type(key, preserve_unknown=True)
                    record_unknown(key)
                if not rel:
                    if isinstance(value, list) and value:
                        record_skipped(key)
                    continue
                reverse = _embedded_direction(key, rel, owner_type)
                for other in _iter_relation_values(value):
                    add_relation(other, ident, rel) if reverse else add_relation(ident, other, rel)

    for u, v, _ in relations:
        nodes.setdefault(u, {"name": u, "ntype": "Other"})
        nodes.setdefault(v, {"name": v, "ntype": "Other"})
    if entry_weight:
        entries.update({str(k): float(v) for k, v in entry_weight.items() if float(v) > 0})
    if target_value:
        targets.update({str(k): float(v) for k, v in target_value.items() if float(v) > 0})
    missing = (set(entries) | set(targets)) - set(nodes)
    if missing:
        raise ADImportError(f"入口/目标引用了不存在的节点: {sorted(missing)[:5]}")
    graph = build_graph(nodes, relations)
    report["nodes"] = len(nodes)
    report["relations"] = len(relations)
    inst = Instance(graph, entries, targets, name=name or Path(path).stem,
                    meta={"source": str(path), "format": "offline-ad-export",
                          "import_stats": report})
    if stats is not None:
        stats.update(report)
    inst.validate()
    return inst


load_bloodhound = load_ad_export
read_ad_export = load_ad_export
