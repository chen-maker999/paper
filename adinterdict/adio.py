"""Offline readers for neutralized Active Directory graph exports.

The reader accepts a small, stable interchange format and a tolerant subset of
BloodHound-style JSON exports.  It never connects to a domain and never reads
credentials: it only converts already-exported nodes and relationships into
the project's :class:`~adinterdict.model.Instance` model.

Normalized input format::

    {"nodes": [{"id": "u1", "type": "User", "name": "..."}],
     "edges": [{"source": "u1", "target": "g1", "type": "MemberOf"}],
     "entries": [{"id": "u1", "weight": 1}],
     "targets": [{"id": "g1", "value": 10}]}

For a directory, every ``*.json`` file is read and records with explicit
``source``/``target`` fields are treated as relationships.  Node records may
also contain a list-valued relationship property such as ``MemberOf``.
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
    "hassession": "HasSession", "has_session": "HasSession", "sessions": "HasSession",
    "canrdp": "CanRDP", "rdp": "CanRDP", "hasrdp": "CanRDP",
    "executedcom": "ExecuteDCOM", "dcom": "ExecuteDCOM",
    "canpsremote": "CanPSRemote", "psremote": "CanPSRemote",
    "genericall": "GenericAll", "genericwrite": "GenericWrite",
    "writedacl": "WriteDacl", "writeowner": "WriteOwner", "owns": "Owns",
    "forcechangepassword": "ForceChangePassword", "addmember": "AddMember",
    "contains": "Contains", "gplink": "GPLink",
}


def _key(value: Any) -> str:
    return "".join(ch.lower() for ch in str(value) if ch.isalnum() or ch == "_")


def _relation_type(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    return _RELATION_ALIASES.get(_key(text), text if text in _RELATION_ALIASES.values() else None)


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
    """Return node and edge records without losing combined normalized files."""
    if isinstance(payload, dict) and any(k in payload for k in ("nodes", "edges", "relationships")):
        nodes = payload.get("nodes", [])
        edges = payload.get("edges", payload.get("relationships", []))
        return ([x for x in nodes if isinstance(x, dict)] if isinstance(nodes, list) else [],
                [x for x in edges if isinstance(x, dict)] if isinstance(edges, list) else [])
    records = _records(payload)
    edges, nodes = [], []
    for record in records:
        src = _endpoint(record, ("source", "SourceNodeId", "source_id", "src", "from"))
        dst = _endpoint(record, ("target", "TargetNodeId", "target_id", "dst", "to"))
        rel = _relation_type(record.get("type", record.get("Relationship", record.get("relationship", record.get("Kind")))))
        (edges if src and dst and rel else nodes).append(record)
    return nodes, edges


def _props(record: dict[str, Any]) -> dict[str, Any]:
    p = record.get("properties", record.get("Properties", {}))
    return p if isinstance(p, dict) else {}


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
    value = value or p.get("type") or p.get("objecttype") or "Other"
    text = str(value).lower()
    for known in NODE_TYPES:
        if text == known.lower() or text.endswith(known.lower()):
            return known
    return "Other"


def _endpoint(record: dict[str, Any], names: tuple[str, ...]) -> str | None:
    for key in names:
        if record.get(key) is not None:
            value = record[key]
            if isinstance(value, dict):
                return _id(value)
            return str(value)
    p = _props(record)
    for key in names:
        if p.get(key) is not None:
            return str(p[key])
    return None


def _iter_relation_values(value: Any) -> Iterable[str]:
    values = value if isinstance(value, list) else [value]
    for item in values:
        if isinstance(item, dict):
            ident = _id(item)
            if ident:
                yield ident
        elif item is not None and str(item).strip():
            yield str(item)


def _load_payloads(path: str | os.PathLike[str]) -> list[tuple[str, Any]]:
    p = Path(path)
    files = sorted(p.glob("*.json")) if p.is_dir() else [p]
    if not files:
        raise ADImportError(f"没有找到 JSON 文件: {p}")
    out = []
    for file in files:
        try:
            with file.open(encoding="utf-8") as fh:
                out.append((file.name, json.load(fh)))
        except (OSError, json.JSONDecodeError) as exc:
            raise ADImportError(f"无法读取 {file}: {exc}") from exc
    return out


def load_ad_export(path: str | os.PathLike[str], *, name: str | None = None,
                   entry_weight: dict[str, float] | None = None,
                   target_value: dict[str, float] | None = None) -> Instance:
    """Load a normalized or BloodHound-style offline export.

    Explicit ``entry_weight``/``target_value`` mappings override annotations in
    the files.  Without overrides, node fields ``entry_weight``/``target_value``
    and ``entry``/``target`` are honored.  This keeps asset selection a data
    preparation decision rather than guessing from object names.
    """
    payloads = _load_payloads(path)
    nodes: dict[str, dict[str, Any]] = {}
    relations: list[tuple[str, str, str]] = []
    entries: dict[str, float] = {}
    targets: dict[str, float] = {}

    for filename, payload in payloads:
        node_records, edge_records = _split_payload(payload)
        if isinstance(payload, dict):
            for item in payload.get("entries", []):
                if isinstance(item, dict) and _id(item) is not None:
                    value = item.get("weight", item.get("entry_weight", 1))
                    if float(value) > 0:
                        entries[str(_id(item))] = float(value)
            for item in payload.get("targets", []):
                if isinstance(item, dict) and _id(item) is not None:
                    value = item.get("value", item.get("target_value", 1))
                    if float(value) > 0:
                        targets[str(_id(item))] = float(value)
        for rec in edge_records + node_records:
            # Explicit edge records.
            src = _endpoint(rec, ("source", "SourceNodeId", "source_id", "src", "from"))
            dst = _endpoint(rec, ("target", "TargetNodeId", "target_id", "dst", "to"))
            etype = _relation_type(rec.get("type", rec.get("Relationship", rec.get("relationship", rec.get("Kind")))))
            if src and dst and etype:
                relations.append((src, dst, etype))
                continue

            ident = _id(rec)
            if not ident:
                continue
            p = _props(rec)
            attrs = {
                "name": str(rec.get("name", rec.get("Name", p.get("name", ident)))),
                "ntype": _node_type(rec),
                "tier0": bool(rec.get("tier0", p.get("tier0", False))),
                "wellknown": str(rec.get("wellknown", p.get("wellknown", ""))).upper(),
            }
            primary = rec.get("primary_group", p.get("primary_group"))
            if primary:
                attrs["primary_group"] = str(primary)
            nodes.setdefault(ident, {}).update(attrs)

            ew = rec.get("entry_weight", p.get("entry_weight", rec.get("entry")))
            tv = rec.get("target_value", p.get("target_value", rec.get("target")))
            if ew is not None and float(ew) > 0:
                entries[ident] = float(ew)
            if tv is not None and float(tv) > 0:
                targets[ident] = float(tv)
            for key, value in {**rec, **p}.items():
                rel = _relation_type(key)
                if not rel:
                    continue
                for other in _iter_relation_values(value):
                    relations.append((ident, other, rel))

    # Add endpoint-only nodes with neutral attributes.
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
    inst = Instance(graph, entries, targets, name=name or Path(path).stem,
                    meta={"source": str(path), "format": "offline-ad-export"})
    inst.validate()
    return inst


# Short aliases used by scripts and notebooks.
load_bloodhound = load_ad_export
read_ad_export = load_ad_export
