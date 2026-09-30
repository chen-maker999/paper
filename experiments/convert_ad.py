"""Convert an offline ADSynth/BloodHound JSONL export to graphio CSV.

Entry and target selection is explicit: one ID per line, optionally followed
by a comma or tab and its weight/value. No node names are guessed.

Example::

    python -m experiments.convert_ad \
      --input generated_datasets/vul_1k.json \
      --entries entries.txt --targets targets.txt \
      --out data/vul_1k_s0
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from adinterdict.adio import load_ad_export
from adinterdict.graphio import save_instance


def _mapping(path: str, default: float) -> dict[str, float]:
    """Read ``id`` or ``id,value`` lines; JSON objects are also accepted."""
    text = Path(path).read_text(encoding="utf-8")
    try:
        value = json.loads(text)
        if isinstance(value, dict):
            return {str(k): float(v) for k, v in value.items() if float(v) > 0}
    except json.JSONDecodeError:
        pass
    result: dict[str, float] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        fields = next(csv.reader([line], delimiter="\t"))
        if len(fields) == 1:
            fields = next(csv.reader([line]))
        ident = fields[0].strip()
        if ident:
            result[ident] = float(fields[1]) if len(fields) > 1 and fields[1].strip() else default
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="ADSynth/BloodHound JSON or JSONL export")
    ap.add_argument("--entries", required=True, help="file containing entry IDs and optional weights")
    ap.add_argument("--targets", required=True, help="file containing target IDs and optional values")
    ap.add_argument("--tier0", default=None, help="file containing explicit Tier-0 node IDs")
    ap.add_argument("--wellknown", default=None, help="JSON object mapping node IDs to well-known group labels")
    ap.add_argument("--entry-default", type=float, default=1.0)
    ap.add_argument("--target-default", type=float, default=1.0)
    ap.add_argument("--out", required=True, help="graphio instance directory")
    ap.add_argument("--name", default=None)
    args = ap.parse_args()
    entries = _mapping(args.entries, args.entry_default)
    targets = _mapping(args.targets, args.target_default)
    # ``None`` preserves explicit annotations in the export.  Passing an empty
    # list is meaningful: it deliberately clears all Tier-0 annotations.
    tier0 = list(_mapping(args.tier0, 1.0)) if args.tier0 else None
    wellknown = json.loads(Path(args.wellknown).read_text(encoding="utf-8")) if args.wellknown else {}
    if not isinstance(wellknown, dict):
        raise SystemExit("--wellknown 必须是 JSON 对象")
    if not entries or not targets:
        raise SystemExit("入口和目标文件都必须至少包含一个 ID")
    inst = load_ad_export(args.input, name=args.name,
                          entry_weight=entries, target_value=targets,
                          tier0=tier0, wellknown=wellknown)
    save_instance(inst, args.out)
    print(f"saved {args.out}: n={inst.G.number_of_nodes()} m={inst.G.number_of_edges()} "
          f"entries={len(entries)} targets={len(targets)}")
    print(json.dumps(inst.meta["import_stats"], ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
