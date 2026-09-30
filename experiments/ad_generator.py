"""CLI for generating offline abstract AD-style graph instances.

Example::

    python -m experiments.ad_generator --users 1000 --groups 80 \
        --computers 300 --targets 5 --seeds 0-4 --out data/ad_smoke
"""
from __future__ import annotations

import argparse
import os

from adinterdict.ad_generator import ADGeneratorConfig, generate_ad_graph
from adinterdict.graphio import save_instance


def _seeds(value: str) -> list[int]:
    if "-" in value:
        left, right = value.split("-", 1)
        return list(range(int(left), int(right) + 1))
    return [int(x) for x in value.split(",") if x.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--users", type=int, default=100)
    ap.add_argument("--groups", type=int, default=20)
    ap.add_argument("--computers", type=int, default=40)
    ap.add_argument("--targets", type=int, default=3)
    ap.add_argument("--entry-fraction", type=float, default=0.10)
    ap.add_argument("--seeds", default="0-4")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    for seed in _seeds(args.seeds):
        cfg = ADGeneratorConfig(users=args.users, groups=args.groups,
                                computers=args.computers, targets=args.targets,
                                entry_fraction=args.entry_fraction, seed=seed)
        inst = generate_ad_graph(cfg)
        path = os.path.join(args.out, inst.name)
        save_instance(inst, path)
        print(f"{inst.name}: n={inst.G.number_of_nodes()} m={inst.G.number_of_edges()} -> {path}")


if __name__ == "__main__":
    main()
