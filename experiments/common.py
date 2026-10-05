"""Shared helpers: decision points and folds."""
from __future__ import annotations

import zlib

import numpy as np


def decision_points(turns, n=6, min_tokens=4000):
    """Up to `n` evenly spaced decision points (index of the last tool turn before an
    agent turn) whose history, excluding the system prompt, exceeds `min_tokens`."""
    hist_tok = np.cumsum([t.tokens if t.role != "system" else 0 for t in turns])
    cands = [i for i in range(3, len(turns) - 1)
             if turns[i].role == "tool" and turns[i + 1].role == "assistant" and hist_tok[i] > min_tokens]
    if not cands:
        return [], hist_tok
    pick = sorted(set(np.linspace(0, len(cands) - 1, min(n, len(cands))).round().astype(int)))
    return [cands[j] for j in pick], hist_tok


def instance_split(instance: str) -> int:
    """Deterministic 2-way split of SWE-bench instances (shared by all models)."""
    return zlib.crc32(instance.encode()) % 2
