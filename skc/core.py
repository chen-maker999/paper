"""Core data model: turns, state writes, context units, and the ideal reader.

An agent context is a sequence of turns. Some turns carry *state writes*:
evidence that a state variable (key) took a value at that turn. A compressor
maps the turn history to a list of context units under a token budget. The
ideal reader decides, for a state key, whether the compressed context still
contains the latest value (CORRECT), only a superseded value (STALE), or
nothing (MISSING).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

_TOKEN_RE = re.compile(r"\w+|[^\w\s]")
_LINENO_RE = re.compile(r"^\s*\d+[:\t]\s?")
IDENT_RE = re.compile(r"[A-Za-z_][\w./-]{3,}[\w]")   # identifiers, dotted names, paths


def count_tokens(text: str) -> int:
    """Approximate token count (word pieces + punctuation), tokenizer-agnostic."""
    return len(_TOKEN_RE.findall(text))


def norm_line(line: str) -> str:
    """Normalise a line for evidence matching: drop editor line numbers and whitespace."""
    line = line.replace("\r", "")
    line = _LINENO_RE.sub("", line)
    return " ".join(line.split())


def informative_lines(text: str, min_len: int = 6) -> list[str]:
    """Distinct normalised lines that are long enough to be evidence."""
    out, seen = [], set()
    for raw in text.split("\n"):
        ln = norm_line(raw)
        if len(ln) < min_len or not re.search(r"[A-Za-z0-9]", ln) or ln in seen:
            continue
        seen.add(ln)
        out.append(ln)
    return out


def sample_evenly(items: list, k: int) -> list:
    if len(items) <= k:
        return list(items)
    step = (len(items) - 1) / (k - 1)
    return [items[round(i * step)] for i in range(k)]


@dataclass
class Write:
    key: str            # state variable, e.g. "edit:/repo/a.py"
    ktype: str          # key family, e.g. "edit", "cmd", "view", "search", "task"
    value: str          # raw value text carried by the turn
    turn: int           # index of the turn that carries the write
    evidence: tuple = ()  # normalised lines that identify the value

    def __post_init__(self):
        if not self.evidence:
            self.evidence = tuple(default_evidence(self.ktype, self.value))


def default_evidence(ktype: str, value: str, k: int = 12) -> list[str]:
    lines = informative_lines(value)
    if ktype == "cmd":          # results and errors are reported at the end
        return lines[-k:]
    return sample_evenly(lines, k)


@dataclass
class Turn:
    idx: int
    role: str                  # "system" | "task" | "assistant" | "tool" | "user"
    text: str
    writes: list = field(default_factory=list)
    meta: dict = field(default_factory=dict)
    tokens: int = -1

    def __post_init__(self):
        if self.tokens < 0:
            self.tokens = count_tokens(self.text)


@dataclass
class Unit:
    """A piece of compressed context. `src` is the originating turn (or -1)."""
    src: int
    kind: str                  # "turn" | "masked" | "trunc" | "ledger"
    text: str
    tokens: int = -1
    _lines: frozenset = None

    def __post_init__(self):
        if self.tokens < 0:
            self.tokens = count_tokens(self.text)

    @property
    def lines(self) -> frozenset:
        if self._lines is None:
            self._lines = frozenset(norm_line(l) for l in self.text.split("\n"))
        return self._lines


def context_tokens(units: list) -> int:
    return sum(u.tokens for u in units)


# --------------------------------------------------------------------------
# Ideal reader
# --------------------------------------------------------------------------
CORRECT, STALE, MISSING = "correct", "stale", "missing"


class ContextIndex:
    """Inverted index from normalised line -> source groups, for coverage queries.

    Units derived from the same source turn (e.g. several retained chunks of one
    tool output, or a turn and its ledger record) form one evidence group: the
    reader may combine pieces of the same original turn.
    """

    def __init__(self, units: list):
        self.units = units
        self.index: dict = {}
        for u in units:
            g = u.src
            for ln in u.lines:
                self.index.setdefault(ln, set()).add(g)
        self._tok = None

    def coverage(self, evidence) -> float:
        """Max over source groups of the fraction of evidence lines found in that group."""
        if not evidence:
            return 0.0
        hits: dict = {}
        for ln in evidence:
            for g in self.index.get(ln, ()):
                hits[g] = hits.get(g, 0) + 1
        return max(hits.values(), default=0) / len(evidence)

    def has_line(self, line: str) -> bool:
        return line in self.index

    def has_token(self, tok: str) -> bool:
        if self._tok is None:
            self._tok = set()
            for u in self.units:
                self._tok.update(IDENT_RE.findall(u.text))
        return tok in self._tok


def read_key(cidx: ContextIndex, writes: list, thresh: float = 0.8) -> str:
    """Classify recoverability of a key given its writes so far (in turn order)."""
    latest = writes[-1]
    if cidx.coverage(latest.evidence) >= thresh:
        return CORRECT
    latest_ev = set(latest.evidence)
    for w in reversed(writes[:-1]):
        if set(w.evidence) == latest_ev:
            continue  # identical value: not a different (stale) state
        if cidx.coverage(w.evidence) >= thresh:
            return STALE
    return MISSING
