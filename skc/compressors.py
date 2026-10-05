"""Context compressors.

Every compressor receives the full turn history (``turns[0]`` may be a system
turn that is always kept and not charged to the budget; the task turn is
pinned and charged) and a token budget, and returns a list of Units.
"""
from __future__ import annotations

import math
import random
import re
from collections import Counter

from .core import Turn, Unit, count_tokens

MASK_TEXT = "[old tool output omitted]"


def truncate_text(text: str, max_tokens: int, head_frac: float = 1 / 3) -> str:
    """Keep the head and tail of `text` (line-granular) within roughly max_tokens."""
    if count_tokens(text) <= max_tokens:
        return text
    lines = text.split("\n")
    head_budget = int(max_tokens * head_frac)
    tail_budget = max_tokens - head_budget
    head, used = [], 0
    for ln in lines:
        c = count_tokens(ln) + 1
        if used + c > head_budget:
            break
        head.append(ln); used += c
    tail, used = [], 0
    for ln in reversed(lines[len(head):]):
        c = count_tokens(ln) + 1
        if used + c > tail_budget:
            break
        tail.append(ln); used += c
    tail.reverse()
    return "\n".join(head + ["[... truncated ...]"] + tail)


def _split(turns):
    pinned = [t for t in turns if t.role in ("system", "task")]
    hist = [t for t in turns if t.role not in ("system", "task")]
    return pinned, hist


def _pinned_units(pinned):
    units, cost = [], 0
    for t in pinned:
        units.append(Unit(t.idx, "turn", t.text, t.tokens))
        if t.role != "system":
            cost += t.tokens
    return units, cost


def _turn_unit(t: Turn) -> Unit:
    return Unit(t.idx, "turn", t.text, t.tokens)


def _fit_recent(hist, budget, always_last=1):
    """Most recent whole turns that fit; the last `always_last` turns are truncated if needed."""
    chosen, used = [], 0
    for j, t in enumerate(reversed(hist)):
        if used + t.tokens <= budget:
            chosen.append(_turn_unit(t)); used += t.tokens
        elif j < always_last and budget - used > 32:
            txt = truncate_text(t.text, budget - used - 8)
            u = Unit(t.idx, "trunc", txt)
            chosen.append(u); used += u.tokens
        else:
            break
    chosen.reverse()
    return chosen, used


class Compressor:
    name = "base"

    def __call__(self, turns: list, budget: int) -> list:
        raise NotImplementedError


class Full(Compressor):
    name = "full"

    def __call__(self, turns, budget):
        return [_turn_unit(t) for t in turns]


class RecencyWindow(Compressor):
    """Pin the task, keep the most recent turns that fit (FIFO truncation)."""
    name = "window"

    def __call__(self, turns, budget):
        pinned, hist = _split(turns)
        units, cost = _pinned_units(pinned)
        rec, _ = _fit_recent(hist, budget - cost)
        return units + rec


class ObservationMasking(Compressor):
    """Lindenbauer et al. (2025): keep every agent turn, mask tool outputs older
    than the last `keep_obs` ones; drop oldest turns if still over budget."""
    name = "obs_mask"

    def __init__(self, keep_obs: int = 10):
        self.keep_obs = keep_obs

    def __call__(self, turns, budget):
        pinned, hist = _split(turns)
        units, cost = _pinned_units(pinned)
        tool_idx = [i for i, t in enumerate(hist) if t.role == "tool"]
        recent_tools = set(tool_idx[-self.keep_obs:])
        view = []
        for i, t in enumerate(hist):
            if t.role == "tool" and i not in recent_tools:
                view.append(Turn(t.idx, "masked", MASK_TEXT))
            else:
                view.append(t)
        rec, _ = _fit_recent(view, budget - cost)
        return units + rec


class TruncateObservations(Compressor):
    """Cap each older tool output at `cap` tokens (head+tail), keep the last
    `keep_full` outputs intact, then FIFO-truncate to the budget."""
    name = "obs_trunc"

    def __init__(self, cap: int = 200, keep_full: int = 2):
        self.cap, self.keep_full = cap, keep_full

    def __call__(self, turns, budget):
        pinned, hist = _split(turns)
        units, cost = _pinned_units(pinned)
        tool_idx = [i for i, t in enumerate(hist) if t.role == "tool"]
        keep = set(tool_idx[-self.keep_full:])
        view = []
        for i, t in enumerate(hist):
            if t.role == "tool" and i not in keep and t.tokens > self.cap:
                view.append(Turn(t.idx, "tool", truncate_text(t.text, self.cap)))
            else:
                view.append(t)
        rec, _ = _fit_recent(view, budget - cost)
        return units + rec


_WORD = re.compile(r"[A-Za-z_][A-Za-z0-9_]{2,}")


def bm25_scores(pinned, hist, rest, k1=1.2, b=0.75):
    """BM25 score of each turn in `rest` against (task + latest agent turn)."""
    query_txt = " ".join(t.text for t in pinned if t.role == "task")
    last_agent = [t for t in hist if t.role == "assistant"][-1:]
    query_txt += " " + " ".join(t.text for t in last_agent)
    q = Counter(w.lower() for w in _WORD.findall(query_txt))
    docs = [Counter(w.lower() for w in _WORD.findall(t.text)) for t in rest]
    n = max(len(docs), 1)
    avgdl = sum(sum(d.values()) for d in docs) / n or 1.0
    df = Counter()
    for d in docs:
        df.update(d.keys())
    scores = []
    for d in docs:
        dl = sum(d.values())
        s = 0.0
        for w in q:
            f = d.get(w, 0)
            if f:
                idf = math.log(1 + (n - df[w] + 0.5) / (df[w] + 0.5))
                s += idf * f * (k1 + 1) / (f + k1 * (1 - b + b * dl / avgdl))
        scores.append(s)
    return scores


class BM25Select(Compressor):
    """Salience-based selection: keep the last `keep_last` turns, then the turns
    with the highest BM25 score against (task + latest agent turn), in original
    order. A turn-level stand-in for token-importance pruning methods."""
    name = "bm25"

    def __init__(self, keep_last: int = 2, k1: float = 1.2, b: float = 0.75):
        self.keep_last, self.k1, self.b = keep_last, k1, b

    def __call__(self, turns, budget):
        pinned, hist = _split(turns)
        units, cost = _pinned_units(pinned)
        budget -= cost
        tail = hist[-self.keep_last:]
        rest = hist[: len(hist) - len(tail)]
        rec, used = _fit_recent(tail, budget)
        scores = bm25_scores(pinned, hist, rest, self.k1, self.b)
        order = sorted(range(len(rest)), key=lambda i: -scores[i])
        picked = []
        for i in order:
            if used + rest[i].tokens <= budget:
                picked.append(i); used += rest[i].tokens
        sel = [_turn_unit(rest[i]) for i in sorted(picked)]
        return units + sel + rec


class RandomSelect(Compressor):
    name = "random"

    def __init__(self, seed: int = 0, keep_last: int = 2):
        self.seed, self.keep_last = seed, keep_last

    def __call__(self, turns, budget):
        pinned, hist = _split(turns)
        units, cost = _pinned_units(pinned)
        budget -= cost
        tail = hist[-self.keep_last:]
        rest = hist[: len(hist) - len(tail)]
        rec, used = _fit_recent(tail, budget)
        rng = random.Random(self.seed * 1000003 + len(hist))
        order = list(range(len(rest)))
        rng.shuffle(order)
        picked = []
        for i in order:
            if used + rest[i].tokens <= budget:
                picked.append(i); used += rest[i].tokens
        return units + [_turn_unit(rest[i]) for i in sorted(picked)] + rec


# --------------------------------------------------------------------------
# State-Keyed Compaction (ours)
# --------------------------------------------------------------------------
DEFAULT_TYPE_WEIGHT = {"edit": 4.0, "cmd": 2.0, "search": 1.5, "view": 1.0, "fact": 2.0}


class StateKeyedCompaction(Compressor):
    """SKC: a latest-value-per-key ledger (log compaction) + recent window +
    action skeleton.

    1. pin the task;
    2. keep the last `recent` turns verbatim;
    3. build the ledger: for every key, the *latest* write seen by the extractor,
       rendered verbatim (value capped at `value_cap` tokens, head+tail);
       ledger entries whose source turn is already kept are skipped;
       entries are admitted greedily by priority p_k / size_k, with
       p_k = w_type * tau / (tau + age_k) (a knapsack-ratio rule);
    4. spend what is left extending the window backwards with whole turns, then
       with agent turns only (tool outputs masked).

    `extractor(turn) -> list[Write]` decides which writes are visible to the
    ledger; by default the gold structural writes attached to each turn.
    """
    name = "skc"

    def __init__(self, recent: int = 4, value_cap: int = 300, tau: float = 20.0,
                 type_weight=None, extractor=None, use_ledger=True, extend=True,
                 extend_mode="recent", name=None):
        self.recent, self.value_cap, self.tau = recent, value_cap, tau
        self.type_weight = type_weight or DEFAULT_TYPE_WEIGHT
        self.extractor = extractor or (lambda t: t.writes)
        self.use_ledger, self.extend, self.extend_mode = use_ledger, extend, extend_mode
        if name:
            self.name = name

    def ledger(self, hist):
        latest = {}
        for t in hist:
            for w in self.extractor(t):
                latest[w.key] = w
        return latest

    def __call__(self, turns, budget):
        pinned, hist = _split(turns)
        units, cost = _pinned_units(pinned)
        budget -= cost
        tail = hist[-self.recent:]
        rec, used = _fit_recent(tail, budget)
        kept_turns = {u.src for u in rec}
        led_units = []
        if self.use_ledger and hist:
            now = hist[-1].idx
            cands = []
            for k, w in self.ledger(hist).items():
                if w.turn in kept_turns:
                    continue
                val = truncate_text(w.value, self.value_cap, head_frac=1 / 3)
                text = f"[state] {k} (as of turn {w.turn}):\n{val}"
                u = Unit(w.turn, "ledger", text)
                age = max(now - w.turn, 0) / 2.0  # in agent steps
                p = self.type_weight.get(w.ktype, 1.0) * self.tau / (self.tau + age)
                cands.append((p / max(u.tokens, 1), u))
            cands.sort(key=lambda x: -x[0])
            for _, u in cands:
                if used + u.tokens <= budget:
                    led_units.append(u); used += u.tokens
            led_units.sort(key=lambda u: u.src)
        ext = []
        if self.extend and self.extend_mode == "bm25":
            # Relevance-ranked extension. Safe w.r.t. staleness: older turns can only
            # add older writes, while the ledger already holds the latest ones.
            rest = hist[: len(hist) - len(tail)]
            scores = bm25_scores(pinned, hist, rest)
            picked = []
            for i in sorted(range(len(rest)), key=lambda i: -scores[i]):
                if used + rest[i].tokens <= budget:
                    picked.append(i); used += rest[i].tokens
            ext = [_turn_unit(rest[i]) for i in sorted(picked)]
        elif self.extend:
            rest = hist[: len(hist) - len(tail)]
            masking = False
            for t in reversed(rest):
                if not masking and used + t.tokens <= budget:
                    ext.append(_turn_unit(t)); used += t.tokens
                    continue
                masking = True
                if t.role == "tool":
                    c = count_tokens(MASK_TEXT)
                    if used + c <= budget:
                        ext.append(Unit(t.idx, "masked", MASK_TEXT, c)); used += c
                elif used + t.tokens <= budget:
                    ext.append(_turn_unit(t)); used += t.tokens
            ext.reverse()
        return units + led_units + ext + rec


def default_suite(seed: int = 0):
    return [
        RecencyWindow(),
        ObservationMasking(),
        TruncateObservations(),
        BM25Select(),
        RandomSelect(seed=seed),
        StateKeyedCompaction(),
        StateKeyedCompaction(extend_mode="bm25", name="skc+bm25"),
    ]
