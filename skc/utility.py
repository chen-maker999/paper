"""Utility-guided, supersession-safe compaction.

The history outside the recent window is turned into *items*:

  * ledger records - the latest write of every key (log compaction, verbatim
    value with provenance);
  * chunks         - blocks of at most `chunk_lines` lines of older turns, so the
    budget can be spent at a finer granularity than whole tool outputs.

Each item gets a predicted utility p = P(item is needed by the next step) from a
small learned model over model-agnostic features (age, item type, supersession
status, lexical relevance to the current step, how often its identifiers were
referenced since, ...). Selection is a knapsack solved greedily by
(p + lambda * [ledger record]) / tokens, under the *supersession-safety*
constraint of Corollary 1: an item carrying a superseded value of key k is only
admitted if the latest value of k is already in the context.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from .compressors import Compressor, _pinned_units, _split, _fit_recent, _WORD, truncate_text
from .core import IDENT_RE, Turn, Unit, count_tokens, informative_lines, norm_line

_ERR = re.compile(r"Error|Exception|Traceback|FAILED|failed|assert|error:")
_STOPID = set("""python python3 pytest testbed import return print self None True False class
def from with else elif while break continue lambda assert raise except finally yield""".split())

ITEM_TYPES = ["agent", "tool:edit", "tool:view", "tool:cmd", "tool:search", "tool:none",
              "led:edit", "led:view", "led:cmd", "led:search", "led:fact"]
FEATURES = (["log_age", "log_tokens", "bm25", "bm25_rank", "overlap_last", "overlap_lastobs",
             "overlap_task", "shared_rare", "log_refcount",
             "log_since_ref", "is_latest", "is_superseded", "has_error", "first_chunk",
             "last_chunk", "chunk_frac", "n_ident"] + [f"type={t}" for t in ITEM_TYPES])


@dataclass
class Item:
    src: int
    kind: str                  # "agent" | "tool" | "ledger"
    text: str
    tokens: int
    itype: str
    latest_keys: frozenset = frozenset()   # keys whose latest value this item carries
    stale_keys: frozenset = frozenset()    # keys whose superseded value this item carries
    chunk_idx: int = 0
    n_chunks: int = 1
    feats: dict = field(default_factory=dict)


def _idents(text):
    return {m for m in IDENT_RE.findall(text) if m.lower() not in _STOPID}


def _chunks(text, chunk_lines, max_tokens):
    lines = text.split("\n")
    out, cur, cur_tok = [], [], 0
    for ln in lines:
        c = count_tokens(ln) + 1
        if cur and (len(cur) >= chunk_lines or cur_tok + c > max_tokens):
            out.append("\n".join(cur)); cur, cur_tok = [], 0
        cur.append(ln); cur_tok += c
    if cur:
        out.append("\n".join(cur))
    return out


DEDUP_MARK = "[...]"


def latest_write_turns(hist, extractor=None):
    extractor = extractor or (lambda t: t.writes)
    latest = {}
    for t in hist:
        for w in extractor(t):
            latest[w.key] = w.turn
    return set(latest.values())


def dedup_history(pinned, rest, tail, min_len=6, protect=frozenset()):
    """Content-level compaction: keep every distinct line only at its latest
    occurrence. Lines of the pinned task and of the recent turns count as later
    occurrences. A removed run of lines is replaced by a short marker.
    Turns in `protect` (those carrying the current value of some key) are kept
    intact, so content compaction never splits the evidence of current state
    (supersession-aware deduplication).
    Returns {turn idx: deduplicated text}."""
    seen = set()
    for t in list(pinned) + list(tail):
        if t.role != "system":
            seen.update(norm_line(l) for l in t.text.split("\n"))
    out = {}
    for t in reversed(rest):
        if t.idx in protect:
            out[t.idx] = t.text
            seen.update(norm_line(l) for l in t.text.split("\n"))
            continue
        kept, skipping = [], False
        for raw in t.text.split("\n"):
            n = norm_line(raw)
            if len(n) >= min_len and n in seen:
                if not skipping:
                    kept.append(DEDUP_MARK)
                skipping = True
                continue
            skipping = False
            kept.append(raw)
            seen.add(n)
        out[t.idx] = "\n".join(kept)
    return out


def build_items(turns, recent=2, chunk_lines=12, chunk_tokens=200, value_cap=300, extractor=None,
                dedup=True):
    """Items for the history outside the pinned task and the last `recent` turns."""
    extractor = extractor or (lambda t: t.writes)
    pinned, hist = _split(turns)
    tail = hist[len(hist) - recent:] if recent > 0 else []
    tail_idx = {t.idx for t in tail}
    rest = [t for t in hist if t.idx not in tail_idx]
    texts = (dedup_history(pinned, rest, tail, protect=latest_write_turns(hist, extractor)) if dedup
             else {t.idx: t.text for t in rest})
    # latest (extracted) write per key
    latest, writes_of_turn = {}, {}
    for t in hist:
        ws = extractor(t)
        writes_of_turn[t.idx] = ws
        for w in ws:
            latest[w.key] = w
    items = []
    for t in rest:
        ws = writes_of_turn.get(t.idx, [])
        lk = frozenset(w.key for w in ws if latest.get(w.key) is w)
        sk = frozenset(w.key for w in ws if latest.get(w.key) is not w)
        if t.role in ("assistant", "user"):
            ktype = "agent"
        else:
            kts = [w.ktype for w in t.writes]
            ktype = "tool:" + (kts[0] if kts and kts[0] in ("edit", "view", "cmd", "search") else "none")
        pieces = [p for p in _chunks(texts[t.idx], chunk_lines, chunk_tokens)
                  if p.strip() and p.strip() != DEDUP_MARK]
        for j, piece in enumerate(pieces):
            items.append(Item(t.idx, "agent" if ktype == "agent" else "tool", piece, count_tokens(piece),
                              ktype, lk, sk, j, len(pieces)))
    for k, w in latest.items():
        if w.turn in tail_idx:
            continue
        val = truncate_text(w.value, value_cap, head_frac=1 / 3)
        text = f"[state] {k} (as of turn {w.turn}):\n{val}"
        kt = w.ktype if w.ktype in ("edit", "view", "cmd", "search") else "fact"
        items.append(Item(w.turn, "ledger", text, count_tokens(text), f"led:{kt}", frozenset([k])))
    return pinned, hist, tail, items


def featurize(items, pinned, hist):
    """Fill `item.feats` with decision-time features (no look-ahead)."""
    if not items:
        return
    now = hist[-1].idx if hist else 0
    agent_turns = [t for t in hist if t.role == "assistant"]
    agent_ids = [(t.idx, _idents(t.text)) for t in agent_turns]
    last_ids = set().union(*[ids for _, ids in agent_ids[-2:]]) if agent_ids else set()
    tool_turns = [t for t in hist if t.role == "tool"]
    lastobs_ids = _idents(tool_turns[-1].text) if tool_turns else set()
    task_ids = set().union(*[_idents(t.text) for t in pinned if t.role == "task"]) if pinned else set()
    hot = last_ids | lastobs_ids
    item_ids = [_idents(it.text) for it in items]
    id_df = Counter()
    for ids in item_ids:
        id_df.update(ids)
    # BM25 against task + latest agent turn
    q_txt = " ".join(t.text for t in pinned if t.role == "task")
    if agent_turns:
        q_txt += " " + agent_turns[-1].text
    q = Counter(w.lower() for w in _WORD.findall(q_txt))
    docs = [Counter(w.lower() for w in _WORD.findall(it.text)) for it in items]
    n = len(docs)
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
                s += idf * f * 2.2 / (f + 1.2 * (0.25 + 0.75 * dl / avgdl))
        scores.append(s)
    order = np.argsort(-np.array(scores), kind="stable")
    rank = np.empty(n)
    rank[order] = np.arange(n) / max(n - 1, 1)
    for it, sc, rk, ids in zip(items, scores, rank, item_ids):
        ref, last_ref = 0, None
        for idx, aids in agent_ids:
            if idx > it.src and ids & aids:
                ref += 1
                last_ref = idx
        f = it.feats
        f["log_age"] = math.log1p(max(now - it.src, 0) / 2)
        f["log_tokens"] = math.log1p(it.tokens)
        f["bm25"] = math.log1p(sc)
        f["bm25_rank"] = rk
        f["overlap_last"] = math.log1p(len(ids & last_ids))
        f["overlap_lastobs"] = math.log1p(len(ids & lastobs_ids))
        f["overlap_task"] = math.log1p(len(ids & task_ids))
        shared = ids & hot
        f["shared_rare"] = max((math.log(n / id_df[x]) for x in shared), default=-1.0)
        f["log_refcount"] = math.log1p(ref)
        f["log_since_ref"] = math.log1p((now - last_ref) / 2) if last_ref is not None else 5.0
        f["is_latest"] = float(bool(it.latest_keys))
        f["is_superseded"] = float(bool(it.stale_keys))
        f["has_error"] = float(bool(_ERR.search(it.text)))
        f["first_chunk"] = float(it.chunk_idx == 0)
        f["last_chunk"] = float(it.chunk_idx == it.n_chunks - 1)
        f["chunk_frac"] = it.chunk_idx / max(it.n_chunks - 1, 1)
        f["n_ident"] = math.log1p(len(ids))
        for tname in ITEM_TYPES:
            f[f"type={tname}"] = float(it.itype == tname)


def feature_matrix(items):
    return np.array([[it.feats[k] for k in FEATURES] for it in items], dtype=np.float32)


def count_items(items, need_idents, need_lines):
    """Number of distinct identifiers and copied lines of the next action that the item contains.
    This is the quantity the knapsack maximises (needed pieces covered per token)."""
    y = np.zeros(len(items), dtype=np.float32)
    if not need_idents and not need_lines:
        return y
    for i, it in enumerate(items):
        n = 0
        if need_idents:
            n += len(need_idents & set(IDENT_RE.findall(it.text)))
        if need_lines:
            n += len(need_lines & set(informative_lines(it.text, min_len=12)))
        y[i] = n
    return y


def label_items(items, need_idents, need_lines):
    """1 if the item contains an identifier or a copied line used by the next action."""
    y = np.zeros(len(items), dtype=np.int8)
    if not need_idents and not need_lines:
        return y
    for i, it in enumerate(items):
        if need_idents and need_idents & set(IDENT_RE.findall(it.text)):
            y[i] = 1
        elif need_lines and need_lines & set(informative_lines(it.text, min_len=12)):
            y[i] = 1
    return y


def _heuristic_utility(items):
    """Fallback utility without a learned model (used before training / in tests)."""
    w = {"led:edit": .8, "led:cmd": .6, "led:search": .5, "led:view": .4, "led:fact": .6}
    out = []
    for it in items:
        base = w.get(it.itype, 0.1)
        out.append(base * (0.5 + 0.5 * (1 - it.feats["bm25_rank"])) / (1 + it.feats["log_age"]))
    return np.array(out)


class UtilityCompaction(Compressor):
    """SKC: learned-utility, supersession-safe knapsack over ledger records and chunks.

    model: object with predict_proba(X) (e.g. a fitted sklearn classifier) or None
    lam:   state bonus added to the utility of ledger records (value of keeping
           the current state even when it is not needed at the very next step)
    safe:  enforce the supersession-safety constraint
    """
    name = "skc"

    def __init__(self, model=None, lam=0.3, recent=2, chunk_lines=12, chunk_tokens=200,
                 value_cap=300, safe=True, use_ledger=True, extractor=None, dedup=True, name=None):
        self.model, self.lam, self.recent, self.dedup = model, lam, recent, dedup
        self.chunk_lines, self.chunk_tokens, self.value_cap = chunk_lines, chunk_tokens, value_cap
        self.safe, self.use_ledger, self.extractor = safe, use_ledger, extractor
        if name:
            self.name = name

    def utilities(self, items):
        if self.model is None:
            return _heuristic_utility(items)
        X = feature_matrix(items)
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)[:, 1]
        return np.maximum(self.model.predict(X), 0.0)  # expected number of needed pieces

    def __call__(self, turns, budget):
        pinned, hist, tail, items = build_items(turns, self.recent, self.chunk_lines,
                                                self.chunk_tokens, self.value_cap, self.extractor,
                                                dedup=self.dedup)
        units, cost = _pinned_units(pinned)
        budget -= cost
        rec, used = _fit_recent(tail, budget)
        if not self.use_ledger:
            items = [it for it in items if it.kind != "ledger"]
        if items and used < budget:
            featurize(items, pinned, hist)
            u = self.utilities(items)
            u = u + self.lam * np.array([it.kind == "ledger" for it in items], dtype=float)
            ratio = u / np.maximum([it.tokens for it in items], 1)
            # keys whose latest value is already in context (recent window)
            have_latest = set()
            for t in tail:
                for w in (self.extractor or (lambda x: x.writes))(t):
                    have_latest.add(w.key)
            chosen = []
            for i in np.argsort(-ratio, kind="stable"):
                it = items[i]
                if used + it.tokens > budget:
                    continue
                if self.safe and it.stale_keys and not it.stale_keys <= have_latest:
                    continue
                chosen.append(it)
                used += it.tokens
                if it.kind == "ledger":  # a chunk may hold only part of a value; a record holds all of it
                    have_latest |= it.latest_keys
            units += _assemble(chosen)
        return units + rec


class OracleUtility(UtilityCompaction):
    """Upper bound: utilities are the true labels (does the next action use the item?).
    Set `need = (identifiers, lines)` of the next action before each call."""
    name = "skc-oracle"
    need = (set(), set())

    def utilities(self, items):
        return count_items(items, *self.need).astype(float)


def _assemble(chosen):
    """Merge chosen chunks of the same turn into one unit (gaps marked)."""
    by_src = {}
    for it in chosen:
        by_src.setdefault((it.src, it.kind == "ledger"), []).append(it)
    out = []
    for (src, is_led), its in sorted(by_src.items()):
        if is_led:
            for it in its:
                out.append(Unit(src, "ledger", it.text, it.tokens))
            continue
        its.sort(key=lambda x: x.chunk_idx)
        parts, prev = [], -1
        for it in its:
            if prev >= 0 and it.chunk_idx != prev + 1:
                parts.append("[...]")
            parts.append(it.text)
            prev = it.chunk_idx
        text = "\n".join(parts)
        out.append(Unit(src, "chunk", text, sum(x.tokens for x in its)))
    return out


class Dedup(Compressor):
    """Wrap any compressor with content-level compaction of the older history."""

    def __init__(self, inner, recent=2):
        self.inner, self.recent = inner, recent
        self.name = inner.name + "+dedup"

    def __call__(self, turns, budget):
        pinned, hist = _split(turns)
        tail = hist[len(hist) - self.recent:] if self.recent > 0 else []
        tail_idx = {t.idx for t in tail}
        rest = [t for t in hist if t.idx not in tail_idx]
        texts = dedup_history(pinned, rest, tail, protect=latest_write_turns(hist))
        new = []
        for t in turns:
            if t.idx in texts:
                new.append(Turn(t.idx, t.role, texts[t.idx], writes=t.writes, meta=t.meta))
            else:
                new.append(t)
        return self.inner(new, budget)


class ChunkBM25(Compressor):
    """Fine-grained salience baseline: the same chunks (no ledger), ranked by BM25
    score per token against (task + latest agent turn)."""
    name = "bm25_chunk"

    def __init__(self, recent=2, chunk_lines=12, chunk_tokens=200):
        self.recent, self.chunk_lines, self.chunk_tokens = recent, chunk_lines, chunk_tokens

    def __call__(self, turns, budget):
        pinned, hist, tail, items = build_items(turns, self.recent, self.chunk_lines, self.chunk_tokens,
                                                dedup=False)
        items = [it for it in items if it.kind != "ledger"]
        units, cost = _pinned_units(pinned)
        budget -= cost
        rec, used = _fit_recent(tail, budget)
        if items and used < budget:
            featurize(items, pinned, hist)
            sc = np.array([math.expm1(it.feats["bm25"]) for it in items])
            ratio = (sc + 1e-6) / np.maximum([it.tokens for it in items], 1)
            chosen = []
            for i in np.argsort(-ratio, kind="stable"):
                if used + items[i].tokens <= budget:
                    chosen.append(items[i]); used += items[i].tokens
            units += _assemble(chosen)
        return units + rec


class SelfInfoLines(Compressor):
    """Selective-Context-style pruning without an LM: keep the lines with the highest
    mean self-information under a unigram model of the history itself, until the
    budget is met (recent turns kept). A model-free proxy for token-importance
    pruning (Li et al. 2023; Jiang et al. 2023)."""
    name = "selfinfo"

    def __init__(self, recent=4):
        self.recent = recent

    def __call__(self, turns, budget):
        pinned, hist = _split(turns)
        units, cost = _pinned_units(pinned)
        budget -= cost
        tail = hist[len(hist) - self.recent:] if self.recent > 0 else []
        rec, used = _fit_recent(tail, budget)
        rest = hist[: len(hist) - len(tail)]
        cnt = Counter()
        for t in hist:
            cnt.update(w.lower() for w in _WORD.findall(t.text))
        tot = sum(cnt.values()) or 1
        cand = []
        for t in rest:
            for j, ln in enumerate(t.text.split("\n")):
                ws = [w.lower() for w in _WORD.findall(ln)]
                if not ws:
                    continue
                si = sum(-math.log(cnt[w] / tot) for w in ws) / len(ws)
                cand.append((si, t.idx, j, ln, count_tokens(ln) + 1))
        cand.sort(key=lambda x: -x[0])
        keep = {}
        for si, src, j, ln, c in cand:
            if used + c <= budget:
                keep.setdefault(src, []).append((j, ln)); used += c
        for src in sorted(keep):
            lines = [ln for _, ln in sorted(keep[src])]
            units.append(Unit(src, "lines", "\n".join(lines)))
        return units + rec
