"""Probe-based evaluation of compressors with the ideal reader."""
from __future__ import annotations

from .core import (CORRECT, IDENT_RE as _IDENT, MISSING, STALE, ContextIndex, context_tokens,
                   informative_lines, read_key)
_STOP = set("""python python3 pytest search_dir search_file find_file open goto edit create
scroll_down scroll_up submit str_replace_editor view str_replace insert new_str old_str
file_text view_range testbed import from return print self None True False class def
echo grep head tail sed find with else elif while break continue lambda assert raise
except finally yield global nonlocal pass async await""".split())


def state_probes(turns, prefix_end):
    """Writes per key visible in turns[:prefix_end+1] (keys in order of first write)."""
    by_key = {}
    for t in turns[: prefix_end + 1]:
        for w in t.writes:
            by_key.setdefault(w.key, []).append(w)
    return by_key


def use_probes(turns, prefix_end, system_text=""):
    """Identifiers the agent's next action uses that occurred earlier in the context.

    Returns (token, index of the most recent prior turn containing it) pairs: each
    token occurs in the next agent turn's action, in some turn of the history
    prefix, and not in the system prompt (which is always kept).
    """
    nxt = prefix_end + 1
    if nxt >= len(turns) or turns[nxt].role != "assistant":
        return []
    action = turns[nxt].meta.get("action", turns[nxt].text)
    cands = {m for m in _IDENT.findall(action) if m.lower() not in _STOP}
    if not cands:
        return []
    sys_toks = set(_IDENT.findall(system_text))
    last = {}
    for t in turns[1: prefix_end + 1]:
        for tok in cands.intersection(_IDENT.findall(t.text)):
            last[tok] = t.idx
    return sorted((c, last[c]) for c in cands if c in last and c not in sys_toks)


def copy_probes(turns, prefix_end, system_text=""):
    """Lines the agent's next action copies verbatim from earlier in the context.

    A copy probe is a normalised line (>= 12 chars) of the next action that occurs
    as a line of some earlier non-system turn and not in the system prompt.
    Returns (line, index of the most recent prior turn containing it) pairs.
    """
    nxt = prefix_end + 1
    if nxt >= len(turns) or turns[nxt].role != "assistant":
        return []
    action = turns[nxt].meta.get("action", turns[nxt].text)
    cands = set(informative_lines(action, min_len=12))
    if not cands:
        return []
    sys_lines = set(informative_lines(system_text, min_len=12))
    cands -= sys_lines
    last = {}
    for t in turns[1: prefix_end + 1]:
        for ln in cands.intersection(informative_lines(t.text, min_len=12)):
            last[ln] = t.idx
    return sorted(last.items())


def evaluate_prefix(turns, prefix_end, compressor, budget, key_type=None, with_use=False,
                    extra_thresholds=(), timing=None):
    ctx_turns = turns[: prefix_end + 1]
    if timing is not None:
        import time
        t0 = time.perf_counter()
        units = compressor(ctx_turns, budget)
        timing.append(time.perf_counter() - t0)
    else:
        units = compressor(ctx_turns, budget)
    cidx = ContextIndex(units)
    used = context_tokens([u for u in units if not (u.src == 0 and u.kind == "turn")])
    rows = []
    for key, ws in state_probes(turns, prefix_end).items():
        kt = ws[-1].ktype
        if key_type and kt != key_type:
            continue
        age = (prefix_end - ws[-1].turn)
        rows.append(dict(probe="state", key=key, ktype=kt, n_writes=len(ws), age=age,
                         outcome=read_key(cidx, ws)))
        for th in extra_thresholds:
            rows.append(dict(probe=f"state@{th}", key=key, ktype=kt, n_writes=len(ws), age=age,
                             outcome=read_key(cidx, ws, thresh=th)))
    if with_use:
        sys_text = turns[0].text if turns and turns[0].role == "system" else ""
        for tok, last in use_probes(turns, prefix_end, sys_text):
            rows.append(dict(probe="use", key=tok, ktype="use", n_writes=0, age=prefix_end - last,
                             outcome=CORRECT if cidx.has_token(tok) else MISSING))
        for ln, last in copy_probes(turns, prefix_end, sys_text):
            rows.append(dict(probe="copy", key=ln, ktype="copy", n_writes=0, age=prefix_end - last,
                             outcome=CORRECT if cidx.has_line(ln) else MISSING))
    return rows, used


__all__ = ["evaluate_prefix", "state_probes", "use_probes", "copy_probes", "CORRECT", "STALE", "MISSING"]
