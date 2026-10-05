"""Parse SWE-agent trajectories (.traj) into turns with structural state writes.

Two interface generations are supported:
  * SWE-agent 0.x ACI: open/goto/scroll_*/create/edit a:b ... end_of_edit,
    search_dir/search_file/find_file, plus arbitrary shell commands;
  * SWE-agent 1.x: str_replace_editor {view,create,str_replace,insert}, bash.

State keys (all derived from the agent's own tool calls, i.e. information an
agent framework has without any model in the loop):
  edit:<path>    latest text written into a file (edit body / new_str / file_text)
  view:<path>    latest file content shown to the agent
  search:<cmd>   latest result of a code search
  cmd:<cmd>      latest output of a (normalised) shell command
"""
from __future__ import annotations

import json
import re
import shlex

from .core import Turn, Write, informative_lines

_FILE_HDR = re.compile(r"\[File: (\S+) \(\d+ lines total\)\]")
_CD_PREFIX = re.compile(r"^(cd\s+\S+\s*(&&|;)\s*)+")


def _content_text(c):
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        return "\n".join(x.get("text", "") if isinstance(x, dict) else str(x) for x in c)
    return str(c)


def _task_text(history):
    """Extract the issue statement from the first non-demonstration user message."""
    for m in history:
        if m.get("role") != "user":
            continue
        txt = _content_text(m.get("content"))
        if "--- DEMONSTRATION ---" in txt or m.get("is_demo"):
            continue
        mm = re.search(r"<pr_description>\s*(.*?)\s*</pr_description>", txt, re.S)
        if mm:
            return mm.group(1)
        mm = re.search(r"ISSUE:\n(.*?)\n\s*INSTRUCTIONS:", txt, re.S)
        if mm:
            return mm.group(1)
    return ""


def _system_text(history):
    parts = []
    for m in history:
        txt = _content_text(m.get("content"))
        if m.get("role") == "system" or "--- DEMONSTRATION ---" in txt or m.get("is_demo"):
            parts.append(txt)
    return "\n".join(parts)


def normalise_cmd(action: str) -> str:
    a = _CD_PREFIX.sub("", action.strip())
    return " ".join(a.split())[:200]


_ABOVE = re.compile(r"\((\d+) more lines above\)")


def _view_start(obs: str, opts=None) -> int:
    if opts and "view_range" in opts:
        try:
            return _bucket(int(opts["view_range"].strip("[]").replace(",", " ").split()[0]))
        except (ValueError, IndexError):
            return 1
    m = _ABOVE.search(obs or "")
    return _bucket(int(m.group(1)) + 1 if m else 1)


def _bucket(line: int, width: int = 50) -> int:
    """Views whose windows start within the same `width`-line block share a key."""
    return (max(line, 1) - 1) // width * width + 1


def _parse_editor(action: str):
    try:
        toks = shlex.split(action, posix=True)
    except ValueError:
        return None
    if len(toks) < 3 or toks[0] != "str_replace_editor":
        return None
    sub, path = toks[1], toks[2]
    opts, i = {}, 3
    while i < len(toks):
        if toks[i].startswith("--") and i + 1 < len(toks):
            j = i + 1
            vals = []
            while j < len(toks) and not toks[j].startswith("--"):
                vals.append(toks[j]); j += 1
            opts[toks[i][2:]] = " ".join(vals) if len(vals) > 1 else (vals[0] if vals else "")
            i = j
        else:
            i += 1
    return sub, path, opts


class EditRegions:
    """Assigns region identities to edits so that only edits touching the same
    region of a file supersede each other."""

    def __init__(self):
        self.regions = {}   # path -> list of dicts(id, start, end, text)

    def _new(self, path, **kw):
        lst = self.regions.setdefault(path, [])
        r = dict(id=len(lst), **kw)
        lst.append(r)
        return r

    def line_edit(self, path, a, b, body):
        n = max(body.count("\n") + 1, 1)
        for r in self.regions.get(path, []):
            if r.get("start") is not None and not (b < r["start"] or a > r["end"]):
                r.update(start=a, end=a + n - 1, text=body)
                return f"edit:{path}#{r['id']}"
        r = self._new(path, start=a, end=a + n - 1, text=body)
        return f"edit:{path}#{r['id']}"

    def text_edit(self, path, old, new):
        old_lines = set(informative_lines(old or ""))
        for r in self.regions.get(path, []):
            if old and (old in r["text"] or old_lines & set(informative_lines(r["text"]))):
                r.update(text=new)
                return f"edit:{path}#{r['id']}"
        r = self._new(path, start=None, end=None, text=new)
        return f"edit:{path}#{r['id']}"


def action_writes(action: str, obs: str, open_file: str | None, regions: EditRegions):
    """Return (writes_on_agent_turn, writes_on_tool_turn, new_open_file) as
    (key, ktype, value) triples."""
    a = action.strip()
    head = a.split("\n")[0].strip()
    first = head.split(" ")[0] if head else ""
    aw, ow = [], []
    m = _FILE_HDR.search(obs or "")
    shown = m.group(1) if m else None

    if first == "str_replace_editor":
        p = _parse_editor(a)
        if p:
            sub, path, opts = p
            if sub == "view":
                ow.append((f"view:{path}@{_view_start(obs, opts)}", "view", obs))
            elif sub == "create" and "file_text" in opts:
                regions.regions.pop(path, None)
                aw.append((regions.text_edit(path, None, opts["file_text"]), "edit", opts["file_text"]))
            elif sub == "str_replace" and "new_str" in opts:
                key = regions.text_edit(path, opts.get("old_str", ""), opts["new_str"])
                aw.append((key, "edit", opts["new_str"]))
            elif sub == "insert" and "new_str" in opts:
                aw.append((regions.text_edit(path, None, opts["new_str"]), "edit", opts["new_str"]))
        return aw, ow, open_file
    if first == "edit":
        body = a.split("\n", 1)[1] if "\n" in a else ""
        body = re.sub(r"\n?end_of_edit\s*$", "", body)
        target = open_file or shown or "?"
        mm = re.match(r"edit\s+(\d+):(\d+)", head)
        a0, b0 = (int(mm.group(1)), int(mm.group(2))) if mm else (1, 1)
        aw.append((regions.line_edit(target, a0, b0, body), "edit", body))
        if shown:
            ow.append((f"view:{shown}@{_view_start(obs)}", "view", obs))
        return aw, ow, shown or open_file
    if first in ("open", "goto", "scroll_up", "scroll_down", "create"):
        if shown:
            ow.append((f"view:{shown}@{_view_start(obs)}", "view", obs))
        return aw, ow, shown or open_file
    if first in ("search_dir", "search_file", "find_file"):
        ow.append((f"search:{normalise_cmd(a)}", "search", obs))
        return aw, ow, open_file
    if first in ("submit", "exit_forfeit", "exit_cost", "") or a.startswith("submit"):
        return aw, ow, open_file
    ow.append((f"cmd:{normalise_cmd(a)}", "cmd", obs))
    return aw, ow, shown or open_file


def load_traj(path: str):
    """Return (turns, info). turns[0]=system, turns[1]=task, then alternating
    assistant/tool turns. Each step contributes two turns."""
    d = json.load(open(path))
    hist = d.get("history", [])
    traj = d.get("trajectory", [])
    turns = [Turn(0, "system", _system_text(hist)),
             Turn(1, "task", _task_text(hist).replace("\r", ""))]
    regions = EditRegions()
    turns[1].writes.append(Write("task", "task", turns[1].text, 1))
    open_file = None
    for st in traj:
        action = st.get("action", "") or ""
        thought = st.get("thought", "") or ""
        obs = st.get("observation", "") or ""
        obs = obs.replace("\r", "")
        aw, ow, open_file = action_writes(action, obs, open_file, regions)
        ta = Turn(len(turns), "assistant", (thought.strip() + "\n\n" + action.strip()).strip(),
                  meta={"action": action})
        turns.append(ta)
        to = Turn(len(turns), "tool", obs, meta={"action": action})
        turns.append(to)
        for k, kt, v in aw:
            w = Write(k, kt, v, ta.idx)
            if w.evidence:
                ta.writes.append(w)
        for k, kt, v in ow:
            w = Write(k, kt, v, to.idx)
            if w.evidence:
                to.writes.append(w)
    info = d.get("info", {})
    return turns, info
