"""Parse OpenHands (CodeAct) trajectories into turns with structural state writes.

OpenHands logs an OpenAI-style message list: a system prompt, the task (user),
then assistant messages with function calls (`execute_bash`,
`str_replace_editor`, `think`, `finish`, ...) answered by tool messages. Keys
follow the same scheme as for SWE-agent (see skc/sweagent.py):
`edit:<path>#<region>`, `view:<path>@<window>`, `cmd:<command>`.
"""
from __future__ import annotations

import json
import re

from .core import Turn, Write
from .sweagent import EditRegions, _bucket, normalise_cmd


def _text(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(c.get("text", "") for c in content if isinstance(c, dict))
    return ""


def _task(text):
    m = re.search(r"<issue_description>\s*(.*?)\s*</issue_description>", text, re.S)
    return m.group(1) if m else text


def render_call(name, args):
    """Human-readable rendering of a tool call, used as the agent's action text."""
    if name == "execute_bash":
        return args.get("command", "")
    if name == "str_replace_editor":
        parts = [f"str_replace_editor {args.get('command', '')} {args.get('path', '')}"]
        if "view_range" in args:
            parts[0] += f" --view_range {args['view_range']}"
        for k in ("old_str", "new_str", "file_text"):
            if args.get(k) is not None:
                parts.append(f"--{k}\n{args[k]}")
        if args.get("insert_line") is not None:
            parts[0] += f" --insert_line {args['insert_line']}"
        return "\n".join(parts)
    if name == "think":
        return args.get("thought", "")
    if name == "execute_ipython_cell":
        return args.get("code", "")
    return f"{name} {json.dumps(args)[:2000]}"


def call_writes(name, args, obs, regions):
    """(writes on the agent turn, writes on the tool turn) as (key, ktype, value)."""
    aw, ow = [], []
    if name == "str_replace_editor":
        cmd, path = args.get("command"), args.get("path", "?")
        if cmd == "view":
            vr = args.get("view_range")
            start = vr[0] if isinstance(vr, list) and vr else 1
            if not obs.startswith("ERROR"):
                ow.append((f"view:{path}@{_bucket(int(start) if str(start).lstrip('-').isdigit() else 1)}",
                           "view", obs))
        elif cmd == "create" and args.get("file_text") is not None:
            regions.regions.pop(path, None)
            aw.append((regions.text_edit(path, None, args["file_text"]), "edit", args["file_text"]))
        elif cmd == "str_replace" and args.get("new_str") is not None and not obs.startswith("ERROR"):
            aw.append((regions.text_edit(path, args.get("old_str", ""), args["new_str"]), "edit", args["new_str"]))
        elif cmd == "insert" and args.get("new_str") is not None:
            aw.append((regions.text_edit(path, None, args["new_str"]), "edit", args["new_str"]))
    elif name in ("execute_bash", "execute_ipython_cell"):
        c = args.get("command") or args.get("code") or ""
        if c.strip():
            ow.append((f"cmd:{normalise_cmd(c)}", "cmd", obs))
    return aw, ow


def _meta(e):
    m = e.get("tool_call_metadata")
    if isinstance(m, str):
        return None
    return m


def events_to_messages(events):
    """Convert the OpenHands *event stream* log format into the message format."""
    msgs = []
    for e in events:
        args = e.get("args") if isinstance(e.get("args"), dict) else {}
        meta = _meta(e) or {}
        if e.get("action") == "system":
            msgs.append({"role": "system", "content": args.get("content") or e.get("message", "")})
        elif e.get("source") == "user" and e.get("action") == "message":
            msgs.append({"role": "user", "content": args.get("content") or e.get("message", "")})
        elif e.get("source") == "agent" and e.get("action") and meta.get("function_name"):
            name = meta["function_name"]
            a = dict(args)
            if e["action"] == "read":
                a = {"command": "view", "path": args.get("path", "")}
                if args.get("view_range"):
                    a["view_range"] = args["view_range"]
            call = {"id": meta.get("tool_call_id"), "type": "function",
                    "function": {"name": name, "arguments": json.dumps(a)}}
            if msgs and msgs[-1].get("role") == "assistant" and msgs[-1].get("_resp") == \
                    (meta.get("model_response") or {}).get("id"):
                msgs[-1]["tool_calls"].append(call)
            else:
                msgs.append({"role": "assistant", "content": args.get("thought", ""), "tool_calls": [call],
                             "_resp": (meta.get("model_response") or {}).get("id")})
        elif e.get("source") == "agent" and e.get("action") == "message":
            msgs.append({"role": "assistant", "content": args.get("content") or e.get("message", "")})
        elif e.get("observation") and meta.get("tool_call_id"):
            msgs.append({"role": "tool", "tool_call_id": meta["tool_call_id"], "name": meta.get("function_name"),
                         "content": e.get("content", "")})
    return msgs


_FN = re.compile(r"<function=([\w.-]+)>(.*?)(?:</function>|$)", re.S)
_PARAM = re.compile(r"<parameter=([\w.-]+)>(.*?)</parameter>", re.S)
_EXEC = re.compile(r"^EXECUTION RESULT of \[([\w.-]+)\]:\n?", re.S)


def text_calls_to_messages(msgs):
    """Convert prompt-based function calling (`<function=name><parameter=k>v</parameter></function>`
    in assistant text, results as `EXECUTION RESULT of [name]:` user messages) into tool calls."""
    out, pending, k = [], [], 0
    for m in msgs:
        txt = _text(m.get("content"))
        if m.get("role") == "assistant" and "<function=" in txt:
            calls = []
            for name, body in _FN.findall(txt):
                args = {}
                for key, val in _PARAM.findall(body):
                    val = val.strip("\n")
                    if key in ("view_range", "insert_line"):
                        try:
                            val = json.loads(val)
                        except json.JSONDecodeError:
                            pass
                    args[key] = val
                k += 1
                calls.append({"id": f"t{k}", "type": "function",
                              "function": {"name": name, "arguments": json.dumps(args)}})
            pending = [c["id"] for c in calls]
            out.append({"role": "assistant", "content": txt.split("<function=")[0], "tool_calls": calls})
        elif m.get("role") == "user" and _EXEC.match(txt):
            name = _EXEC.match(txt).group(1)
            cid = pending.pop(0) if pending else None
            out.append({"role": "tool", "tool_call_id": cid, "name": name, "content": _EXEC.sub("", txt, count=1)})
        else:
            out.append(m)
    return out


def load_openhands(path: str):
    msgs = json.load(open(path))
    if isinstance(msgs, dict):
        msgs = msgs.get("history") or msgs.get("messages") or []
    if msgs and isinstance(msgs[0], dict) and "role" not in msgs[0] and ("action" in msgs[0] or "observation" in msgs[0]):
        msgs = events_to_messages(msgs)
    if not any(m.get("tool_calls") for m in msgs) and any(
            m.get("role") == "assistant" and "<function=" in _text(m.get("content")) for m in msgs):
        msgs = text_calls_to_messages(msgs)
    sys_txt = "\n".join(_text(m.get("content")) for m in msgs if m.get("role") == "system")
    users = [m for m in msgs if m.get("role") == "user"]
    task = _task(_text(users[0].get("content"))) if users else ""
    turns = [Turn(0, "system", sys_txt), Turn(1, "task", task.replace("\r", ""))]
    turns[1].writes.append(Write("task", "task", turns[1].text, 1))
    regions = EditRegions()
    pending = {}  # tool_call_id -> (name, args, agent turn)
    first_user_seen = False
    for m in msgs:
        role = m.get("role")
        if role == "system":
            continue
        if role == "user":
            if not first_user_seen:
                first_user_seen = True
                continue
            t = Turn(len(turns), "user", _text(m.get("content")))
            turns.append(t)
            continue
        if role == "assistant":
            calls = m.get("tool_calls") or []
            rendered, parsed = [], []
            for c in calls:
                fn = c.get("function", {})
                try:
                    args = json.loads(fn.get("arguments") or "{}")
                except json.JSONDecodeError:
                    args = {}
                parsed.append((c.get("id"), fn.get("name", ""), args))
                rendered.append(render_call(fn.get("name", ""), args))
            action = "\n".join(rendered)
            text = (_text(m.get("content")).strip() + "\n\n" + action).strip()
            t = Turn(len(turns), "assistant", text, meta={"action": action})
            turns.append(t)
            for cid, name, args in parsed:
                pending[cid] = (name, args, t)
            continue
        if role == "tool":
            obs = _text(m.get("content")).replace("\r", "")
            name, args, at = pending.pop(m.get("tool_call_id"), (m.get("name", ""), {}, None))
            t = Turn(len(turns), "tool", obs, meta={"action": render_call(name, args)})
            turns.append(t)
            aw, ow = call_writes(name, args, obs, regions)
            for k, kt, v in aw:
                if at is not None:
                    w = Write(k, kt, v, at.idx)
                    if w.evidence:
                        at.writes.append(w)
            for k, kt, v in ow:
                w = Write(k, kt, v, t.idx)
                if w.evidence:
                    t.writes.append(w)
    return turns, {}
