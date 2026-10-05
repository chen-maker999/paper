"""LLM-in-the-loop evaluation of context compressors (requires an Anthropic API key).

Two tasks are posed to an LLM reader on the *compressed* context of real agent
trajectories (same decision points and compressors as run_eval.py):

  A. State QA     - "Quote the last line of the most recent output of `<cmd>`."
                    Graded correct / stale (an older output's last line) / wrong.
                    Commands are chosen among those last run >= 3 steps earlier.
  B. Next action  - "Predict the agent's next tool call." Graded by whether the
                    prediction names the same file path(s) as the true action and
                    by identifier-level F1 against the true action.

An LLM-summarisation compressor is included as a baseline: the history outside
the last two turns is summarised by the same LLM into at most half the budget.

Two backends:
  --backend anthropic  Claude via the Message Batches API (50% price); needs ANTHROPIC_API_KEY.
  --backend local      any OpenAI-compatible server (vLLM, Ollama, llama.cpp server, LM Studio),
                       e.g. --base-url http://localhost:8000/v1 --model Qwen/Qwen3-14B

Phases (results are cached, so an interrupted run can be resumed):
  python experiments/llm_eval.py prepare   --cache C --out results/llm
  python experiments/llm_eval.py summarise --out results/llm [backend options]   # LLM-summary baseline
  python experiments/llm_eval.py submit    --out results/llm [backend options]
  python experiments/llm_eval.py collect   --out results/llm
See docs/LOCAL_LLM_EVAL.md.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from common import decision_points  # noqa: E402
from corpus import SHORT, load_corpus  # noqa: E402
from run_eval import DEV_PER_MODEL, utility_model  # noqa: E402
from skc.compressors import BM25Select, ObservationMasking, RecencyWindow  # noqa: E402
from skc.core import IDENT_RE, count_tokens, informative_lines, norm_line  # noqa: E402
from skc.utility import UtilityCompaction  # noqa: E402

ARGS = None
BUDGET = 8000
SYSTEM = ("You are given the (possibly compressed) working history of a software-engineering agent "
          "that is fixing a GitHub issue. Parts of the history may have been removed. Answer only from "
          "the history; do not guess.")


def render(units):
    parts = []
    for u in units:
        if u.src == 0:
            continue
        tag = {"ledger": "state record", "masked": "omitted tool output"}.get(u.kind, "turn")
        parts.append(f"<{tag} index={u.src}>\n{u.text}\n</{tag}>")
    return "\n".join(parts)


def last_line(text):
    ls = informative_lines(text)
    return ls[-1] if ls else None


def pick_cmd_probe(turns, i, rng):
    """A command whose latest output is >= 3 steps old and has a distinctive last line."""
    by_key = {}
    for t in turns[: i + 1]:
        for w in t.writes:
            if w.ktype == "cmd":
                by_key.setdefault(w.key, []).append(w)
    cands = []
    for k, ws in by_key.items():
        if (i - ws[-1].turn) / 2 < 3:
            continue
        ll = last_line(ws[-1].value)
        if ll and len(ll) >= 12:
            older = [last_line(w.value) for w in ws[:-1]]
            cands.append((k, ll, [o for o in older if o and o != ll]))
    if not cands:
        return None
    multi = [c for c in cands if c[2]]
    return rng.choice(multi if multi and rng.random() < 0.6 else cands)


def compressors(model_short, inst):
    hgb = utility_model("hgb", model_short, inst)
    return {"window": RecencyWindow(), "obs_mask": ObservationMasking(), "bm25": BM25Select(),
            "skc": UtilityCompaction(model=hgb)}


def prepare(a):
    rng = random.Random(0)
    samples = []
    for short in SHORT.values():
        corpus = load_corpus(a.cache, short)
        names = [n for n, _ in sorted(corpus.items())][DEV_PER_MODEL:]
        rng.shuffle(names)
        n_take = 0
        for name in names:
            if n_take >= a.per_model:
                break
            turns = corpus[name][0]
            pts, ht = decision_points(turns, n=6, min_tokens=BUDGET)
            pts = [p for p in pts if ht[p] > BUDGET]
            if not pts:
                continue
            i = rng.choice(pts)
            probe = pick_cmd_probe(turns, i, rng)
            action = turns[i + 1].meta.get("action", turns[i + 1].text)
            ctxs = {}
            for mname, comp in compressors(short, name).items():
                ctxs[mname] = render(comp(turns[: i + 1], BUDGET))
            # material for the LLM-summary baseline: older history + recent two turns
            hist = [t for t in turns[2: i + 1]]
            older = "\n".join(f"<turn index={t.idx}>\n{t.text}\n</turn>" for t in hist[:-2])
            recent = "\n".join(f"<turn index={t.idx}>\n{t.text}\n</turn>" for t in hist[-2:])
            samples.append(dict(id=f"{short}/{name}/{i}", model=short, instance=name, point=i,
                                task=turns[1].text, probe=probe, action=action, contexts=ctxs,
                                older=older, recent=recent,
                                history_idents=sorted(set().union(*[set(IDENT_RE.findall(t.text))
                                                                     for t in turns[1: i + 1]]))))
            n_take += 1
        print("prepared", short, n_take, flush=True)
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "samples.jsonl"), "w") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")


def qa_prompt(s, ctx):
    cmd = s["probe"][0][len("cmd:"):]
    return (f"<issue>\n{s['task']}\n</issue>\n<history>\n{ctx}\n</history>\n\n"
            f"The agent ran the command `{cmd}` at some point. Quote, exactly and on a single line, the last "
            f"non-empty line of the output of the MOST RECENT run of this command. If the history does not "
            f"contain that output, answer exactly UNKNOWN. Output only the line.")


def next_prompt(s, ctx):
    return (f"<issue>\n{s['task']}\n</issue>\n<history>\n{ctx}\n</history>\n\n"
            "Predict the agent's next action: write the exact tool call it will issue next (a shell command, "
            "or `str_replace_editor <command> <path> ...` with its arguments). Output only the action.")


def _client():
    import anthropic
    return anthropic.Anthropic()


_THINK = re.compile(r"<think>.*?</think>", re.S)


def _run_local(a, reqs, out_path):
    """OpenAI-compatible chat completions with a thread pool; resumable via out_path."""
    from concurrent.futures import ThreadPoolExecutor

    from openai import OpenAI
    client = OpenAI(base_url=a.base_url, api_key=os.environ.get(a.api_key_env, "local"))
    res = json.load(open(out_path)) if os.path.exists(out_path) else {}
    todo = [r for r in reqs if r["custom_id"] not in res]
    extra = None
    if a.no_think:  # DashScope (Alibaba Bailian) takes a top-level flag; vLLM/SGLang use the chat template
        extra = ({"enable_thinking": False} if "dashscope" in a.base_url
                 else {"chat_template_kwargs": {"enable_thinking": False}})

    def one(r):
        p = r["params"]
        msgs = [{"role": "system", "content": p["system"]}] + p["messages"]
        for attempt in range(3):
            try:
                out = client.chat.completions.create(model=a.model, messages=msgs, temperature=0,
                                                     max_tokens=p["max_tokens"], extra_body=extra)
                return r["custom_id"], _THINK.sub("", out.choices[0].message.content or "").strip()
            except Exception as e:  # noqa: BLE001 - keep going on a single failed request
                err = e
                time.sleep(2 ** attempt)
        print("failed", r["custom_id"], repr(err)[:200], flush=True)
        return r["custom_id"], None

    done = 0
    with ThreadPoolExecutor(a.concurrency) as ex:
        for cid, txt in ex.map(one, todo):
            if txt is not None:
                res[cid] = txt
            done += 1
            if done % 50 == 0:
                json.dump(res, open(out_path, "w"))
                print(f"{done}/{len(todo)}", flush=True)
    json.dump(res, open(out_path, "w"))
    return res


def call_all(a, reqs, out_path):
    if a.backend == "local":
        return _run_local(a, reqs, out_path)
    return _run_batch(_client(), reqs, out_path)


def _req(cid, prompt, max_tokens=1024):
    return {"custom_id": cid, "params": {
        "model": ARGS.model, "max_tokens": max_tokens, "system": SYSTEM,
        "output_config": {"effort": "low"},
        "messages": [{"role": "user", "content": prompt}]}}


def _run_batch(client, reqs, out_path):
    batch = client.messages.batches.create(requests=reqs)
    print("batch", batch.id, len(reqs), flush=True)
    while client.messages.batches.retrieve(batch.id).processing_status != "ended":
        time.sleep(30)
    res = {}
    for r in client.messages.batches.results(batch.id):
        if r.result.type == "succeeded":
            res[r.custom_id] = "".join(b.text for b in r.result.message.content if b.type == "text")
    with open(out_path, "w") as f:
        json.dump(res, f)
    return res


def load_samples(out):
    return [json.loads(l) for l in open(os.path.join(out, "samples.jsonl"))]


def summarise(a):
    from skc.compressors import truncate_text
    samples = load_samples(a.out)
    reqs = []
    for k, s in enumerate(samples):
        cap = BUDGET // 2
        older = s["older"]
        if a.max_context:  # local models: keep head and tail of the history that fits the window
            older = truncate_text(older, max(a.max_context - cap - 2000, 2000))
        prompt = (f"<issue>\n{s['task']}\n</issue>\n<history>\n{older}\n</history>\n\n"
                  f"Summarise this agent history for the agent itself, so that it can continue the task "
                  f"without the original history. Keep file paths, function names, edits made, commands run "
                  f"and their latest results, errors, and open hypotheses. At most {cap} tokens.")
        reqs.append(_req(f"s{k}", prompt, max_tokens=cap))
    res = call_all(a, reqs, os.path.join(a.out, "summaries.json"))
    for k, s in enumerate(samples):
        summ = res.get(f"s{k}", "")
        s["contexts"]["llm_summary"] = f"<summary>\n{summ}\n</summary>\n{s['recent']}"
    with open(os.path.join(a.out, "samples.jsonl"), "w") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")


def submit(a):
    samples = load_samples(a.out)
    reqs = []
    for k, s in enumerate(samples):
        for m, ctx in s["contexts"].items():
            if s["probe"]:
                reqs.append(_req(f"qa|{k}|{m}", qa_prompt(s, ctx), 256))
            reqs.append(_req(f"na|{k}|{m}", next_prompt(s, ctx), 1024))
    call_all(a, reqs, os.path.join(a.out, "answers.json"))


def _paths(text):
    return {p.strip("'\"`") for p in re.findall(r"(?:/[\w.-]+)+\.\w+", text)}


def grade_qa(ans, probe):
    _, latest, older = probe
    a = norm_line(ans.strip().splitlines()[0] if ans.strip() else "")
    if not a or a.upper() == "UNKNOWN":
        return "unknown"
    if latest in a or a in latest and len(a) >= 12:
        return "correct"
    if any(o in a or (a in o and len(a) >= 12) for o in older):
        return "stale"
    return "wrong"


def grade_next(pred, action, hist_idents):
    tp, pp = _paths(action), _paths(pred)
    path_hit = float(bool(tp) and bool(tp & pp)) if tp else None
    ti = {x for x in IDENT_RE.findall(action) if x in hist_idents}
    pi = {x for x in IDENT_RE.findall(pred) if x in hist_idents}
    if not ti:
        return path_hit, None
    p = len(ti & pi) / len(pi) if pi else 0.0
    r = len(ti & pi) / len(ti)
    return path_hit, (2 * p * r / (p + r) if p + r else 0.0)


def collect(a):
    import pandas as pd
    samples = load_samples(a.out)
    ans = json.load(open(os.path.join(a.out, "answers.json")))
    rows = []
    for k, s in enumerate(samples):
        for m in s["contexts"]:
            row = dict(sample=s["id"], model=s["model"], method=m,
                       ctx_tokens=count_tokens(s["contexts"][m]))
            if s["probe"] and f"qa|{k}|{m}" in ans:
                row["qa"] = grade_qa(ans[f"qa|{k}|{m}"], s["probe"])
            if f"na|{k}|{m}" in ans:
                row["path_hit"], row["ident_f1"] = grade_next(ans[f"na|{k}|{m}"], s["action"],
                                                             set(s["history_idents"]))
            rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(a.out, "llm_eval_rows.csv"), index=False)
    qa = df.dropna(subset=["qa"]).groupby("method")["qa"].value_counts(normalize=True).unstack().fillna(0) * 100
    na = df.groupby("method")[["path_hit", "ident_f1"]].mean() * 100
    tok = df.groupby("method")["ctx_tokens"].mean()
    out = qa.join(na).join(tok)
    out.to_csv(os.path.join(a.out, "llm_eval_summary.csv"))
    print(out.round(1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["prepare", "summarise", "submit", "collect"])
    ap.add_argument("--cache")
    ap.add_argument("--out", default="results/llm")
    ap.add_argument("--per-model", type=int, default=60)
    ap.add_argument("--backend", choices=["anthropic", "local"], default="anthropic")
    ap.add_argument("--model", default="claude-opus-5-5", help="model id (anthropic) or served model name (local)")
    ap.add_argument("--base-url", default="http://localhost:8000/v1", help="OpenAI-compatible endpoint (local)")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--api-key-env", default="OPENAI_API_KEY",
                    help="environment variable holding the key of an OpenAI-compatible cloud API "
                         "(e.g. GEMINI_API_KEY with --base-url https://generativelanguage.googleapis.com/v1beta/openai/)")
    ap.add_argument("--max-context", type=int, default=0,
                    help="context window of a local model; long inputs to the summariser are cut to fit")
    ap.add_argument("--no-think", action="store_true", help="disable thinking mode (Qwen3-style chat templates)")
    a = ap.parse_args()
    ARGS = a
    {"prepare": prepare, "summarise": summarise, "submit": submit, "collect": collect}[a.phase](a)
