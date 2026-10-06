"""End-to-end downstream evaluation on ALFWorld under a context budget.

An LLM agent plays the 134 out-of-distribution (valid_unseen) ALFWorld text
games. At every step the interaction history is compressed to a token budget
by one of the context-management methods and the agent sees only the
compressed context. The downstream metric is the task success rate.

Methods
  full      no compression (reference; ignores the budget)
  window    task + most recent steps that fit (FIFO truncation)
  obs_mask  observation masking (keep the last 10 observations, then FIFO)
  summary   rolling LLM summary: when the history overflows, the oldest steps
            are folded into a running summary (one extra LLM call), recent
            steps fill the rest of the budget (as in Claude Code / MemGPT)
  skc       SKC with the ALFWorld extractor: ledger of the latest state per
            receptacle / inventory / transformed object + recent steps +
            action skeleton (no LLM calls)

Usage
  python experiments/alfworld_eval.py run --data $ALFWORLD_DATA --out results/alfworld_<model> \
      --backend local --base-url ... --api-key-env ... --model ... --no-think --concurrency 8
  python experiments/alfworld_eval.py report --out results/alfworld_<model>

`--backend expert` replaces the LLM with the PDDL planner (offline plumbing test).
Every finished episode is written to <out>/episodes/ and skipped on re-run.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
import time
import types
import zlib
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from skc.alfworld import AlfWorldState  # noqa: E402
from skc.compressors import (ObservationMasking, RecencyWindow,  # noqa: E402
                             StateKeyedCompaction, _fit_recent, truncate_text)
from skc.core import Turn, Unit, count_tokens  # noqa: E402

TASK_TYPES = ["pick_and_place_simple", "look_at_obj_in_light", "pick_clean_then_place_in_recep",
              "pick_heat_then_place_in_recep", "pick_cool_then_place_in_recep", "pick_two_obj_and_place"]
SHORT_TT = {"pick_and_place_simple": "put", "look_at_obj_in_light": "examine",
            "pick_clean_then_place_in_recep": "clean", "pick_heat_then_place_in_recep": "heat",
            "pick_cool_then_place_in_recep": "cool", "pick_two_obj_and_place": "puttwo"}

SYSTEM = """You are an agent in a household text game. Reach the goal given in "Your task is to".
Available commands (replace X, Y by object or receptacle names, e.g. "apple 1", "fridge 1"):
  go to Y | open Y | close Y | take X from Y | move X to Y | examine Y | inventory | look
  use X (e.g. a desklamp) | clean X with Y (sinkbasin) | heat X with Y (microwave) | cool X with Y (fridge)
  think: <your reasoning or notes>   (does not change the world; the game replies "OK.")
You must be at a receptacle (go to Y) before you can take from or move to it, and you can hold one object at a time.
"Nothing happens." means the command was invalid. Your memory of the past is limited to what is shown below.
Reply with exactly one command on one line and nothing else."""


def _stub_jericho():
    """TextWorld imports jericho (Z-machine games) at module level; ALFWorld does not use it."""
    if "jericho" not in sys.modules:
        try:
            import jericho  # noqa: F401
        except ImportError:
            j = types.ModuleType("jericho")
            j.__getattr__ = lambda n: type(n, (Warning,), {})
            sys.modules["jericho"] = j


def game_files(data):
    files = sorted(glob.glob(os.path.join(data, "json_2.1.1", "valid_unseen", "*", "*", "game.tw-pddl")))
    out = []
    for f in files:
        if "movable" in f or "Sliced" in f:
            continue
        if json.load(open(f)).get("solvable", False):
            out.append(f)
    return out


def game_id(path):
    trial, task = os.path.basename(os.path.dirname(path)), os.path.basename(os.path.dirname(os.path.dirname(path)))
    return f"{task}/{trial}"


def task_type(gid):
    return next(t for t in TASK_TYPES if gid.startswith(t))


# --------------------------------------------------------------------------- context
def ledger_fmt(k, w, val):
    return f"[known, step {w.turn // 2}] {val}"


def render(units, n_hist_turns):
    """Prompt text for a list of context units; gaps in the history are marked."""
    out, prev, in_ledger = [], 0, False
    for u in units:
        if u.kind == "ledger":
            if not in_ledger:
                out.append("Known state (from earlier observations):")
                in_ledger = True
            out.append(u.text)
            continue
        if u.src == 0:
            out.append(u.text.strip()); out.append("")
            continue
        if in_ledger or not prev:
            out.append(""); out.append("History:")
            in_ledger = False
        if u.src > prev + 1 and prev >= 0:
            out.append(f"[... {(u.src - prev - 1) // 2} earlier step(s) not shown ...]")
        out.append(u.text)
        prev = u.src
    return "\n".join(out).strip()


def history_text(turns):
    return "\n".join(t.text for t in turns)


def make_compressor(method):
    if method == "window":
        return RecencyWindow()
    if method == "obs_mask":
        return ObservationMasking(keep_obs=10)
    if method == "skc":
        return StateKeyedCompaction(recent=4, value_cap=120, fmt=ledger_fmt)
    if method == "skc_noledger":
        return StateKeyedCompaction(recent=4, use_ledger=False, name="skc_noledger")
    return None


SUMMARY_PROMPT = """You are helping an agent in a household text game keep track of its progress under a short memory.
Task and starting room:
{task}

Previous summary of earlier steps:
{summary}

Steps that are about to be dropped from memory:
{steps}

Write an updated summary (at most {words} words) that will replace both the previous summary and these steps.
Keep everything needed to finish the task: which receptacles were checked and what they contain (or that they
were empty or closed), where relevant objects are, what the agent holds, and which subgoals are done.
Reply with the summary only."""


# --------------------------------------------------------------------------- LLM
_THINK = re.compile(r"<think>.*?</think>", re.S)
_client = None


def llm(a, system, user, max_tokens):
    global _client
    from openai import OpenAI
    if _client is None:
        _client = OpenAI(base_url=a["base_url"], api_key=os.environ.get(a["api_key_env"], "local"),
                         timeout=120)
    extra = None
    if a["no_think"]:
        extra = ({"enable_thinking": False} if "dashscope" in a["base_url"]
                 else {"chat_template_kwargs": {"enable_thinking": False}})
    err = None
    for attempt in range(6):
        try:
            out = _client.chat.completions.create(
                model=a["model"], temperature=0, max_tokens=max_tokens + a["think_tokens"], extra_body=extra,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}])
            txt = _THINK.sub("", out.choices[0].message.content or "").strip()
            u = out.usage
            return txt, (u.prompt_tokens if u else 0), (u.completion_tokens if u else 0)
        except Exception as e:  # noqa: BLE001 - retry transient API errors
            err = e
            time.sleep(min(2 ** attempt, 30))
    raise RuntimeError(f"LLM call failed: {err!r}"[:300])


_CMD = re.compile(r"^(?:action|command|next action|>)\s*[:>]?\s*", re.I)


def parse_action(reply):
    for ln in reply.split("\n"):
        ln = ln.strip().strip("`").strip()
        if ln:
            break
    else:
        return "look"
    ln = _CMD.sub("", ln).strip()
    if ln.lower().startswith("think"):
        return "think: " + re.sub(r"^think\s*:?\s*", "", ln, flags=re.I)
    ln = ln.lower().rstrip(".").strip()
    ln = re.sub(r"^put (.+?) (?:in|on|in/on|into|onto) (.+)$", r"move \1 to \2", ln)
    return re.sub(r"\s+", " ", ln)


# --------------------------------------------------------------------------- episode
def run_episode(gamefile, method, budget, a):
    _stub_jericho()
    import textworld
    from alfworld.agents.environment.alfred_tw_env import AlfredDemangler, AlfredInfos

    infos = textworld.EnvInfos(won=True, admissible_commands=True, policy_commands=a["backend"] == "expert")
    env = textworld.start(gamefile, infos, wrappers=[AlfredDemangler, AlfredInfos])
    gs = env.reset()
    obs0 = gs.feedback.replace("-= Welcome to TextWorld, ALFRED! =-", "").strip()
    turns = [Turn(0, "task", obs0)]
    world = AlfWorldState()
    comp = make_compressor(method)
    summary, summarised, n_summ = "", 1, 0   # summary state: turns[1:summarised] folded into `summary`
    steps, won, done = [], False, False
    in_tok = out_tok = summ_in = summ_out = 0
    t0 = time.time()
    rng = __import__("random").Random(zlib.crc32(gamefile.encode()))
    for step in range(a["max_steps"]):
        hist = turns[1:]
        if method == "summary":   # fold the oldest steps into the running summary on overflow
            task_cost, cap = turns[0].tokens, budget // 3
            live = turns[summarised:]
            if task_cost + count_tokens(summary) + sum(t.tokens for t in live) > budget:
                keep, _ = _fit_recent(live, (budget - task_cost - cap) // 2)
                cut = keep[0].src if keep else len(turns)
                drop = turns[summarised:cut]
                if drop and a["backend"] != "expert":
                    prompt = SUMMARY_PROMPT.format(task=obs0, summary=summary or "(none)",
                                                   steps=history_text(drop), words=int(cap * 0.6))
                    s, pi, po = llm(a, "You summarise agent histories faithfully and concisely.", prompt, cap)
                    summary = truncate_text(s, cap)
                    summ_in += pi; summ_out += po; n_summ += 1
                summarised = cut

        full_ctx = render([Unit(t.idx, "turn", t.text, t.tokens) for t in turns], len(hist))

        def build(b):
            if method == "summary":
                rec, _ = _fit_recent(turns[summarised:], b - turns[0].tokens - count_tokens(summary))
                parts = [obs0, ""]
                if summary:
                    parts += ["Summary of earlier steps:", summary, ""]
                first = rec[0].src if rec else len(turns)
                parts.append("History:")
                if first > 1:
                    parts.append(f"[... {(first - 1) // 2} earlier step(s) not shown ...]")
                return "\n".join(parts + [u.text for u in rec]).strip()
            return render(comp(turns, b), len(hist))

        # the budget applies to the rendered prompt context (separators and headers included)
        # SKC compacts only on overflow; window / masking / summary apply their own policy
        if method == "full" or (method.startswith("skc") and count_tokens(full_ctx) <= budget):
            ctx = full_ctx
        else:
            b = budget
            for _ in range(12):
                ctx = build(b)
                over = count_tokens(ctx) - budget
                if over <= 0:
                    break
                b -= over + 4
        ctx_tokens = count_tokens(ctx)
        if a["backend"] == "expert":   # planner, with random detours (`--noise`) to lengthen episodes
            if rng.random() < a["noise"]:
                reply = rng.choice(gs.admissible_commands)
            else:
                reply = (gs.policy_commands or ["look"])[0]
            pi = po = 0
        else:
            reply, pi, po = llm(a, SYSTEM, ctx + "\n\nYour next command:", a["max_out"])
        in_tok += pi; out_tok += po
        action = parse_action(reply)
        if action.startswith("think:"):
            obs = "OK."
        else:
            gs, _, done = env.step(action)
            obs = gs.feedback.strip()
            won = bool(gs.won)
        n = len(turns)
        turns.append(Turn(n, "assistant", f"> {action}"))
        turns.append(Turn(n + 1, "tool", obs, writes=world.update(action, obs, n + 1)))
        steps.append({"action": action, "obs": obs, "ctx_tokens": ctx_tokens, "reply": reply[:300]})
        if a["dump_context"]:
            steps[-1]["context"] = ctx
        if won or done:
            break
    env.close()
    return {"game": game_id(gamefile), "task_type": task_type(game_id(gamefile)), "method": method,
            "budget": budget, "won": won, "n_steps": len(steps),
            "n_env_actions": sum(not s["action"].startswith("think:") for s in steps),
            "n_invalid": sum(s["obs"] == "Nothing happens." for s in steps),
            "mean_ctx_tokens": sum(s["ctx_tokens"] for s in steps) / max(len(steps), 1),
            "max_ctx_tokens": max((s["ctx_tokens"] for s in steps), default=0),
            "history_tokens": sum(t.tokens for t in turns),
            "prompt_tokens": in_tok, "completion_tokens": out_tok,
            "summary_calls": n_summ, "summary_prompt_tokens": summ_in, "summary_completion_tokens": summ_out,
            "seconds": round(time.time() - t0, 1), "model": a["model"], "steps": steps}


def _job(args):
    gamefile, method, budget, a, path = args
    try:
        res = run_episode(gamefile, method, budget, a)
    except Exception as e:  # noqa: BLE001 - one failed episode must not stop the run
        return path, None, repr(e)[:300]
    tmp = path + ".tmp"
    json.dump(res, open(tmp, "w"))
    os.replace(tmp, path)
    return path, res["won"], None


def ep_path(out, method, budget, gid):
    b = "inf" if method == "full" else str(budget)
    return os.path.join(out, "episodes", f"{method}_{b}", gid.replace("/", "__") + ".json")


def run(a):
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    files = game_files(a.data)
    if a.games:
        files = files[: a.games]
    print(f"{len(files)} games", flush=True)
    cfg = {k: getattr(a, k) for k in ["backend", "base_url", "api_key_env", "model", "no_think",
                                       "think_tokens", "max_steps", "max_out", "noise", "dump_context"]}
    jobs = []
    for method in a.methods:
        for budget in ([0] if method == "full" else a.budgets):
            for f in files:
                p = ep_path(a.out, method, budget, game_id(f))
                if not os.path.exists(p):
                    os.makedirs(os.path.dirname(p), exist_ok=True)
                    jobs.append((f, method, budget, cfg, p))
    # interleave methods so that partial runs stay comparable across methods
    jobs.sort(key=lambda j: (zlib.crc32(game_id(j[0]).encode()), j[1], j[2]))
    print(f"{len(jobs)} episodes to run", flush=True)
    done = fails = 0
    # fast-downward copies its 32 MB libdownward.so to a temp dir on every game load and never
    # unloads it, so the deleted copies hold disk space until the worker exits; recycle workers
    with ProcessPoolExecutor(a.concurrency, max_tasks_per_child=20) as ex:
        futs = [ex.submit(_job, j) for j in jobs]
        for fu in as_completed(futs):
            path, won, err = fu.result()
            done += 1
            if err:
                fails += 1
                print("failed", path, err, flush=True)
            if done % 20 == 0 or done == len(jobs):
                print(f"{done}/{len(jobs)} episodes ({fails} failed)", flush=True)


# --------------------------------------------------------------------------- report
def load(out):
    rows = []
    for p in glob.glob(os.path.join(out, "episodes", "*", "*.json")):
        r = json.load(open(p))
        r.pop("steps", None)
        rows.append(r)
    return rows


def report(a):
    import numpy as np
    import pandas as pd
    from scipy.stats import binomtest

    df = pd.DataFrame(load(a.out))
    if df.empty:
        print("no episodes"); return
    df["cfg"] = df.apply(lambda r: "full" if r.method == "full" else f"{r.method}@{r.budget}", axis=1)
    rng = np.random.default_rng(0)
    rows = []
    for cfg, g in df.groupby("cfg"):
        w = g.won.astype(float).values
        bs = [rng.choice(w, len(w)).mean() for _ in range(2000)]
        tok = g.prompt_tokens + g.summary_prompt_tokens
        rows.append({"cfg": cfg, "method": g.method.iloc[0], "budget": g.budget.iloc[0], "n": len(g),
                     "success": 100 * w.mean(), "ci_lo": 100 * np.percentile(bs, 2.5),
                     "ci_hi": 100 * np.percentile(bs, 97.5), "steps": g.n_steps.mean(),
                     "invalid": g.n_invalid.mean(), "ctx_tokens": g.mean_ctx_tokens.mean(),
                     "input_tokens_k": tok.mean() / 1000, "summary_calls": g.summary_calls.mean(),
                     **{SHORT_TT[t]: 100 * g[g.task_type == t].won.mean() for t in TASK_TYPES}})
    tab = pd.DataFrame(rows).sort_values(["budget", "method"])
    # paired exact (McNemar) tests of SKC against every other method at the same budget
    tests = []
    for b in sorted(df.budget.unique()):
        skc = df[(df.method == "skc") & (df.budget == b)].set_index("game").won
        if skc.empty:
            continue
        for cfg in sorted(df.cfg.unique()):
            if cfg.startswith("skc@"):
                continue
            if cfg != "full" and not cfg.endswith(f"@{b}"):
                continue
            o = df[df.cfg == cfg].set_index("game").won
            common = skc.index.intersection(o.index)
            x, y = skc[common].astype(bool), o[common].astype(bool)
            b10, b01 = int((x & ~y).sum()), int((~x & y).sum())
            p = binomtest(b10, b10 + b01, 0.5).pvalue if b10 + b01 else 1.0
            tests.append({"budget": b, "vs": cfg, "n": len(common), "skc": 100 * x.mean(), "other": 100 * y.mean(),
                          "diff": 100 * (x.mean() - y.mean()), "skc_only": b10, "other_only": b01, "p_mcnemar": p})
    os.makedirs(a.out, exist_ok=True)
    tab.to_csv(os.path.join(a.out, "alfworld_summary.csv"), index=False)
    pd.DataFrame(tests).to_csv(os.path.join(a.out, "alfworld_tests.csv"), index=False)
    pd.set_option("display.width", 200)
    print(tab.round(1).to_string(index=False))
    print()
    print(pd.DataFrame(tests).round(3).to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["run", "report"])
    ap.add_argument("--data", default=os.environ.get("ALFWORLD_DATA", "data/alfworld"),
                    help="directory containing json_2.1.1/ (ALFWorld TW-PDDL game files)")
    ap.add_argument("--out", default="results/alfworld")
    ap.add_argument("--methods", nargs="+", default=["full", "window", "obs_mask", "summary", "skc"])
    ap.add_argument("--budgets", nargs="+", type=int, default=[600, 1200],
                    help="context budgets in tokens (task + history; the fixed system prompt is not charged)")
    ap.add_argument("--games", type=int, default=0, help="only the first N games (0 = all 134)")
    ap.add_argument("--max-steps", type=int, default=50)
    ap.add_argument("--max-out", type=int, default=64, help="max output tokens per command")
    ap.add_argument("--backend", choices=["local", "expert"], default="local")
    ap.add_argument("--model", default="qwen3.8-flash")
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--api-key-env", default="OPENAI_API_KEY")
    ap.add_argument("--no-think", action="store_true")
    ap.add_argument("--think-tokens", type=int, default=0)
    ap.add_argument("--noise", type=float, default=0.0, help="expert backend: probability of a random command")
    ap.add_argument("--dump-context", action="store_true", help="store the prompt context of every step")
    ap.add_argument("--concurrency", type=int, default=8, help="episodes played in parallel")
    a = ap.parse_args()
    {"run": run, "report": report}[a.phase](a)
