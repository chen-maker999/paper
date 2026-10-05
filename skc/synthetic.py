"""StateTrack: a controllable generator of agent-like trajectories with
known state dynamics.

Three key families model what long-horizon agents must remember:
  pinned  - constraints given by the user up front (rarely revised, but a
            revision must be honoured);
  slow    - facts discovered by tools that change occasionally (config, paths);
  fast    - frequently changing status (counters, test results, plan step).
Writes are embedded in tool outputs (and some in agent turns) among
distractor log lines. Every value is unique, so the ideal reader is exact.
"""
from __future__ import annotations

import random
import zlib
from dataclasses import dataclass

from .core import Turn, Write

_WORDS = ("alpha beta gamma delta sigma omega vector cache shard batch queue node worker "
          "region bucket schema index replica lease token parser buffer socket stream "
          "kernel module handler router metric sample trace span commit branch").split()
_TOOLS = ["search_docs", "read_file", "run_tests", "query_db", "http_get", "list_dir", "grep_logs"]


@dataclass
class StateTrackConfig:
    steps: int = 120
    n_pinned: int = 4
    n_slow: int = 12
    n_fast: int = 6
    p_slow: float = 0.03        # per key, per step write probability
    p_fast: float = 0.25
    p_revise_pinned: float = 0.004
    obs_lines_mean: float = 25  # distractor lines per tool output (lognormal mean)
    obs_lines_sigma: float = 0.8
    p_write_in_agent: float = 0.2
    seed: int = 0


def _val(rng):
    return f"{rng.choice(_WORDS)}-{rng.randrange(10**6):06d}"


def _noise_line(rng):
    w = rng.sample(_WORDS, 3)
    return (f"{rng.randrange(24):02d}:{rng.randrange(60):02d}:{rng.randrange(60):02d} "
            f"INFO {w[0]}-{rng.randrange(16)} {w[1]} {w[2]} id={rng.randrange(10**7)} "
            f"took {rng.randrange(1, 900)}ms")


def generate(cfg: StateTrackConfig):
    rng = random.Random(cfg.seed)
    keys = ([("pinned", f"constraint.{rng.choice(_WORDS)}_{i}") for i in range(cfg.n_pinned)]
            + [("slow", f"config.{rng.choice(_WORDS)}_{i}") for i in range(cfg.n_slow)]
            + [("fast", f"status.{rng.choice(_WORDS)}_{i}") for i in range(cfg.n_fast)])
    turns = [Turn(0, "system", "You are an agent. Use tools to complete the task.")]
    task_lines = ["Task: migrate the service and keep the following constraints."]
    task = Turn(1, "task", "")
    for fam, k in keys:
        if fam == "pinned":
            v = _val(rng)
            task_lines.append(f"{k} = {v}")
            task.writes.append(Write(k, fam, f"{k} = {v}", 1))
    task.text = "\n".join(task_lines)
    task.tokens = -1
    task.__post_init__()
    turns.append(task)
    for step in range(cfg.steps):
        tool = rng.choice(_TOOLS)
        agent_writes, obs_writes, user_writes = [], [], []
        for fam, k in keys:
            if fam == "pinned":
                if rng.random() < cfg.p_revise_pinned:
                    user_writes.append((fam, k))
                continue
            p = cfg.p_slow if fam == "slow" else cfg.p_fast
            if rng.random() < p:
                (agent_writes if rng.random() < cfg.p_write_in_agent else obs_writes).append((fam, k))
        if user_writes:
            t = Turn(len(turns), "user", "")
            lines = ["User: please update the constraints:"]
            for fam, k in user_writes:
                v = _val(rng); lines.append(f"{k} = {v}")
                t.writes.append(Write(k, fam, f"{k} = {v}", t.idx))
            t.text = "\n".join(lines); t.tokens = -1; t.__post_init__()
            turns.append(t)
        a = Turn(len(turns), "assistant", "")
        lines = [f"Thought: step {step}, calling {tool} to make progress on {rng.choice(_WORDS)}.",
                 f"Action: {tool}(\"{rng.choice(_WORDS)}/{rng.choice(_WORDS)}\")"]
        for fam, k in agent_writes:
            v = _val(rng); lines.append(f"Note: {k} = {v}")
            a.writes.append(Write(k, fam, f"Note: {k} = {v}", a.idx))
        a.text = "\n".join(lines); a.tokens = -1; a.__post_init__()
        turns.append(a)
        o = Turn(len(turns), "tool", "")
        n = max(1, int(rng.lognormvariate(0, cfg.obs_lines_sigma) * cfg.obs_lines_mean))
        lines = [_noise_line(rng) for _ in range(n)]
        for fam, k in obs_writes:
            v = _val(rng)
            lines.insert(rng.randrange(len(lines) + 1), f"{k} = {v}")
            o.writes.append(Write(k, fam, f"{k} = {v}", o.idx))
        o.text = "\n".join(lines); o.tokens = -1; o.__post_init__()
        turns.append(o)
    return turns


def noisy_extractor(recall: float, seed: int = 0):
    """Extractor that detects each write independently with probability `recall`
    (deterministic in the write identity), modelling an imperfect LLM/rule extractor."""
    def ext(turn):
        out = []
        for w in turn.writes:
            h = zlib.crc32(f"{w.key}|{w.turn}|{seed}".encode()) / 2**32
            if h < recall:
                out.append(w)
        return out
    return ext
