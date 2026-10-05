import json

import pytest

from skc.compressors import (BM25Select, ObservationMasking, RandomSelect, RecencyWindow,
                             StateKeyedCompaction, TruncateObservations, truncate_text)
from skc.core import CORRECT, MISSING, STALE, ContextIndex, Turn, Unit, Write, count_tokens, read_key
from skc.evaluate import state_probes, use_probes
from skc.sweagent import load_traj
from skc.synthetic import StateTrackConfig, generate, noisy_extractor

ALL = [RecencyWindow(), ObservationMasking(), TruncateObservations(), BM25Select(),
       RandomSelect(), StateKeyedCompaction(), StateKeyedCompaction(extend_mode="bm25")]


def test_reader_outcomes():
    w1 = Write("k", "fast", "k = old-000001", 2)
    w2 = Write("k", "fast", "k = new-000002", 4)
    assert read_key(ContextIndex([Unit(4, "turn", "x\nk = new-000002")]), [w1, w2]) == CORRECT
    assert read_key(ContextIndex([Unit(2, "turn", "k = old-000001")]), [w1, w2]) == STALE
    assert read_key(ContextIndex([Unit(3, "turn", "unrelated line")]), [w1, w2]) == MISSING


def test_line_numbers_are_ignored():
    w = Write("edit:a.py#0", "edit", "def foo(x):\n    return x + 1\n", 3)
    u = Unit(4, "turn", "10:def foo(x):\n11:    return x + 1")
    assert read_key(ContextIndex([u]), [w]) == CORRECT


@pytest.mark.parametrize("seed", range(3))
@pytest.mark.parametrize("budget", [1500, 6000])
def test_budgets_respected(seed, budget):
    turns = generate(StateTrackConfig(steps=60, seed=seed))
    for comp in ALL:
        units = comp(turns, budget)
        charged = sum(u.tokens for u in units if not (u.src == 0 and u.kind == "turn"))
        assert charged <= budget + 16, comp.name


@pytest.mark.parametrize("seed", range(5))
def test_suffix_closed_window_never_stale_on_untruncated_turns(seed):
    # Proposition 2: a compressor that keeps a suffix of the history cannot be stale,
    # except for revisions of pinned keys, whose original value lives in the pinned task.
    turns = generate(StateTrackConfig(steps=80, seed=seed, p_revise_pinned=0.0))
    units = RecencyWindow()(turns, 3000)
    cidx = ContextIndex(units)
    for key, ws in state_probes(turns, len(turns) - 1).items():
        assert read_key(cidx, ws) != STALE


def test_skc_full_recall_retains_everything_when_ledger_fits():
    turns = generate(StateTrackConfig(steps=100, seed=7))
    units = StateKeyedCompaction()(turns, 2000)
    cidx = ContextIndex(units)
    outs = [read_key(cidx, ws) for ws in state_probes(turns, len(turns) - 1).values()]
    assert all(o == CORRECT for o in outs)


def test_missed_latest_write_turns_into_stale():
    turns = generate(StateTrackConfig(steps=100, seed=3, p_fast=0.5))
    ext = noisy_extractor(0.0)
    units = StateKeyedCompaction(extractor=ext, extend=False, recent=0)(turns, 2000)
    assert not [u for u in units if u.kind == "ledger"]


def test_truncate_text_keeps_head_and_tail():
    text = "\n".join(f"line {i} payload" for i in range(200))
    out = truncate_text(text, 60)
    assert out.startswith("line 0 payload") and out.endswith("line 199 payload")
    assert count_tokens(out) <= 70


def _write_traj(tmp_path, steps):
    d = {"history": [{"role": "system", "content": "SYSTEM"},
                     {"role": "user", "content": "<pr_description>\nFix `foo_bar` in mod.py\n</pr_description>"}],
         "trajectory": steps, "info": {"exit_status": "submitted"}}
    p = tmp_path / "x.traj"
    p.write_text(json.dumps(d))
    return str(p)


def test_sweagent_parser_new_interface(tmp_path):
    steps = [
        {"thought": "look", "action": "str_replace_editor view /testbed/mod.py",
         "observation": "Here's the result of running `cat -n` on /testbed/mod.py:\n     1\tdef foo_bar():\n     2\t    return 1"},
        {"thought": "fix", "action": "str_replace_editor str_replace /testbed/mod.py   --old_str '    return 1' --new_str '    return 2  # fixed value'",
         "observation": "The file /testbed/mod.py has been edited."},
        {"thought": "fix again", "action": "str_replace_editor str_replace /testbed/mod.py   --old_str '    return 2  # fixed value' --new_str '    return 3  # final value'",
         "observation": "The file /testbed/mod.py has been edited."},
        {"thought": "run", "action": "cd /testbed && python repro.py", "observation": "Traceback\nValueError: boom happened"},
    ]
    turns, info = load_traj(_write_traj(tmp_path, steps))
    assert turns[1].role == "task" and "foo_bar" in turns[1].text
    keys = [w.key for t in turns for w in t.writes]
    assert keys.count("edit:/testbed/mod.py#0") == 2       # second edit supersedes the first
    assert "cmd:python repro.py" in keys
    assert any(k.startswith("view:/testbed/mod.py@") for k in keys)
    toks = [tok for tok, _ in use_probes(turns, 3)]
    assert "testbed/mod.py" in toks
    assert ContextIndex([Unit(3, "turn", "edited /testbed/mod.py ok")]).has_token("testbed/mod.py")


def test_sweagent_parser_old_interface(tmp_path):
    steps = [
        {"thought": "", "action": "open src/mod.py\n",
         "observation": "[File: /repo/src/mod.py (300 lines total)]\n1:import os\n2:def compute_total(a):"},
        {"thought": "", "action": "edit 2:2\ndef compute_total(a, b):\nend_of_edit\n",
         "observation": "[File: /repo/src/mod.py (300 lines total)]\n1:import os\n2:def compute_total(a, b):"},
        {"thought": "", "action": "edit 2:3\ndef compute_total(a, b, c):\n    return a\nend_of_edit\n",
         "observation": "[File: /repo/src/mod.py (300 lines total)]\n1:import os\n2:def compute_total(a, b, c):"},
        {"thought": "", "action": "edit 250:250\n    print('unrelated region')\nend_of_edit\n",
         "observation": "[File: /repo/src/mod.py (300 lines total)]\n(248 more lines above)\n249:x\n250:    print('unrelated region')"},
    ]
    turns, _ = load_traj(_write_traj(tmp_path, steps))
    keys = [w.key for t in turns for w in t.writes if w.ktype == "edit"]
    assert keys == ["edit:/repo/src/mod.py#0", "edit:/repo/src/mod.py#0", "edit:/repo/src/mod.py#1"]
    views = {w.key for t in turns for w in t.writes if w.ktype == "view"}
    assert views == {"view:/repo/src/mod.py@1", "view:/repo/src/mod.py@201"}
