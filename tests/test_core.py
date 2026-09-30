"""核心模块的正确性测试：全部与暴力枚举对照。"""
from __future__ import annotations

import itertools
import random

import numpy as np
import pytest

from adinterdict.algorithms import greedy, hubcut, ip, mincut
from adinterdict.dominators import single_edge_gains
from adinterdict.flow import global_min_cut
from adinterdict.reduction import reduce_instance

from helpers import brute_force, random_instance

SEEDS = range(40)


@pytest.mark.parametrize("seed", SEEDS)
def test_single_edge_gains_match_recompute(seed):
    inst = random_instance(seed, n=10, p=0.3, max_deletable=30)
    rng = random.Random(seed)
    dels = np.flatnonzero(inst.deletable).tolist()
    F = set(rng.sample(dels, min(2, len(dels))))
    gains = single_edge_gains(inst, F)
    r = inst.risk(F)
    for e in range(inst.m):
        if e in F or not inst.deletable[e]:
            continue
        assert gains[e] == pytest.approx(r - inst.risk(F | {e})), e


@pytest.mark.parametrize("seed", SEEDS)
def test_global_min_cut_is_min_full_separation(seed):
    inst = random_instance(seed, max_deletable=10)
    lam, cut, used = global_min_cut(inst)
    if not used:
        return
    dels = np.flatnonzero(inst.deletable).tolist()
    # 最小割：切断所有“可切断入口”的最小成本。用可达性暴力验证。
    def separated(F):
        mask = inst.alive_mask(F)
        return not inst.reach(used, mask=mask)[inst.targets].any()
    best = min(inst.cost_of(F) for k in range(len(dels) + 1)
               for F in itertools.combinations(dels, k) if separated(F))
    assert lam == best
    assert separated(cut) and inst.cost_of(cut) == lam


@pytest.mark.parametrize("seed", SEEDS)
def test_reduction_preserves_optimum(seed):
    inst = random_instance(seed)
    red, stats = reduce_instance(inst)
    total = int(inst.cost[inst.deletable].sum())
    for budget in (1, 3, total // 2 + 1):
        opt, _ = brute_force(inst, budget)
        opt_red, F_red = brute_force(red, budget)
        assert opt_red == pytest.approx(opt)
        F = red.to_original(F_red)
        assert inst.cost_of(F) == red.cost_of(F_red)
        assert inst.risk(F) == pytest.approx(opt_red)


@pytest.mark.parametrize("seed", SEEDS)
def test_reduced_solution_mapping(seed):
    inst = random_instance(seed, n=12, p=0.3, max_deletable=40)
    red, _ = reduce_instance(inst)
    assert red.risk() == pytest.approx(inst.risk())
    rng = random.Random(seed)
    dels = np.flatnonzero(red.deletable).tolist()
    for _ in range(10):
        F = set(rng.sample(dels, rng.randint(0, len(dels)))) if dels else set()
        assert inst.risk(red.to_original(F)) == pytest.approx(red.risk(F))


@pytest.mark.parametrize("seed", range(25))
def test_ip_matches_brute_force(seed):
    inst = random_instance(seed)
    for budget in (2, 5):
        opt, _ = brute_force(inst, budget)
        sol = ip.solve(inst, budget, time_limit=60)
        assert sol.info["status"] == "optimal"
        assert sol.cost <= budget
        assert inst.risk(sol.removed) == pytest.approx(opt)
        red, _ = reduce_instance(inst)
        sol_r = ip.solve(red, budget, time_limit=60)
        assert inst.risk(sol_r.removed) == pytest.approx(opt)


@pytest.mark.parametrize("seed", range(25))
def test_heuristics_feasible_and_not_better_than_opt(seed):
    inst = random_instance(seed)
    for budget in (2, 5):
        opt, _ = brute_force(inst, budget)
        for sol in (greedy.solve(inst, budget), mincut.solve_b2(inst, budget),
                    hubcut.solve(inst, budget), hubcut.solve(inst, budget, reduce=False)):
            assert sol.cost <= budget
            assert inst.cost_of(sol.removed) == sol.cost
            assert inst.risk(sol.removed) >= opt - 1e-9


@pytest.mark.parametrize("seed", range(25))
def test_lp_is_lower_bound_and_matches_ip_structure(seed):
    from adinterdict.lp import lp_relaxation
    inst = random_instance(seed)
    for budget in (2, 5):
        opt, _ = brute_force(inst, budget)
        lb, x, status = lp_relaxation(inst, budget)
        assert status == "optimal"
        assert lb <= opt + 1e-7
        assert (inst.cost * x).sum() <= budget + 1e-7
    # 预算为 0 时 LP 必须等于 R(空集)
    lb0, _, _ = lp_relaxation(inst, 0)
    assert lb0 == pytest.approx(inst.risk())
