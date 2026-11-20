"""GPU paired experiment for missing-probe and cost-aware AIS diagnosis policies."""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from agent_experiment import PRIOR, POSITIVE, ROUTES, MISSING

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '7')
import cupy as cp

TOOLS = 8
CAUSES = 6
BUDGET = 4
MIN_OBSERVED = 2
GATE = 0.85
N = 100_000
CALIBRATION = 10_000
COST = np.array([1.0, 2.0, 1.5, 3.0, 2.5, 2.0, 1.0, 1.5])


def entropy(p):
    p = cp.clip(p, 1e-10, 1 - 1e-10)
    return -p * cp.log2(p) - (1 - p) * cp.log2(1 - p)


def make_cases(seed, n, missing, p):
    rng = cp.random.RandomState(seed)
    labels = cp.searchsorted(cp.cumsum(cp.asarray(PRIOR)), rng.random_sample(n))
    positive = rng.random_sample((n, TOOLS)) < p[labels]
    unavailable = rng.random_sample((n, TOOLS)) < cp.asarray(missing)
    evidence = cp.where(unavailable, -1, positive.astype(cp.int8))
    return labels, evidence


def policy_run(labels, evidence, kind, availability, p, cost):
    n = len(labels)
    ix = cp.arange(n)
    belief = cp.broadcast_to(cp.asarray(PRIOR), (n, CAUSES)).copy()
    used = cp.zeros((n, TOOLS), dtype=cp.bool_)
    active = cp.ones(n, dtype=cp.bool_)
    observed = cp.zeros(n, dtype=cp.int8)
    calls = cp.zeros(n, dtype=cp.int8)
    total_cost = cp.zeros(n, dtype=cp.float32)
    first = cp.zeros(n, dtype=cp.int8)
    routes = cp.asarray([ROUTES[-1], ROUTES[0], ROUTES[1]], dtype=cp.int32)
    conditional = entropy(p)
    for step in range(BUDGET):
        if kind == 'workflow':
            tool = cp.zeros(n, dtype=cp.int32) if step == 0 else routes[first + 1, step - 1]
        else:
            marginal = belief @ p
            mi = entropy(marginal) - belief @ conditional
            if kind == 'aware':
                mi = mi * availability[None, :] / cost[None, :]
            mi = cp.where(used, -cp.inf, mi)
            tool = cp.argmax(mi, axis=1).astype(cp.int32)
        signal = evidence[ix, tool]
        if step == 0:
            first = signal.copy()
        calls += active
        total_cost += cp.where(active, cost[tool], 0)
        used[ix, tool] |= active
        observed += active & (signal >= 0)
        likelihood_positive = p[:, tool].T
        likelihood = cp.where(signal[:, None] == 1, likelihood_positive, 1 - likelihood_positive)
        weight = belief * likelihood
        posterior = weight / cp.sum(weight, axis=1, keepdims=True)
        belief = cp.where((active & (signal >= 0))[:, None], posterior, belief)
        active &= ~((observed >= MIN_OBSERVED) & (cp.max(belief, axis=1) >= GATE))
    confirm = (observed >= MIN_OBSERVED) & (cp.max(belief, axis=1) >= GATE)
    prediction = cp.argmax(belief, axis=1)
    correct = confirm & (prediction == labels)
    out = {
        'correct': int(cp.sum(correct).get()),
        'confirmed': int(cp.sum(confirm).get()),
        'false_confirmed': int(cp.sum(confirm & ~correct).get()),
        'escalated': int(cp.sum(~confirm).get()),
        'calls_mean': float(cp.mean(calls).get()),
        'cost_mean': float(cp.mean(total_cost).get()),
        'calls_histogram': cp.bincount(calls.astype(cp.int32), minlength=BUDGET + 1).get().tolist(),
    }
    return out, correct


def paired(a, b, n):
    plus = int(cp.sum(a & ~b).get())
    minus = int(cp.sum(~a & b).get())
    diff = (plus - minus) / n
    se = (((plus + minus) / n - diff * diff) / n) ** 0.5
    return {'difference': diff, 'ci95_normal_approx': [diff - 1.96 * se, diff + 1.96 * se],
            'a_only': plus, 'b_only': minus}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--cases', type=int, default=N)
    ap.add_argument('--calibration', type=int, default=CALIBRATION)
    args = ap.parse_args()
    if cp.cuda.runtime.getDeviceCount() != 1:
        raise RuntimeError('Set CUDA_VISIBLE_DEVICES=7; expected exactly one visible GPU')
    cp.cuda.Device(0).use()
    p = cp.asarray(POSITIVE, dtype=cp.float32)
    cost = cp.asarray(COST, dtype=cp.float32)
    result = {'device': cp.cuda.runtime.getDeviceProperties(0)['name'].decode(),
              'visible_device': os.environ.get('CUDA_VISIBLE_DEVICES'),
              'cases_per_setting': args.cases, 'calibration_cases': args.calibration,
              'budget': BUDGET, 'confidence_gate': GATE, 'cost': COST.tolist(),
              'settings': {}}
    start = time.monotonic()
    for setting, missing in MISSING.items():
        labels, evidence = make_cases(20261115 + (setting == 'degraded'), args.cases, missing, p)
        calibration_labels, calibration = make_cases(20261117 + (setting == 'degraded'), args.calibration, missing, p)
        estimated_availability = cp.mean(calibration >= 0, axis=0)
        normal_availability = cp.asarray([1 - x for x in MISSING['normal']], dtype=cp.float32)
        calibration_fixed, _ = policy_run(calibration_labels, calibration, 'aware', normal_availability, p, cost)
        calibration_recalibrated, _ = policy_run(calibration_labels, calibration, 'aware', estimated_availability, p, cost)
        use_recalibrated = (calibration_recalibrated['correct'] > calibration_fixed['correct']
                            and calibration_recalibrated['false_confirmed'] <= calibration_fixed['false_confirmed'])
        gated_availability = estimated_availability if use_recalibrated else normal_availability
        policies = {'workflow': ('workflow', normal_availability),
                    'online_mi': ('online', normal_availability),
                    'aware_fixed': ('aware', normal_availability),
                    'aware_recalibrated': ('aware', estimated_availability),
                    'aware_gated': ('aware', gated_availability)}
        scores, correctness = {}, {}
        for name, (kind, available) in policies.items():
            scores[name], correctness[name] = policy_run(labels, evidence, kind, available, p, cost)
        result['settings'][setting] = {
            'estimated_availability': cp.asnumpy(estimated_availability).tolist(),
            'calibration_fixed': calibration_fixed,
            'calibration_recalibrated': calibration_recalibrated,
            'gated_choice': 'recalibrated' if use_recalibrated else 'fixed',
            'policies': scores,
            'paired_aware_recalibrated_vs_workflow': paired(correctness['aware_recalibrated'], correctness['workflow'], args.cases),
            'paired_aware_recalibrated_vs_online_mi': paired(correctness['aware_recalibrated'], correctness['online_mi'], args.cases),
            'paired_aware_recalibrated_vs_aware_fixed': paired(correctness['aware_recalibrated'], correctness['aware_fixed'], args.cases),
        }
        cp.cuda.Stream.null.synchronize()
    result['elapsed_seconds'] = time.monotonic() - start
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
