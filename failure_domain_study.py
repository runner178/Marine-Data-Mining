"""Stress test: correlated probe outages within shared AIS log domains."""
import argparse
import json
import os
from pathlib import Path

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '7')
import cupy as cp
import numpy as np
import ais_fault_study as base

DOMAIN = np.asarray([0, 0, 1, 0, 0, 2, 2, 0], np.int32)
DOMAIN_FAILURE = np.asarray([.25, .08, .25], np.float32)
INDEPENDENT_FAILURE = .05


def missing_evidence(signals, seed):
    rng = np.random.default_rng(seed)
    outage = rng.random((len(signals), 3)) < DOMAIN_FAILURE
    mask = outage[:, DOMAIN] | (rng.random(signals.shape) < INDEPENDENT_FAILURE)
    return np.where(mask, -1, signals).astype(np.int8)


def run_domain_aware(labels, evidence, p, availability, gate):
    n = len(labels); rows = cp.arange(n); domain = cp.asarray(DOMAIN)
    belief = cp.full((n, 6), 1 / 6, cp.float32)
    active = cp.ones(n, cp.bool_)
    observed = cp.zeros(n, cp.int8)
    used = cp.zeros((n, 8), cp.bool_)
    available = cp.broadcast_to(availability, (n, 8)).copy()
    calls = cp.zeros(n, cp.int8)
    cost_total = cp.zeros(n, cp.float32)
    cost = cp.asarray(base.COST)
    conditional = base.entropy(p)
    for _ in range(base.BUDGET):
        mi = base.entropy(belief @ p) - belief @ conditional
        tool = cp.argmax(cp.where(used, -cp.inf, mi * available / cost[None, :]), axis=1).astype(cp.int32)
        signal = evidence[rows, tool]
        missing = active & (signal < 0)
        sibling = domain[None, :] == domain[tool, None]
        available = cp.where(missing[:, None] & sibling, .05, available)
        used[rows, tool] = active
        calls += active; cost_total += cp.where(active, cost[tool], 0)
        observed += active & (signal >= 0)
        likelihood = cp.where(signal[:, None] == 1, p[:, tool].T, 1 - p[:, tool].T)
        updated = belief * likelihood
        updated /= cp.sum(updated, axis=1, keepdims=True)
        belief = cp.where((active & (signal >= 0))[:, None], updated, belief)
        active &= ~((observed >= 2) & (cp.max(belief, axis=1) >= gate))
    confirm = (observed >= 2) & (cp.max(belief, axis=1) >= gate)
    correct = confirm & (cp.argmax(belief, axis=1) == labels)
    return {'correct': int(cp.sum(correct).get()),
            'false_confirmed': int(cp.sum(confirm & ~correct).get()),
            'escalated': int(cp.sum(~confirm).get()),
            'mean_calls': float(cp.mean(calls).get()),
            'mean_cost': float(cp.mean(cost_total).get())}, cp.asnumpy(correct)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gate', type=float, default=.8)
    ap.add_argument('--out', type=Path, default=Path('fault_results/failure_domain.json'))
    args = ap.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '7' or cp.cuda.runtime.getDeviceCount() != 1:
        raise RuntimeError('Use only physical GPU 7')
    data = base.metadata(Path('formatted_trajectories.json.gz'), Path('fault_results/metadata.npz'))
    tr_y, tr_s, _ = base.episodes(data, 2020, 100)
    va_y, va_s, _ = base.episodes(data, 2021, 200)
    te_y, te_s, groups = base.episodes(data, 2022, 300)
    counts = np.stack([tr_s[tr_y == c].sum(axis=0) for c in range(6)])
    totals = np.asarray([np.sum(tr_y == c) for c in range(6)])[:, None]
    p = cp.asarray((counts + 1) / (totals + 2), cp.float32)
    val_ev = missing_evidence(va_s, 20261122)
    test_ev = missing_evidence(te_s, 20261123)
    missing_corr = np.corrcoef((val_ev < 0).astype(float), rowvar=False)
    same_domain_pairs = [(i, j) for i in range(8) for j in range(i + 1, 8) if DOMAIN[i] == DOMAIN[j]]
    mean_same_domain_corr = float(np.mean([missing_corr[i, j] for i, j in same_domain_pairs]))
    use_domain_aware = mean_same_domain_corr > .2
    availability = cp.asarray(np.mean(val_ev >= 0, axis=0).astype(np.float32))
    order, selection = base.optimize_fixed_order(va_y, val_ev, p, args.gate)
    y, e = cp.asarray(te_y), cp.asarray(test_ev)
    results = {}; correct = {}
    for name, method in [('optimized_fixed', 'fixed'), ('online_mi', 'online'), ('static_cost_aware', 'cost_aware')]:
        results[name], correct[name], _, _ = base.run_policy(y, e, p, method, order, availability, args.gate)
    results['domain_aware'], correct['domain_aware'] = run_domain_aware(y, e, p, availability, args.gate)
    guarded_name = 'domain_aware' if use_domain_aware else 'static_cost_aware'
    results['guarded'] = results[guarded_name].copy()
    result = {'device': cp.cuda.runtime.getDeviceProperties(0)['name'].decode(),
              'visible_device': '7', 'gate': args.gate, 'cases': len(te_y),
              'domain_assignment': DOMAIN.tolist(),
              'domain_failure_probability': DOMAIN_FAILURE.tolist(),
              'independent_failure_probability': INDEPENDENT_FAILURE,
              'validation_availability': cp.asnumpy(availability).tolist(),
              'validation_mean_same_domain_missing_correlation': mean_same_domain_corr,
              'guarded_choice': guarded_name,
              'optimized_fixed_order': order, 'validation_selection': selection,
              'policies': results,
              'domain_aware_vs_static_correct': base.cluster_interval(correct['domain_aware'].astype(float), correct['static_cost_aware'].astype(float), groups // 6),
              'domain_aware_vs_online_correct': base.cluster_interval(correct['domain_aware'].astype(float), correct['online_mi'].astype(float), groups // 6)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
