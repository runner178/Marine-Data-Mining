"""CPU-only robustness sweep for the shared-failure-domain idea.

All fault labels and outages are injected into real AIS episode backgrounds.
The 2021 split chooses the correlation gate; 2022 is held out.
"""
import argparse
import json
from pathlib import Path

import numpy as np

import ais_fault_study as base

DOMAIN = np.array([0, 0, 1, 0, 0, 2, 2, 0])
Q_VALUES = [0.0, 0.05, 0.10, 0.15, 0.25, 0.35]


def evidence(signals, q, seed):
    rng = np.random.default_rng(seed)
    domain_p = np.array([q, 0.32 * q, q])
    outage = rng.random((len(signals), 3)) < domain_p
    mask = outage[:, DOMAIN] | (rng.random(signals.shape) < .05)
    return np.where(mask, -1, signals).astype(np.int8)


def binary_entropy(x):
    x = np.clip(x, 1e-8, 1 - 1e-8)
    return -x * np.log2(x) - (1 - x) * np.log2(1 - x)


def evaluate(labels, ev, p, availability, aware, gate=.8):
    n = len(labels)
    rows = np.arange(n)
    belief = np.full((n, 6), 1 / 6, np.float32)
    used = np.zeros((n, 8), bool)
    active = np.ones(n, bool)
    observed = np.zeros(n, np.int8)
    calls = np.zeros(n, np.int8)
    cost_total = np.zeros(n, np.float32)
    available = np.broadcast_to(availability, (n, 8)).copy()
    conditional = binary_entropy(p)
    for _ in range(base.BUDGET):
        mi = binary_entropy(belief @ p) - belief @ conditional
        score = mi * available / base.COST[None, :]
        tool = np.argmax(np.where(used, -np.inf, score), axis=1)
        signal = ev[rows, tool]
        missing = active & (signal < 0)
        if aware:
            siblings = DOMAIN[None, :] == DOMAIN[tool, None]
            available = np.where(missing[:, None] & siblings, .05, available)
        used[rows, tool] = active
        calls += active
        cost_total += np.where(active, base.COST[tool], 0)
        observed += active & (signal >= 0)
        likelihood = np.where(signal[:, None] == 1, p[:, tool].T, 1 - p[:, tool].T)
        updated = belief * likelihood
        updated /= updated.sum(axis=1, keepdims=True)
        belief = np.where((active & (signal >= 0))[:, None], updated, belief)
        active &= ~((observed >= 2) & (np.max(belief, axis=1) >= gate))
    confirm = (observed >= 2) & (np.max(belief, axis=1) >= gate)
    correct = confirm & (np.argmax(belief, axis=1) == labels)
    stats = {'correct': int(correct.sum()),
             'false_confirmed': int((confirm & ~correct).sum()),
             'escalated': int((~confirm).sum()),
             'mean_calls': float(calls.mean()),
             'mean_cost': float(cost_total.mean())}
    return stats, correct


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, default=Path('fault_results/innovation_sweep_cpu.json'))
    ap.add_argument('--replicates', type=int, default=12)
    args = ap.parse_args()
    data = base.metadata(Path('formatted_trajectories.json.gz'), Path('fault_results/metadata.npz'))
    train_y, train_s, _ = base.episodes(data, 2020, 100)
    _, val_s, _ = base.episodes(data, 2021, 200)
    test_y, test_s, test_groups = base.episodes(data, 2022, 300)
    counts = np.stack([train_s[train_y == c].sum(axis=0) for c in range(6)])
    totals = np.array([(train_y == c).sum() for c in range(6)])[:, None]
    p = ((counts + 1) / (totals + 2)).astype(np.float32)
    same_pairs = [(i, j) for i in range(8) for j in range(i+1, 8) if DOMAIN[i] == DOMAIN[j]]
    rows = []
    replicate_summary = []
    for q in Q_VALUES:
        val_ev = evidence(val_s, q, 20261122)
        test_ev = evidence(test_s, q, 20261123)
        missing = (val_ev < 0).astype(float)
        corr = np.corrcoef(missing, rowvar=False)
        mean_corr = float(np.mean([corr[i, j] for i, j in same_pairs]))
        available = (val_ev >= 0).mean(axis=0).astype(np.float32)
        static, static_flags = evaluate(test_y, test_ev, p, available, False)
        aware, aware_flags = evaluate(test_y, test_ev, p, available, True)
        guard_choice = 'domain_aware' if mean_corr > .2 else 'static_cost_aware'
        paired = base.cluster_interval(aware_flags.astype(float), static_flags.astype(float), test_groups // 6)
        rows.append({'q': q, 'validation_mean_same_domain_missing_correlation': mean_corr,
                     'guard_choice': guard_choice, 'static': static, 'domain_aware': aware,
                     'paired_correct_difference': paired})
        repeats = []
        for rep in range(1, args.replicates + 1):
            val_rep = evidence(val_s, q, 20261122 + 10007 * rep)
            test_rep = evidence(test_s, q, 20261123 + 10007 * rep)
            corr_rep = np.corrcoef((val_rep < 0).astype(float), rowvar=False)
            within = float(np.mean([corr_rep[i, j] for i, j in same_pairs]))
            avail_rep = (val_rep >= 0).mean(axis=0).astype(np.float32)
            static_rep, _ = evaluate(test_y, test_rep, p, avail_rep, False)
            aware_rep, _ = evaluate(test_y, test_rep, p, avail_rep, True)
            repeats.append({'seed_index': rep, 'correct_difference':
                            (aware_rep['correct'] - static_rep['correct']) / len(test_y),
                            'guard_choice': 'domain_aware' if within > .2 else 'static_cost_aware'})
        delta = np.array([r['correct_difference'] for r in repeats])
        replicate_summary.append({'q': q, 'n_seeds': args.replicates,
                                  'mean_difference': float(delta.mean()),
                                  'min_difference': float(delta.min()),
                                  'max_difference': float(delta.max()),
                                  'positive_seeds': int((delta > 0).sum()),
                                  'domain_guard_seeds': sum(r['guard_choice'] == 'domain_aware' for r in repeats),
                                  'repeats': repeats})
    result = {'device': 'CPU', 'train_year': 2020, 'validation_year': 2021,
              'test_year': 2022, 'cases': len(test_y), 'independent_probe_failure': .05,
              'domain_probabilities': '[q, 0.32*q, q]', 'gate': .8,
              'correlation_gate': .2, 'rows': rows,
              'replicate_summary': replicate_summary}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    for row in rows:
        print(f"q={row['q']:.2f} corr={row['validation_mean_same_domain_missing_correlation']:.3f} "
              f"guard={row['guard_choice']} correct={row['static']['correct']}/{row['domain_aware']['correct']} "
              f"delta={100*row['paired_correct_difference']['difference']:+.2f}pp")
    for row in replicate_summary:
        print(f"repeat q={row['q']:.2f} mean={100*row['mean_difference']:+.2f}pp "
              f"range=[{100*row['min_difference']:+.2f},{100*row['max_difference']:+.2f}]pp "
              f"positive={row['positive_seeds']}/{row['n_seeds']} "
              f"guard_domain={row['domain_guard_seeds']}/{row['n_seeds']}")


if __name__ == '__main__':
    main()
