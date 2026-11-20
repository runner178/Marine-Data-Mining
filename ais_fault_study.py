"""AIS trajectory-ingest fault injection and sequential diagnostic policies.

This is a SEMI-SYNTHETIC experiment. Real released AIS trajectories supply
timestamps, vessel types, and positions. Fault labels and probe availability
are injected and must not be confused with real server incident labels.
"""
import argparse
import itertools
import json
import os
from pathlib import Path

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '7')
import cupy as cp
import numpy as np

from real_ais_benchmark import records

CAUSES = ['receiver_loss', 'west_receiver_loss', 'type_filter',
          'duplicate_replay', 'clock_shift', 'arrival_backlog']
PROBES = ['count_drop', 'count_rise', 'duplicate_id', 'type_mix_shift',
          'spatial_mix_shift', 'event_clock_shift', 'arrival_lag', 'west_share_shift']
COST = np.asarray([1, 1, 1.5, 2, 2, 1.5, 2, 1.5], np.float32)
MISSING = {'normal': np.full(8, .05, np.float32),
           'degraded': np.asarray([.25, .15, .4, .3, .4, .2, .45, .3], np.float32)}
EPISODE_SIZE = 32
BUDGET = 4
GATE = .8


def metadata(raw, cache):
    if cache.exists():
        z = np.load(cache, allow_pickle=False)
        return {k: z[k] for k in z.files}
    out = {k: [] for k in ('year', 'timestamp', 'mmsi', 'type', 'lat', 'lon', 'speed', 'label')}
    for _, item in records(raw):
        ts = item.get('timestamp') or []
        if len(ts) < 2 or len(item['lat']) < 2:
            continue
        start = int(ts[0]); year = int(np.datetime64(start, 's').astype('datetime64[Y]').astype(int) + 1970)
        out['year'].append(year); out['timestamp'].append(start)
        out['mmsi'].append(str(item['mmsi'])); out['type'].append(str(item['ship_type']))
        out['lat'].append(float(np.mean(item['lat']))); out['lon'].append(float(np.mean(item['lon'])))
        out['speed'].append(float(np.median(item['speed']))); out['label'].append(int(item['anomaly_label']))
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, **{k: np.asarray(v) for k, v in out.items()})
    return metadata(raw, cache)


def distribution(values, categories):
    return np.asarray([np.mean(values == c) for c in categories])


def probe_episode(base, cause, rng):
    n = len(base['mmsi'])
    ids = np.arange(n)
    chosen = ids.copy()
    current_time = base['timestamp'].copy()
    arrival_delay = np.zeros(n, np.int32)
    if cause == 0:
        chosen = ids[rng.random(n) >= rng.uniform(.35, .55)]
    elif cause == 1:
        west = base['lon'] < np.median(base['lon'])
        chosen = ids[~(west & (rng.random(n) < rng.uniform(.7, .95)))]
    elif cause == 2:
        types, counts = np.unique(base['type'], return_counts=True)
        affected = types[np.argmax(counts)]
        chosen = ids[~((base['type'] == affected) & (rng.random(n) < rng.uniform(.7, .95)))]
    elif cause == 3:
        repeated = ids[rng.random(n) < rng.uniform(.3, .5)]
        chosen = np.r_[ids, repeated]
        current_time = base['timestamp'][chosen]
        arrival_delay = np.zeros(len(chosen), np.int32)
    elif cause == 4:
        current_time = base['timestamp'] + rng.choice([-1, 1]) * rng.integers(3600, 7201)
    elif cause == 5:
        arrival_delay = np.where(rng.random(n) < rng.uniform(.3, .6), rng.integers(3600, 10801), 0)
    if len(chosen) == 0:
        chosen = ids[:1]
    if cause not in (3, 4):
        current_time = base['timestamp'][chosen]
        arrival_delay = arrival_delay[chosen]
    # Reference counts come from upstream source-track metadata, as in a
    # paired ingest audit. They are not obtained from the injected label.
    source_type = base['type']; sink_type = base['type'][chosen]
    cats = np.unique(source_type)
    type_tv = .5 * np.sum(np.abs(distribution(source_type, cats) - distribution(sink_type, cats)))
    lat_m, lon_m = np.median(base['lat']), np.median(base['lon'])
    source_quad = (base['lat'] >= lat_m).astype(int) * 2 + (base['lon'] >= lon_m).astype(int)
    sink_quad = source_quad[chosen]
    geo_tv = .5 * np.sum(np.abs(distribution(source_quad, range(4)) - distribution(sink_quad, range(4))))
    west_source = np.mean(base['lon'] < lon_m)
    west_sink = np.mean(base['lon'][chosen] < lon_m)
    ratio = len(chosen) / n
    duplicate_fraction = 1 - len(np.unique(chosen)) / len(chosen)
    clock_diff = np.median(np.abs(current_time - base['timestamp'][chosen]))
    late_fraction = np.mean(arrival_delay > 1800)
    return np.asarray([ratio < .8, ratio > 1.2, duplicate_fraction > .08,
                       type_tv > .18, geo_tv > .18, clock_diff > 1800,
                       late_fraction > .2, abs(west_sink - west_source) > .15], np.int8)


def episodes(data, year, seed):
    indexes = np.flatnonzero((data['year'] == year) & (data['label'] == 0))
    indexes = indexes[np.argsort(data['timestamp'][indexes])]
    labels, signals, groups = [], [], []
    for group, start in enumerate(range(0, len(indexes) - EPISODE_SIZE + 1, EPISODE_SIZE)):
        chunk = indexes[start:start + EPISODE_SIZE]
        base = {k: data[k][chunk] for k in ('mmsi', 'type', 'lat', 'lon', 'timestamp')}
        for cause in range(len(CAUSES)):
            rng = np.random.default_rng(seed + group * 37 + cause * 7919)
            labels.append(cause); signals.append(probe_episode(base, cause, rng)); groups.append(group)
    return np.asarray(labels, np.int32), np.asarray(signals, np.int8), np.asarray(groups, np.int32)


def entropy(p):
    p = cp.clip(p, 1e-8, 1 - 1e-8)
    return -p * cp.log2(p) - (1 - p) * cp.log2(1 - p)


def run_policy(labels, evidence, p, policy, order, availability, gate):
    n = len(labels); rows = cp.arange(n)
    belief = cp.full((n, len(CAUSES)), 1 / len(CAUSES), cp.float32)
    used = cp.zeros((n, len(PROBES)), cp.bool_)
    active = cp.ones(n, cp.bool_)
    observed = cp.zeros(n, cp.int8)
    calls = cp.zeros(n, cp.int8)
    paths = cp.full((n, BUDGET), -1, cp.int8)
    cost_total = cp.zeros(n, cp.float32)
    cost = cp.asarray(COST)
    conditional_entropy = entropy(p)
    for step in range(BUDGET):
        if policy == 'fixed':
            tool = cp.full(n, int(order[step]), cp.int32)
        else:
            marginal = belief @ p
            mi = entropy(marginal) - belief @ conditional_entropy
            if policy == 'cost_aware':
                mi = mi * availability[None, :] / cost[None, :]
            tool = cp.argmax(cp.where(used, -cp.inf, mi), axis=1).astype(cp.int32)
        signal = evidence[rows, tool]
        paths[:, step] = cp.where(active, tool, -1)
        used[rows, tool] = active
        calls += active
        cost_total += cp.where(active, cost[tool], 0)
        observed += active & (signal >= 0)
        likelihood = cp.where(signal[:, None] == 1, p[:, tool].T, 1 - p[:, tool].T)
        updated = belief * likelihood
        updated /= cp.sum(updated, axis=1, keepdims=True)
        belief = cp.where((active & (signal >= 0))[:, None], updated, belief)
        active &= ~((observed >= 2) & (cp.max(belief, axis=1) >= gate))
    confident = (observed >= 2) & (cp.max(belief, axis=1) >= gate)
    correct = confident & (cp.argmax(belief, axis=1) == labels)
    false_confirmed = confident & ~correct
    return {'correct': int(cp.sum(correct).get()),
            'false_confirmed': int(cp.sum(confident & ~correct).get()),
            'escalated': int(cp.sum(~confident).get()),
            'mean_calls': float(cp.mean(calls).get()),
            'mean_cost': float(cp.mean(cost_total).get())}, cp.asnumpy(correct), cp.asnumpy(false_confirmed), cp.asnumpy(paths)


def cluster_interval(a, b, groups, seed=20261116):
    unique = np.unique(groups)
    per_group = np.asarray([np.mean(a[groups == group] - b[groups == group]) for group in unique])
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(unique), size=(4000, len(unique)))
    boot = np.mean(per_group[draws], axis=1)
    return {'difference': float(np.mean(per_group)),
            'cluster_bootstrap_95pct': np.quantile(boot, [.025, .975]).tolist(),
            'cluster_count': len(unique)}


def compile_online(p, gate):
    """Compile every binary/missing observation path for the frozen model."""
    table = {}
    terminals = 0
    initial = np.full(len(CAUSES), 1 / len(CAUSES), np.float64)
    def binary_entropy(v):
        v = np.clip(v, 1e-8, 1 - 1e-8)
        return -v * np.log2(v) - (1 - v) * np.log2(1 - v)
    conditional = binary_entropy(p)
    def visit(history, belief):
        nonlocal terminals
        if len(history) >= BUDGET or (sum(v >= 0 for _, v in history) >= 2 and np.max(belief) >= gate):
            terminals += 1
            return
        mi = binary_entropy(belief @ p) - belief @ conditional
        for tool, _ in history:
            mi[tool] = -np.inf
        tool = int(np.argmax(mi))
        table[history] = tool
        for signal in (-1, 0, 1):
            posterior = belief.copy()
            if signal >= 0:
                posterior *= p[:, tool] if signal == 1 else 1 - p[:, tool]
                posterior /= posterior.sum()
            visit(history + ((tool, signal),), posterior)
    visit((), initial)
    return table, terminals


def evaluate_compiled(table, p, evidence, labels, gate):
    correct = false = escalated = 0
    paths = np.full((len(evidence), BUDGET), -1, np.int8)
    for i, (row, label) in enumerate(zip(evidence, labels)):
        history = ()
        belief = np.full(len(CAUSES), 1 / len(CAUSES), np.float64)
        while history in table:
            tool = table[history]
            paths[i, len(history)] = tool
            signal = int(row[tool])
            history += ((tool, signal),)
            if signal >= 0:
                belief *= p[:, tool] if signal == 1 else 1 - p[:, tool]
                belief /= belief.sum()
        confident = sum(v >= 0 for _, v in history) >= 2 and np.max(belief) >= gate
        if not confident:
            escalated += 1
        elif int(np.argmax(belief)) == label:
            correct += 1
        else:
            false += 1
    return {'correct': correct, 'false_confirmed': false, 'escalated': escalated}, paths


def optimize_fixed_order(labels, evidence, p, gate):
    """Search all 8P4 fixed schedules on 2021 only; maximize correct yield."""
    all_orders = np.asarray(list(itertools.permutations(range(len(PROBES)), BUDGET)), np.int32)
    order = cp.asarray(all_orders)
    ev = cp.asarray(evidence)
    y = cp.asarray(labels)
    count = len(labels)
    n_orders = len(all_orders)
    rows = cp.arange(count)[None, :]
    belief = cp.full((n_orders, count, len(CAUSES)), 1 / len(CAUSES), cp.float32)
    active = cp.ones((n_orders, count), cp.bool_)
    observed = cp.zeros((n_orders, count), cp.int8)
    costs = cp.zeros((n_orders, count), cp.float32)
    cost_vector = cp.asarray(COST)
    for step in range(BUDGET):
        tool = order[:, step]
        signal = ev[rows, tool[:, None]]
        costs += cp.where(active, cost_vector[tool, None], 0)
        observed += active & (signal >= 0)
        positive = p[:, tool].T[:, None, :]
        likelihood = cp.where(signal[:, :, None] == 1, positive, 1 - positive)
        updated = belief * likelihood
        updated /= cp.sum(updated, axis=2, keepdims=True)
        belief = cp.where((active & (signal >= 0))[:, :, None], updated, belief)
        active &= ~((observed >= 2) & (cp.max(belief, axis=2) >= gate))
    confident = (observed >= 2) & (cp.max(belief, axis=2) >= gate)
    correct = cp.sum(confident & (cp.argmax(belief, axis=2) == y[None, :]), axis=1)
    false = cp.sum(confident, axis=1) - correct
    mean_cost = cp.mean(costs, axis=1)
    utility = correct.astype(cp.float64) * 1e8 - false.astype(cp.float64) * 1e4 - mean_cost
    chosen = int(cp.argmax(utility).get())
    return all_orders[chosen].tolist(), {'correct': int(correct[chosen].get()),
                                          'false_confirmed': int(false[chosen].get()),
                                          'mean_cost': float(mean_cost[chosen].get()),
                                          'orders_searched': n_orders}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--raw', type=Path, default=Path('formatted_trajectories.json.gz'))
    ap.add_argument('--cache', type=Path, default=Path('fault_results/metadata.npz'))
    ap.add_argument('--out', type=Path, default=Path('fault_results/result.json'))
    ap.add_argument('--gate', type=float, default=GATE)
    args = ap.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '7' or cp.cuda.runtime.getDeviceCount() != 1:
        raise RuntimeError('Use only physical GPU 7')
    data = metadata(args.raw, args.cache)
    train_y, train_s, train_groups = episodes(data, 2020, 100)
    val_y, val_s, val_groups = episodes(data, 2021, 200)
    test_y, test_s, test_groups = episodes(data, 2022, 300)
    # Bernoulli likelihood fitted only on injected 2020 cases.
    counts = np.stack([train_s[train_y == c].sum(axis=0) for c in range(len(CAUSES))])
    totals = np.asarray([np.sum(train_y == c) for c in range(len(CAUSES))])[:, None]
    likelihood = (counts + 1) / (totals + 2)
    p = cp.asarray(likelihood, cp.float32)
    prior = cp.full(len(CAUSES), 1 / len(CAUSES), cp.float32)
    mi0 = cp.asnumpy(entropy(prior @ p) - prior @ entropy(p))
    fixed_order = np.argsort(-mi0).tolist()[:BUDGET]
    compiled, terminal_count = compile_online(likelihood, args.gate)
    result = {'device': cp.cuda.runtime.getDeviceProperties(0)['name'].decode(),
              'visible_device': os.environ.get('CUDA_VISIBLE_DEVICES'),
              'study_type': 'semi-synthetic; injected trajectory-ingest faults, not observed real incidents',
              'episode_size': EPISODE_SIZE, 'causes': CAUSES, 'probes': PROBES,
              'confidence_gate': args.gate,
              'fit_2020_cases': len(train_y), 'validation_2021_cases': len(val_y),
              'test_2022_cases': len(test_y), 'test_base_episodes': int(test_groups.max() + 1),
              'learned_positive_likelihood': likelihood.tolist(), 'fixed_order': fixed_order,
              'compiled_online_nodes': len(compiled), 'compiled_terminal_nodes': terminal_count,
              'settings': {}}
    for setting, missing in MISSING.items():
        rng = np.random.default_rng(20261116 + (setting == 'degraded'))
        val_ev = np.where(rng.random(val_s.shape) < missing, -1, val_s)
        test_ev = np.where(rng.random(test_s.shape) < missing, -1, test_s)
        # 2021 calibration estimates availability; 2022 labels are not used.
        estimated = cp.asarray(np.mean(val_ev >= 0, axis=0).astype(np.float32))
        optimized_order, validation_fixed = optimize_fixed_order(val_y, val_ev, p, args.gate)
        evidence = cp.asarray(test_ev)
        labels = cp.asarray(test_y)
        policies = {}
        correctness = {}
        false_flags = {}
        paths = {}
        for name, method, available, order_used in [('greedy_fixed', 'fixed', estimated, fixed_order),
                                                    ('optimized_fixed', 'fixed', estimated, optimized_order),
                                                    ('online_mi', 'online', estimated, fixed_order),
                                                    ('cost_aware', 'cost_aware', estimated, fixed_order)]:
            policies[name], correctness[name], false_flags[name], paths[name] = run_policy(labels, evidence, p, method, order_used, available, args.gate)
            policies[name]['correct_by_cause'] = [int(np.sum(correctness[name][test_y == c])) for c in range(len(CAUSES))]
            policies[name]['false_confirmed_by_cause'] = [int(np.sum(false_flags[name][test_y == c])) for c in range(len(CAUSES))]
        compiled_scores, compiled_paths = evaluate_compiled(compiled, likelihood, test_ev, test_y, args.gate)
        exact_path_match = bool(np.array_equal(compiled_paths, paths['online_mi']))
        if not exact_path_match:
            raise AssertionError('Compiled and online paths differ')
        result['settings'][setting] = {'missing_rates': missing.tolist(),
                                       'estimated_availability': cp.asnumpy(estimated).tolist(),
                                       'optimized_fixed_order': optimized_order,
                                       'validation_fixed_selection': validation_fixed,
                                       'policies': policies,
                                       'compiled_online': compiled_scores,
                                       'compiled_online_exact_path_match': exact_path_match,
                                       'paired_online_vs_optimized_fixed': cluster_interval(correctness['online_mi'].astype(float), correctness['optimized_fixed'].astype(float), test_groups),
                                       'paired_online_vs_optimized_fixed_block6': cluster_interval(correctness['online_mi'].astype(float), correctness['optimized_fixed'].astype(float), test_groups // 6),
                                       'paired_cost_aware_vs_online': cluster_interval(correctness['cost_aware'].astype(float), correctness['online_mi'].astype(float), test_groups),
                                       'paired_cost_aware_vs_online_block6': cluster_interval(correctness['cost_aware'].astype(float), correctness['online_mi'].astype(float), test_groups // 6),
                                       'paired_cost_aware_vs_optimized_fixed': cluster_interval(correctness['cost_aware'].astype(float), correctness['optimized_fixed'].astype(float), test_groups),
                                       'paired_cost_aware_false_vs_online': cluster_interval(false_flags['cost_aware'].astype(float), false_flags['online_mi'].astype(float), test_groups)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
