"""复核现有 real_ais_scores.csv；只用 Python 标准库，不重新训练模型。"""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def auc(labels, scores):
    positives = sum(labels)
    negatives = len(labels) - positives
    if not positives or not negatives:
        return None
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    rank_sum = 0.0
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and scores[order[j]] == scores[order[i]]:
            j += 1
        mean_rank = (i + 1 + j) / 2
        rank_sum += mean_rank * sum(labels[order[k]] for k in range(i, j))
        i = j
    return (rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


def average_precision(labels, scores):
    positives = sum(labels)
    if not positives:
        return None
    ranked = sorted(range(len(scores)), key=lambda i: (-scores[i], i))
    found = 0
    total = 0.0
    for rank, index in enumerate(ranked, 1):
        if labels[index]:
            found += 1
            total += found / rank
    return total / positives


def review(rows, name, threshold):
    labels = [int(r['label']) for r in rows]
    scores = [float(r[name]) for r in rows]
    predicted = [s >= threshold for s in scores]
    tp = sum(y and p for y, p in zip(labels, predicted))
    fp = sum(not y and p for y, p in zip(labels, predicted))
    fn = sum(y and not p for y, p in zip(labels, predicted))
    tn = sum(not y and not p for y, p in zip(labels, predicted))
    by_vessel = defaultdict(list)
    for row in rows:
        by_vessel[row['mmsi']].append(row)
    vessel_labels = [int(any(int(r['label']) for r in group)) for group in by_vessel.values()]
    vessel_scores = [max(float(r[name]) for r in group) for group in by_vessel.values()]
    ranking = sorted(range(len(scores)), key=lambda i: (-scores[i], i))
    return {
        'trajectory_count': len(rows), 'positive_trajectories': sum(labels),
        'prevalence': sum(labels) / len(labels), 'roc_auc': auc(labels, scores),
        'average_precision': average_precision(labels, scores),
        'threshold': threshold, 'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
        'precision_at_threshold': tp / (tp + fp) if tp + fp else None,
        'recall_at_threshold': tp / (tp + fn) if tp + fn else None,
        'positive_in_top_100': sum(labels[i] for i in ranking[:100]),
        'vessel_count': len(by_vessel), 'positive_vessels': sum(vessel_labels),
        'vessel_roc_auc': auc(vessel_labels, vessel_scores),
        'vessel_average_precision': average_precision(vessel_labels, vessel_scores),
        'vessel_score_rule': 'maximum trajectory score per MMSI; exploratory only',
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scores', type=Path, required=True)
    parser.add_argument('--summary', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    with args.scores.open(newline='', encoding='utf-8-sig') as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError('scores CSV is empty')
    summary = json.loads(args.summary.read_text(encoding='utf-8'))
    result = {
        name: review(rows, name, values['threshold'])
        for name, values in summary['methods'].items()
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
