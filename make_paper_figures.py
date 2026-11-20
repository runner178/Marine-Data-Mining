"""Generate publication-style, data-linked figures from experiment JSON files."""
import argparse
import json
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from fontTools.ttLib import TTCollection


COLORS = {'optimized_fixed': '#4D5666', 'online_mi': '#0072B2',
          'cost_aware': '#D55E00', 'static_cost_aware': '#D55E00',
          'domain_aware': '#009E73'}
LABELS = {'optimized_fixed': '优化固定流程', 'online_mi': '在线互信息',
          'cost_aware': '成本感知在线'}


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def style():
    font_path = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
    # The TTC defaults to its Japanese face. Select face 2 (Simplified Chinese)
    # explicitly so characters such as 门 use the intended glyph shape.
    sc_font = Path(tempfile.gettempdir()) / 'NotoSansCJKsc-Regular-codex.otf'
    if not sc_font.exists():
        TTCollection(font_path).fonts[2].save(sc_font)
    font_manager.fontManager.addfont(str(sc_font))
    cjk_family = font_manager.FontProperties(fname=str(sc_font)).get_name()
    plt.rcParams.update({
        'font.family': cjk_family,
        'font.size': 9,
        'axes.titlesize': 10,
        'axes.labelsize': 9,
        'axes.edgecolor': '#4D5666',
        'axes.linewidth': .8,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'xtick.color': '#4D5666',
        'ytick.color': '#4D5666',
        'xtick.labelsize': 8.5,
        'ytick.labelsize': 8.5,
        'text.color': '#17212B',
        'axes.labelcolor': '#17212B',
        'grid.color': '#DDE3E8',
        'grid.linewidth': .7,
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'savefig.facecolor': 'white',
        'svg.fonttype': 'path',
    })


def save(fig, out, stem):
    fig.savefig(out / f'{stem}.svg', bbox_inches='tight', pad_inches=.16)
    fig.savefig(out / f'{stem}.png', dpi=320, bbox_inches='tight', pad_inches=.16)
    plt.close(fig)


def fig1(root, out):
    data = {gate: load(root / f'result_gate_{gate}.json') for gate in ('0.8', '0.95')}
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.5), sharex=True, sharey=True, constrained_layout=True)
    for panel, (ax, setting, title) in enumerate(zip(axes, ('normal', 'degraded'), ('常规可用率', '缺失加重'))):
        for name in LABELS:
            points = []
            for gate, marker, filled in [('0.8', 'o', True), ('0.95', 's', False)]:
                p = data[gate]['settings'][setting]['policies'][name]
                x = 100 * p['false_confirmed'] / data[gate]['test_2022_cases']
                y = 100 * p['correct'] / data[gate]['test_2022_cases']
                points.append((x, y))
                ax.scatter(x, y, s=48, marker=marker, linewidth=1.3,
                           facecolor=COLORS[name] if filled else 'white',
                           edgecolor=COLORS[name], zorder=4)
            ax.plot([v[0] for v in points], [v[1] for v in points],
                    color=COLORS[name], lw=1, alpha=.65, zorder=2)
        ax.set_title(f'{chr(65+panel)}  {title}', loc='left', fontweight='semibold')
        ax.set_xlim(-.35, 8.2); ax.set_ylim(28, 91)
        ax.set_xticks([0, 2, 4, 6, 8]); ax.set_yticks([30, 45, 60, 75, 90])
        ax.grid(axis='both', alpha=.8)
        ax.set_xlabel('错误确诊率（%）')
    axes[0].set_ylabel('正确确诊率（%）')
    from matplotlib.lines import Line2D
    legend = [Line2D([0], [0], color=COLORS[k], marker='o', lw=1.5, label=v) for k, v in LABELS.items()]
    fig.legend(handles=legend, ncol=3, frameon=False, loc='lower center', bbox_to_anchor=(.5, -.08), fontsize=8)
    save(fig, out, 'fig1_risk_tradeoff')


def fig2(root, out):
    gates = ['0.8', '0.9', '0.95']
    data = {g: load(root / f'result_gate_{g}.json') for g in gates}
    fig, axes = plt.subplots(2, 2, figsize=(7.4, 5.3), sharex=True, constrained_layout=True)
    for col, (setting, title) in enumerate([('normal', '常规探针可用率'), ('degraded', '探针缺失加重')]):
        for name in LABELS:
            correct = [100 * data[g]['settings'][setting]['policies'][name]['correct'] / 1908 for g in gates]
            false = [100 * data[g]['settings'][setting]['policies'][name]['false_confirmed'] / 1908 for g in gates]
            axes[0, col].plot([.8, .9, .95], correct, color=COLORS[name], marker='o', lw=2, ms=5, label=LABELS[name])
            axes[1, col].plot([.8, .9, .95], false, color=COLORS[name], marker='o', lw=2, ms=5)
        axes[0, col].set_title(f'{chr(65+col)}  {title}', loc='left', fontweight='semibold')
        for row in range(2):
            axes[row, col].set_xticks([.8, .9, .95], ['0.80', '0.90', '0.95'])
            axes[row, col].grid(axis='y')
            axes[row, col].set_xlim(.785, .965)
    axes[0, 0].set_ylabel('正确确诊（%）'); axes[1, 0].set_ylabel('错误确诊（%）')
    axes[1, 0].set_xlabel('确诊后验门槛'); axes[1, 1].set_xlabel('确诊后验门槛')
    axes[0, 0].set_ylim(28, 92); axes[0, 1].set_ylim(28, 92)
    axes[1, 0].set_ylim(-.2, 8); axes[1, 1].set_ylim(-.2, 8)
    fig.legend(*axes[0, 0].get_legend_handles_labels(), ncol=3, frameon=False,
               loc='lower center', bbox_to_anchor=(.5, -.05), fontsize=8)
    save(fig, out, 'fig2_gate_sensitivity')


def fig3(root, out):
    correlated = load(root / 'failure_domain_0.8.json')
    independent = load(root / 'failure_domain_control.json')
    rows = [
        ('同域相关失效', correlated),
        ('独立失效对照', independent),
    ]
    fig, ax = plt.subplots(figsize=(6.8, 2.5), constrained_layout=True)
    for ypos, (label, record) in zip([1, 0], rows):
        z = record['domain_aware_vs_static_correct']
        diff = 100 * z['difference']; lo, hi = [100 * t for t in z['cluster_bootstrap_95pct']]
        color = '#009E73' if ypos else '#D55E00'
        ax.plot([lo, hi], [ypos, ypos], color=color, lw=2.5, solid_capstyle='round')
        ax.scatter([diff], [ypos], s=62, color=color, zorder=3, edgecolors='white', linewidth=1)
        label_x, ha = (hi + .28, 'left') if diff > 0 else (lo - .2, 'right')
        ax.text(label_x, ypos, f'{diff:+.2f}', va='center', ha=ha,
                color=color, fontsize=8.5, fontweight='semibold')
    ax.axvline(0, color='#4D5666', lw=1, ls='--')
    ax.set_yticks([1, 0], ['同域相关失效', '独立失效对照'])
    ax.set_xlim(-3, 7.3); ax.set_ylim(-.5, 1.5)
    ax.set_xlabel('域感知 − 静态成本：正确确诊率差（百分点）')
    ax.grid(axis='x')
    save(fig, out, 'fig4_failure_domain')


def fig4(root, out):
    gates = ['0.8', '0.9', '0.95']
    data = [load(root / f'result_gate_{g}.json') for g in gates]
    decision = [z['compiled_online_nodes'] for z in data]
    terminal = [z['compiled_terminal_nodes'] for z in data]
    x = np.arange(len(gates)); width=.34
    fig, ax = plt.subplots(figsize=(6.8, 3.1), constrained_layout=True)
    a = ax.bar(x-width/2, decision, width, color='#0072B2', label='决策节点')
    b = ax.bar(x+width/2, terminal, width, color='#8FB9D5', label='终止节点')
    ax.bar_label(a, padding=2, fontsize=8); ax.bar_label(b, padding=2, fontsize=8)
    ax.set_xticks(x, ['0.80', '0.90', '0.95']); ax.set_xlabel('确诊后验门槛')
    ax.set_ylabel('节点数'); ax.set_ylim(0, 76); ax.grid(axis='y'); ax.set_axisbelow(True)
    ax.legend(frameon=False, ncol=2, loc='upper left')
    save(fig, out, 'fig3_compilation')


def fig5(root, out):
    data = load(root / 'innovation_sweep_cpu.json')['rows']
    x = np.array([row['q'] for row in data])
    static = np.array([100 * row['static']['correct'] / 1908 for row in data])
    aware = np.array([100 * row['domain_aware']['correct'] / 1908 for row in data])
    diff = np.array([100 * row['paired_correct_difference']['difference'] for row in data])
    ci = np.array([row['paired_correct_difference']['cluster_bootstrap_95pct'] for row in data]) * 100
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.4), constrained_layout=True)
    ax = axes[0]
    ax.plot(x, static, '-o', color='#D55E00', lw=2, label='静态成本策略')
    ax.plot(x, aware, '-o', color='#009E73', lw=2, label='域感知策略')
    ax.set_ylabel('正确确诊 / 全部案例（%）')
    ax.set_title('A  正确确诊率', loc='left', fontweight='semibold')
    ax.legend(frameon=False, fontsize=8)
    ax.grid(axis='y'); ax.set_ylim(45, 90)
    ax = axes[1]
    repeat = load(root / 'innovation_sweep_cpu.json').get('replicate_summary')
    if repeat:
        mean = np.array([100 * row['mean_difference'] for row in repeat])
        lo = np.array([100 * row['min_difference'] for row in repeat])
        hi = np.array([100 * row['max_difference'] for row in repeat])
        ax.fill_between(x, lo, hi, color='#009E73', alpha=.16, lw=0, label='12 种子范围')
        for q, row in zip(x, repeat):
            ax.scatter(np.full(row['n_seeds'], q),
                       [100 * z['correct_difference'] for z in row['repeats']],
                       s=9, color='#009E73', alpha=.25, zorder=2)
        ax.plot(x, mean, '-o', color='#009E73', lw=1.8, ms=4, label='12 种子均值')
    else:
        ax.fill_between(x, ci[:, 0], ci[:, 1], color='#009E73', alpha=.16, lw=0, label='95% 块重采样区间')
        ax.plot(x, diff, '-o', color='#009E73', lw=2, ms=5, label='域感知 − 静态')
    ax.axhline(0, color='#4D5666', ls='--', lw=1)
    ax.set_ylabel('正确确诊率差（百分点）')
    ax.set_title('B  配对收益与种子波动', loc='left', fontweight='semibold')
    ax.legend(frameon=False, fontsize=8, loc='upper left')
    ax.grid(axis='y'); ax.set_ylim(-2.5, 9)
    for ax in axes:
        ax.set_xticks(x, [f'{100 * q:.0f}' for q in x])
        ax.set_xlabel('共同域失效概率 q（%）')
    save(fig, out, 'fig5_domain_sweep')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results', type=Path, default=Path('fault_results'))
    ap.add_argument('--out', type=Path, default=Path('paper_figures'))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    style()
    fig1(args.results, args.out)
    fig2(args.results, args.out)
    fig3(args.results, args.out)
    fig4(args.results, args.out)
    if (args.results / 'innovation_sweep_cpu.json').exists():
        fig5(args.results, args.out)
    print('Generated figures in', args.out)


if __name__ == '__main__':
    main()
