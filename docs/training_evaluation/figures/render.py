"""Render the archived numeric snapshot, without reading original experiment artifacts."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

ROOT = Path(__file__).resolve().parent
DATA = json.loads((ROOT / 'metrics.json').read_text())
GROUPS = DATA['groups']
COLORS = ['#0072B2', '#D55E00', '#009E73', '#CC79A7', '#595959']
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                     'axes.titlesize': 12, 'axes.labelsize': 10,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'savefig.facecolor': 'white', 'axes.axisbelow': True})


def save(fig, name):
    fig.savefig(ROOT / name, dpi=180, bbox_inches='tight')
    plt.close(fig)


def points(group, dataset):
    pairs = sorted((int(s), r[dataset]['strict']['f1'] * 100)
                   for s, r in GROUPS[group].items() if dataset in r)
    return [s / 1000 for s, _ in pairs], [v for _, v in pairs]


def trend(ax, entries, dataset, title, limits=(55, 92)):
    for i, (group, label) in enumerate(entries):
        x, y = points(group, dataset)
        ax.plot(x, y, marker=['o', 's', '^', 'D', 'v'][i], markersize=4,
                linewidth=1.7, color=COLORS[i], label=label)
    ax.set(title=title, xlabel='Training step (thousands)', ylabel='Detection F1 (%)',
           ylim=limits)
    ax.grid(axis='y', color='#e4e4e4')
    ax.xaxis.set_major_locator(MultipleLocator(2))
    ax.legend(fontsize=8, loc='lower right', frameon=True)


def main():
    historical = ['v5.8 4B', 'v5.8 27B', 'v5.9 0.8B', 'v5.9 2B', 'v5.9 4B',
                  'v5.10 0.8B', 'v5.10 4B', 'v5.10 4B rebalance',
                  'v5.10 4B rebalance2', 'v5.10 4B rebalance3']
    fig, ax = plt.subplots(figsize=(10.8, 6.0), layout='constrained')
    labels, values = [], []
    for group in historical:
        pairs = [(int(s), (r['real_v1']['strict']['f1'] + r['real_v2']['strict']['f1']) * 50)
                 for s, r in GROUPS[group].items() if 'real_v1' in r and 'real_v2' in r]
        step, val = max(pairs, key=lambda x: x[1])
        suffix = ' *' if 'v5.8' in group else (' [partial]' if 'rebalance3' in group else '')
        labels.append(f'{group} / {step // 1000}k{suffix}')
        values.append(val)
    bars = ax.barh(labels, values, color=['#999999'] * 2 + [COLORS[0]] * 3 + [COLORS[2]] * 5)
    for bar in list(bars)[:2]:
        bar.set_hatch('//')
    bars[-1].set_hatch('..')
    ax.bar_label(bars, fmt='%.2f', padding=4, fontsize=10)
    ax.invert_yaxis()
    ax.set(xlim=(0, 100), xlabel='Equal mean of real_v1 / real_v2 F1 (%)',
           title='Detection: best measured checkpoint per series')
    ax.grid(axis='x', alpha=.2)
    fig.supxlabel('* v5.8: incomplete inference contract.  Rebalance3: measured 3k-12k only.', fontsize=9)
    save(fig, 'detection_best.png')

    fig, axs = plt.subplots(2, 2, figsize=(12, 8), layout='constrained')
    small = [('v5.9 0.8B', 'v5.9'), ('v5.10 0.8B', 'v5.10'),
             ('v5.10 0.8B LR首轮 high', 'v5.10 LR 6e-5 (round 1)')]
    large = [('v5.9 4B', 'v5.9'), ('v5.10 4B', 'v5.10'),
             ('v5.10 4B rebalance', 'rebalance'), ('v5.10 4B rebalance2', 'rebalance2'),
             ('v5.10 4B rebalance3', 'rebalance3 (partial)')]
    for j, ds in enumerate(['real_v1', 'real_v2']):
        trend(axs[0, j], small, ds, '0.8B / ' + ds, (30, 92))
        trend(axs[1, j], large, ds, '4B / ' + ds, (30, 92))
    fig.suptitle('Detection checkpoint trajectories | measured points, no smoothing', fontsize=14)
    fig.supxlabel('Different series have different data / budgets. Missing checkpoints are not zero.', fontsize=9)
    save(fig, 'checkpoint_trends.png')

    fig, axs = plt.subplots(2, 2, figsize=(12, 8), layout='constrained')
    first = [('v5.10 0.8B LR首轮 ' + g, label) for g, label in
             [('low', '1.5e-5'), ('baseline', '3e-5'), ('high', '6e-5')]]
    upper = [('v5.10 0.8B LR上探 lr' + x, x) for x in ['6e-5', '1e-4', '1.5e-4', '2e-4']]
    for j, ds in enumerate(['real_v1', 'real_v2']):
        trend(axs[0, j], first, ds, 'Round 1: completed / ' + ds, (0, 100))
        trend(axs[1, j], upper, ds, 'Upper LR: through 5k / ' + ds, (0, 100))
        axs[1, j].xaxis.set_major_locator(MultipleLocator(1))
    fig.suptitle('0.8B learning-rate ablation | same checkpoint comparisons', fontsize=14)
    fig.supxlabel('Upper 2e-4 at 1k: no valid predictions (length failures); not a missing measurement.', fontsize=9)
    save(fig, 'lr_ablation.png')

    fig, axs = plt.subplots(1, 2, figsize=(12, 5.7), layout='constrained',
                            gridspec_kw={'width_ratios': [1.2, 1]})
    labels, scores = [], []
    for r in DATA['gt']:
        n = r['model']
        if n.startswith('attr-'):
            _, ver, size, step = n.split('-')
            label = f'v5.10 {"0.8B" if size == "08b" else "4B"} / {int(step)//1000}k'
        else:
            import re
            m = re.search(r'-(0\.8b|2b|4b|27b)-v5-9-ckpt(\d+)', n)
            label = f'v5.9 {m[1].upper()} / {int(m[2])//1000}k'
        labels.append(label)
        scores.append(r['scores']['overall'] * 100)
    bars = axs[0].barh(labels, scores, color=[COLORS[0]] * 4 + [COLORS[2]] * 6)
    axs[0].bar_label(bars, fmt='%.2f', padding=3, fontsize=9)
    axs[0].invert_yaxis()
    axs[0].set(xlim=(0, 100), xlabel='GT-BBox weighted E2E (%)', title='Schema-aligned historical comparison')
    axs[0].grid(axis='x', alpha=.2)
    for i, size in enumerate(['08b', '4b']):
        rs = [r for r in DATA['gt'] if r['model'].startswith('attr-v510-' + size)]
        x = [int(r['model'].split('-')[-1]) / 1000 for r in rs]
        y = [r['scores']['overall'] * 100 for r in rs]
        axs[1].plot(x, y, color=COLORS[i], marker='o', label='0.8B' if size == '08b' else '4B')
        for xx, yy in zip(x, y):
            axs[1].annotate(f'{yy:.3f}', (xx, yy), xytext=(0, 8 if i == 0 else -15),
                            textcoords='offset points', ha='center', fontsize=9)
    axs[1].set(xlim=(8, 26), ylim=(77.7, 78.85), xlabel='Training step (thousands)',
               ylabel='GT-BBox weighted E2E (%)', title='v5.10 late checkpoints (zoomed axis)')
    axs[1].grid(alpha=.2)
    axs[1].legend(loc='lower left')
    fig.suptitle('GT-BBox attributes | corrected scores, NOT predicted-box end-to-end', fontsize=14)
    save(fig, 'gtbox_attributes.png')
    print('Rendered four figures from the archived snapshot:', DATA['snapshot'])


if __name__ == '__main__':
    main()
