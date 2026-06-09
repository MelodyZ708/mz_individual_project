"""
visualize_layer2_topN.py
=========================
Stage 3 Post-processing: Visualise Top-N channels from Layer2 basin analysis.

After running visualize_basin_layer2_128ch.py, this script:
  1. Reads channel_ranking.csv from the Layer2 output directory.
  2. Generates a summary bar chart of BQS for all 128 channels (colour-coded by rank).
  3. Generates a stacked component breakdown chart for Top-20 channels.
  4. Generates a cross-stage comparison table figure (Stage 1 / 2 / 3 Top-10 side-by-side).
  5. Prints a formatted ranking table to stdout.

Usage:
  python visualize_layer2_topN.py [--topn 10] [--outdir vis_results/convergence_basin_layer2]
"""

import matplotlib
matplotlib.use('Agg')

import argparse
import csv
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

plt.rcParams.update({
    'font.size':        10,
    'font.family':      'serif',
    'axes.titlesize':   12,
    'axes.labelsize':   10,
    'figure.dpi':       120,
    'savefig.dpi':      200,
    'mathtext.fontset': 'cm',
})

METRICS = ['LQBS', 'Width', 'ShapeSim', 'SharpRet', 'MinPos', 'Symmetry']
METRIC_COLORS = {
    'LQBS':     '#2196F3',
    'Width':    '#4CAF50',
    'ShapeSim': '#FF9800',
    'SharpRet': '#9C27B0',
    'MinPos':   '#F44336',
    'Symmetry': '#00BCD4',
}


def load_ranking(csv_path: str):
    rows = []
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                'rank':      int(row['rank']),
                'channel':   int(row['channel']),
                'BQS':       float(row['BQS']),
                'BQ':        float(row['BQ']),
                'LQBS':      float(row['LQBS']),
                'Width':     float(row['Width']),
                'ShapeSim':  float(row['ShapeSim']),
                'SharpRet':  float(row['SharpRet']),
                'MinPos':    float(row['MinPos']),
                'Symmetry':  float(row['Symmetry']),
                'Retention': float(row['Retention']),
                'kill_clean': row['kill_clean'],
                'kill_50':    row['kill_50'],
            })
    return rows


def plot_bqs_all_channels(rows, outdir, topn=10):
    """Bar chart of BQS for all 128 channels, sorted by channel index."""
    sorted_by_ch = sorted(rows, key=lambda r: r['channel'])
    channels = [r['channel'] for r in sorted_by_ch]
    bqs_vals = [r['BQS']     for r in sorted_by_ch]
    ranks    = [r['rank']    for r in sorted_by_ch]

    colors = []
    for rank in ranks:
        if rank <= topn:
            colors.append('#E53935')   # Top-N: red
        elif rank <= 20:
            colors.append('#FB8C00')   # Top-20: orange
        else:
            colors.append('#90A4AE')   # Rest: grey

    fig, ax = plt.subplots(figsize=(20, 5))
    bars = ax.bar(channels, bqs_vals, color=colors, width=0.75, edgecolor='none')

    # Annotate top-N bars with channel number
    for r in sorted_by_ch:
        if r['rank'] <= topn:
            ax.text(r['channel'], r['BQS'] + 0.005, f"ch{r['channel']:03d}",
                    ha='center', va='bottom', fontsize=7, color='#B71C1C', fontweight='bold')

    ax.set_xlabel('Channel Index', fontsize=11)
    ax.set_ylabel('BQS', fontsize=11)
    ax.set_title(f'Layer2 (Stage 3) — BQS for All 128 Channels\n'
                 f'(Red = Top-{topn}, Orange = Top-20, Grey = Rest)',
                 fontsize=12)
    ax.set_xlim(-1, 128)
    ax.set_ylim(0, max(bqs_vals) * 1.15)
    ax.axhline(y=np.mean(bqs_vals), color='steelblue', linestyle='--',
               linewidth=1.2, label=f'Mean BQS = {np.mean(bqs_vals):.4f}')
    ax.legend(fontsize=9)

    # Legend patches
    patches = [
        mpatches.Patch(color='#E53935', label=f'Top-{topn}'),
        mpatches.Patch(color='#FB8C00', label='Top-11~20'),
        mpatches.Patch(color='#90A4AE', label='Rest'),
    ]
    ax.legend(handles=patches + [
        plt.Line2D([0], [0], color='steelblue', linestyle='--',
                   label=f'Mean = {np.mean(bqs_vals):.4f}')
    ], fontsize=9, loc='upper right')

    plt.tight_layout()
    out = os.path.join(outdir, 'layer2_bqs_all_channels.png')
    plt.savefig(out, bbox_inches='tight')
    plt.close()
    print(f"  [Saved] {out}")


def plot_component_breakdown(rows, outdir, topn=20):
    """Stacked bar chart showing BQS component breakdown for Top-N channels."""
    top_rows = [r for r in rows if r['rank'] <= topn]
    top_rows.sort(key=lambda r: r['BQS'], reverse=True)

    labels = [f"ch{r['channel']:03d}" for r in top_rows]
    bq_vals = [r['BQ'] for r in top_rows]
    ret_vals = [r['Retention'] for r in top_rows]

    # Sub-components of Retention
    shape_vals = [r['ShapeSim']  for r in top_rows]
    sharp_vals = [r['SharpRet']  for r in top_rows]
    minpos_vals = [r['MinPos']   for r in top_rows]
    sym_vals   = [r['Symmetry']  for r in top_rows]

    x = np.arange(topn)
    width = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(18, 6))

    # Left: BQ vs BQS
    ax = axes[0]
    ax.bar(x, bq_vals,  width, label='BQ (0.75×LQBS + 0.25×Width)', color='#1565C0', alpha=0.85)
    ax.bar(x, [r['BQS'] for r in top_rows], width, label='BQS = BQ × Retention',
           color='#E53935', alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Score')
    ax.set_title(f'Top-{topn} Layer2 Channels: BQ vs BQS')
    ax.legend(fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.grid(axis='y', alpha=0.3)

    # Right: Retention sub-components (stacked)
    ax2 = axes[1]
    # Normalise sub-components so they stack to Retention
    # Retention = ShapeSim * SharpRet * MinPos * Symmetry
    # Show each as fraction of Retention
    bottom = np.zeros(topn)
    comp_data = {
        'ShapeSim':  shape_vals,
        'SharpRet':  sharp_vals,
        'MinPos':    minpos_vals,
        'Symmetry':  sym_vals,
    }
    for metric, vals in comp_data.items():
        ax2.bar(x, vals, width, bottom=bottom, label=metric,
                color=METRIC_COLORS[metric], alpha=0.85)
        bottom += np.array(vals)

    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    ax2.set_ylabel('Component Value (stacked sum)')
    ax2.set_title(f'Top-{topn} Layer2 Channels: Retention Sub-components')
    ax2.legend(fontsize=9, loc='lower right')
    ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    out = os.path.join(outdir, 'layer2_component_breakdown.png')
    plt.savefig(out, bbox_inches='tight')
    plt.close()
    print(f"  [Saved] {out}")


def plot_kill_scatter(rows, outdir, topn=20):
    """Scatter plot: BQS vs Kill%@+50% for all channels, highlighting Top-N."""
    all_bqs   = [r['BQS']  for r in rows]
    all_kill  = [float(r['kill_50']) for r in rows]
    all_ranks = [r['rank'] for r in rows]

    colors = ['#E53935' if rk <= topn else '#90A4AE' for rk in all_ranks]
    sizes  = [80 if rk <= topn else 20 for rk in all_ranks]

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(all_kill, all_bqs, c=colors, s=sizes, alpha=0.75, edgecolors='none')

    # Label Top-N
    for r in rows:
        if r['rank'] <= topn:
            ax.annotate(f"ch{r['channel']:03d}",
                        (float(r['kill_50']), r['BQS']),
                        textcoords='offset points', xytext=(4, 3),
                        fontsize=7, color='#B71C1C')

    ax.axvline(x=10, color='orange', linestyle='--', linewidth=1,
               label='Kill%=10% threshold')
    ax.set_xlabel('Kill% at +50% Brightness', fontsize=11)
    ax.set_ylabel('BQS', fontsize=11)
    ax.set_title('Layer2 (Stage 3) — BQS vs Kill%@+50%\n'
                 '(Red = Top-20, high Kill% may indicate false robustness)',
                 fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    out = os.path.join(outdir, 'layer2_bqs_vs_kill.png')
    plt.savefig(out, bbox_inches='tight')
    plt.close()
    print(f"  [Saved] {out}")


def print_ranking_table(rows, topn=10):
    """Print formatted Top-N ranking table to stdout."""
    top = [r for r in rows if r['rank'] <= topn]
    top.sort(key=lambda r: r['rank'])

    header = (f"{'Rank':>4}  {'Ch':>4}  {'BQS':>7}  {'BQ':>7}  {'LQBS':>7}  "
              f"{'Width':>7}  {'ShapeSim':>9}  {'SharpRet':>9}  "
              f"{'MinPos':>7}  {'Symmetry':>9}  {'Retention':>10}  "
              f"{'Kill%C':>7}  {'Kill%50':>7}")
    print("=" * len(header))
    print(f"  Layer2 (Stage 3) — Top-{topn} Channel Ranking")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for r in top:
        print(f"  #{r['rank']:<3}  {r['channel']:>4}  {r['BQS']:>7.4f}  {r['BQ']:>7.4f}  "
              f"{r['LQBS']:>7.4f}  {r['Width']:>7.4f}  "
              f"{r['ShapeSim']:>9.4f}  {r['SharpRet']:>9.4f}  "
              f"{r['MinPos']:>7.4f}  {r['Symmetry']:>9.4f}  {r['Retention']:>10.4f}  "
              f"{r['kill_clean']:>7}%  {r['kill_50']:>7}%")
    print("=" * len(header))

    # Summary stats
    print(f"\n  Top-{topn} Mean Values:")
    for m in ['BQS', 'BQ', 'LQBS', 'Width', 'ShapeSim', 'SharpRet',
              'MinPos', 'Symmetry', 'Retention']:
        mean_val = np.mean([r[m] for r in top])
        print(f"    {m:<12} = {mean_val:.4f}")


def main():
    parser = argparse.ArgumentParser(description='Layer2 Basin Top-N Visualisation')
    parser.add_argument('--topn',   type=int, default=10,
                        help='Number of top channels to highlight (default: 10)')
    parser.add_argument('--outdir', type=str,
                        default='vis_results/convergence_basin_layer2',
                        help='Directory containing channel_ranking.csv and output PNGs')
    args = parser.parse_args()

    csv_path = os.path.join(args.outdir, 'channel_ranking.csv')
    if not os.path.exists(csv_path):
        print(f"[ERROR] channel_ranking.csv not found at: {csv_path}")
        print("  Please run visualize_basin_layer2_128ch.py first.")
        return

    print(f"\n  Loading rankings from: {csv_path}")
    rows = load_ranking(csv_path)
    print(f"  Loaded {len(rows)} channels.\n")

    print("  Generating plots...")
    plot_bqs_all_channels(rows, args.outdir, topn=args.topn)
    plot_component_breakdown(rows, args.outdir, topn=20)
    plot_kill_scatter(rows, args.outdir, topn=20)

    print()
    print_ranking_table(rows, topn=args.topn)

    print(f"\n  All visualisations saved to: {args.outdir}/")


if __name__ == "__main__":
    main()