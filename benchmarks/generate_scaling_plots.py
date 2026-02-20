#!/usr/bin/env python3
"""
Generate 9 scaling plots for the paper:
- 3 operations (vec_add, vec_dot, matmul)
- 3 metrics (Qubits, Depth, T-count)
- 2 curves per plot (QFT vs Ripple)
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Setup
CSV_PATH = Path(__file__).parent / "scaling_results.csv"
OUTPUT_DIR = Path(__file__).parent.parent / "docs" / "latex" / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load data
df = pd.read_csv(CSV_PATH)

# Plot settings
plt.rcParams.update({
    'font.size': 8,
    'axes.labelsize': 9,
    'axes.titlesize': 10,
    'legend.fontsize': 7,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'figure.dpi': 150,
})

operations = ['vec_add', 'vec_dot', 'matmul']
op_labels = {'vec_add': 'Vector Addition', 'vec_dot': 'Dot Product', 'matmul': 'Matrix Multiplication'}
metrics = ['qubits', 'depth', 't_count']
metric_labels = {'qubits': 'Qubits', 'depth': 'Circuit Depth', 't_count': 'T-count'}

# Colors
colors = {'qft': '#1f77b4', 'ripple': '#ff7f0e'}
markers = {'qft': 'o', 'ripple': 's'}

# Generate individual plots
for op in operations:
    op_data = df[df['operation'] == op]

    for metric in metrics:
        fig, ax = plt.subplots(figsize=(2.8, 2.2))

        for backend in ['qft', 'ripple']:
            data = op_data[op_data['backend'] == backend]
            ax.plot(data['N'], data[metric],
                   marker=markers[backend],
                   color=colors[backend],
                   label=backend.upper() if backend == 'qft' else 'Ripple',
                   linewidth=1.2,
                   markersize=4)

        ax.set_xlabel('N')
        ax.set_ylabel(metric_labels[metric])
        ax.set_title(op_labels[op])
        ax.legend(loc='upper left')
        ax.set_xscale('log', base=2)
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3, linestyle='--')

        plt.tight_layout()
        filename = f"scaling_{op}_{metric}.pdf"
        fig.savefig(OUTPUT_DIR / filename, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved: {filename}")

# Also create a combined 3x3 figure
fig, axes = plt.subplots(3, 3, figsize=(7, 6))

for i, op in enumerate(operations):
    op_data = df[df['operation'] == op]

    for j, metric in enumerate(metrics):
        ax = axes[i, j]

        for backend in ['qft', 'ripple']:
            data = op_data[op_data['backend'] == backend]
            ax.plot(data['N'], data[metric],
                   marker=markers[backend],
                   color=colors[backend],
                   label=backend.upper() if backend == 'qft' else 'Ripple',
                   linewidth=1.0,
                   markersize=3)

        ax.set_xscale('log', base=2)
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3, linestyle='--')

        # Labels
        if i == 2:  # bottom row
            ax.set_xlabel('N')
        if j == 0:  # left column
            ax.set_ylabel(metric_labels[metric])
        if i == 0:  # top row
            ax.set_title(metric_labels[metric])
        if j == 2:  # right column
            ax.annotate(op_labels[op], xy=(1.05, 0.5), xycoords='axes fraction',
                       rotation=-90, va='center', fontsize=8)

        # Legend only on first plot
        if i == 0 and j == 0:
            ax.legend(loc='upper left', fontsize=6)

plt.tight_layout()
fig.savefig(OUTPUT_DIR / "scaling_all.pdf", bbox_inches='tight')
plt.close(fig)
print("Saved: scaling_all.pdf")

print(f"\nAll plots saved to: {OUTPUT_DIR}")
