"""Regenerate the manuscript's ChartX figures from the validation split.

Writes:
    chartx_by_type.png    grouped bars, 95% bootstrap CI error bars
    chartx_heatmap.png    numeric F1 per system and chart type

Every system, DePlot included, is read from the validation split and scored
with numeric F1 from deduplicated value sets (paired_bootstrap.chartx_items),
so the figures match Table 2 of the paper and no comparison crosses metrics
or splits. The application's Benchmarks panel shows the same two files:
pass --out-dir <application repository>/assets to refresh them.

Both figures are sized to be included at the width they are drawn, so the
text prints close to the size set here: the bars at the full text width
of the paper (6.5 in), the heatmap at nine tenths of it.

Colours are set in systems.py, one hue per provider.

Figures are written to out/ unless --out-dir says otherwise.

Usage:
    python benchmarks/plot_chartx_validation.py
    python benchmarks/plot_chartx_validation.py --out-dir <directory>
"""

import argparse
import statistics
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap, Normalize
from matplotlib.patches import Rectangle

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from paths import VALIDATION_ROOT  # noqa: E402
from benchmarks.paired_bootstrap import (  # noqa: E402
    N_BOOT,
    chartx_validation,
    column,
    ranked,
    type_mean,
)
from benchmarks.shared import bootstrap_ci  # noqa: E402
from benchmarks.systems import (  # noqa: E402
    CHART_TYPES,
    DEPLOT,
    HEATMAP_FLOOR,
    TYPE_TICKS,
)

# Default inside the repo. Writing to the manuscript's figures/ is opt-in,
# because a default that writes outside the repository would create a stray
# sibling directory on a fresh clone and can overwrite a paper's figures.
DEFAULT_OUT = VALIDATION_ROOT / "out"

FONT = 8
# Orange stays visible on every shade of the blue heatmap scale.
OUTLINE = "#d95f02"


def draw_bars(scores, systems, out_path):
    fig, ax = plt.subplots(figsize=(6.5, 3.3))
    width = 0.84 / len(systems)
    x = np.arange(len(CHART_TYPES))

    for i, system in enumerate(systems):
        means, below, above = [], [], []
        for chart_type in CHART_TYPES:
            values = list(column(scores[system.key], "f1", [chart_type]).values())
            mean = statistics.mean(values) * 100
            low, high = bootstrap_ci(values, n_boot=N_BOOT)
            means.append(mean)
            below.append(mean - low * 100)
            above.append(high * 100 - mean)
        ax.bar(
            x + i * width - 0.42 + width / 2, means, width,
            label=system.name, color=system.colour,
            edgecolor="#333333", linewidth=0.3,
            yerr=[below, above], capsize=1.2,
            error_kw={"lw": 0.6, "ecolor": "#646c6f"},
        )

    ax.set_xticks(x)
    ax.set_xticklabels([TYPE_TICKS[t] for t in CHART_TYPES], fontsize=FONT)
    ax.set_ylabel("Numeric F1 (%)", fontsize=FONT)
    ax.set_ylim(0, 100)
    ax.set_yticks(range(0, 101, 20))
    ax.tick_params(axis="y", labelsize=FONT)
    ax.tick_params(axis="x", length=0)
    ax.yaxis.grid(True, color="#e5e9ee", linewidth=0.6)
    ax.set_axisbelow(True)
    # Matplotlib fills a legend column by column. Hand it the entries in the
    # order that makes the rows, read left to right, follow the bars.
    handles, labels = ax.get_legend_handles_labels()
    columns = 5
    rows = -(-len(handles) // columns)
    reading_order = [
        r * columns + c for c in range(columns) for r in range(rows)
        if r * columns + c < len(handles)
    ]
    ax.legend(
        [handles[i] for i in reading_order],
        [labels[i] for i in reading_order],
        ncol=columns, fontsize=FONT - 0.5, loc="upper center",
        bbox_to_anchor=(0.5, -0.17), frameon=False,
        columnspacing=1.2, handlelength=1.2, handletextpad=0.5,
    )
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def draw_heatmap(scores, systems, out_path):
    matrix = np.array([
        [type_mean(scores[system.key], t) * 100 for t in CHART_TYPES]
        for system in systems
    ])
    baseline = matrix[[s is DEPLOT for s in systems].index(True)]

    fig, ax = plt.subplots(figsize=(5.6, 3.3))
    norm = Normalize(vmin=HEATMAP_FLOOR, vmax=100, clip=True)
    # The darkest tenth of the scale is dropped so adjacent high scores keep
    # a visible difference.
    cmap = ListedColormap(plt.get_cmap("YlGnBu")(np.linspace(0.0, 0.9, 256)))
    image = ax.imshow(matrix, cmap=cmap, norm=norm, aspect="auto")

    ax.set_xticks(np.arange(len(CHART_TYPES)))
    ax.set_xticklabels([TYPE_TICKS[t] for t in CHART_TYPES], fontsize=FONT)
    ax.set_yticks(np.arange(len(systems)))
    ax.set_yticklabels([s.name for s in systems], fontsize=FONT)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    for i, system in enumerate(systems):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            # Pick the text colour from the cell's own relative luminance
            # (linear light, as WCAG defines contrast), not from a threshold
            # on the value: white on the cells where it gives the higher
            # contrast, black on the rest.
            linear = [
                c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4
                for c in cmap(norm(value))[:3]
            ]
            luminance = (
                0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]
            )
            ax.text(
                j, i, f"{value:.1f}", ha="center", va="center",
                color="white" if luminance < 0.179 else "black", fontsize=FONT,
            )
            # Outline the cells where a VLM scores below DePlot.
            if system is not DEPLOT and value < baseline[j]:
                ax.add_patch(Rectangle(
                    (j - 0.46, i - 0.42), 0.92, 0.82,
                    fill=False, edgecolor=OUTLINE, linewidth=1.6, zorder=4,
                ))

    # Set the baseline row apart from the VLM rows.
    ax.axhline(len(systems) - 1.5, color="white", linewidth=3, zorder=3)
    bar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.02, extend="min")
    bar.set_label("Numeric F1 (%)", fontsize=FONT)
    bar.ax.tick_params(labelsize=FONT)
    # A thin outline, or the pale arrow marking "below the scale" cannot be
    # seen against the page.
    for spine in bar.ax.spines.values():
        spine.set_edgecolor("#9aa0a6")
        spine.set_linewidth(0.5)
    fig.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    scores = chartx_validation()
    systems = [*ranked(scores), DEPLOT]

    draw_bars(scores, systems, args.out_dir / "chartx_by_type.png")
    draw_heatmap(scores, systems, args.out_dir / "chartx_heatmap.png")
    print(f"Wrote chartx_by_type.png and chartx_heatmap.png to {args.out_dir}")

    print("\nNumeric F1, ChartX validation split:")
    for system in systems:
        values = list(column(scores[system.key], "f1").values())
        print(f"  {system.name:24} n={len(values):3d}  "
              f"{statistics.mean(values) * 100:5.1f}%")


if __name__ == "__main__":
    main()
