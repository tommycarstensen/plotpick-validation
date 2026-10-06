"""Paired bootstrap intervals behind the paper's comparative statements.

Every system was scored on the same items, so the uncertainty of a difference
between two systems is that of the per-item differences, not of two separate
means. This script resamples those per-item differences (10,000 draws,
percentile 95% interval, fixed seed) for each comparison the paper makes:

    ChartX   each VLM against DePlot, overall and per chart type
    ChartX   each row of Table 2 against the row below it
    ChartX   the same margins if DePlot's title line is scored as output
    ChartX   the same margins without the numbers that occur only in labels
    ChartX   the same margins if a missing reply is scored as zero
    ChartX   the three unreported moving-alias runs against DePlot
    PlotQA   each VLM against DePlot, best-series and whole-table
    PlotQA   the two unreported moving-alias runs against DePlot
    PlotQA   the same by chart orientation, by magnitude of the values and
             with repeated values kept
    PlotQA   the detailed prompt against the simple prompt, for Claude
    both     every system at 1, 2, 5 and 10 per cent tolerance

The intervals are unadjusted for the number of comparisons. A comparison is
called separable when its interval excludes zero. The Claude Haiku 4.5 run
has no result for one ChartX item, so its comparisons pair the 299 shared
items.

DePlot's replies are scored without their TITLE row (shared.deplot_rows).

export_paper_numbers.py imports the loaders and interval function from here,
so the macros in the paper and the tables printed by this script cannot
disagree.

Usage:
    python benchmarks/paired_bootstrap.py
"""

import ast
import io
import json
import statistics
import sys
from itertools import pairwise
from pathlib import Path
from typing import NamedTuple

import numpy as np

if isinstance(sys.stdout, io.TextIOWrapper):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from paths import (  # noqa: E402
    CHARTX_RESULTS_DIR,
    CHARTX_VAL_META,
    FINAL_VAL_RESULTS_DIR,
    RESULTS_DIR,
)
from benchmarks.score_plotqa import (  # noqa: E402
    detailed_prompt_items,
    mean_of,
    reported_items,
    score_items,
)
from benchmarks.shared import (  # noqa: E402
    compute_numeric_f1,
    compute_recall,
    deplot_rows,
    extract_numbers,
    extract_numbers_from_rows,
    parse_chartx_csv,
)
from benchmarks.systems import (  # noqa: E402
    BY_KEY,
    BY_PLOTQA_KEY,
    CHART_TYPES,
    DEPLOT,
    TYPE_PROSE,
    VLMS,
)

N_BOOT = 10000
SEED = 0

# Relative tolerances at which every system is rescored; the paper's is 5%.
TOLERANCES = (0.01, 0.02, 0.05, 0.10)

# Run made under a moving alias, mentioned in the paper but not tabulated.
MINISTRAL_RUN = (
    RESULTS_DIR / "archive" / "alias_runs" / "final_val_ministral_3b_latest.json"
)


class Item(NamedTuple):
    chart_type: str
    f1: float
    recall: float


class Interval(NamedTuple):
    """Mean paired difference in percentage points, with its 95% interval."""

    mean: float
    low: float
    high: float
    n: int

    @property
    def separable(self):
        return self.low > 0 or self.high < 0

    def __str__(self):
        return f"{self.mean:+5.1f} [{self.low:+5.1f}, {self.high:+5.1f}]"


def paired_interval(a, b, n_boot=N_BOOT, seed=SEED):
    """Bootstrap the mean of a[k] - b[k] over the items both systems have."""
    shared = sorted(set(a) & set(b))
    diffs = np.array([a[k] - b[k] for k in shared], dtype=float) * 100
    rng = np.random.default_rng(seed)
    draws = rng.choice(diffs, size=(n_boot, diffs.size), replace=True).mean(axis=1)
    low, high = np.percentile(draws, [2.5, 97.5])
    return Interval(float(diffs.mean()), float(low), float(high), len(shared))


def as_numbers(value):
    """Accept a JSON list or its repr; deduplicate, as the VLM runs store."""
    if isinstance(value, str):
        value = ast.literal_eval(value)
    return set(value)


def label_only_numbers():
    """{imgname: numbers found only in labels} for the six reported types.

    A number is label-only when it occurs in the header row or the first
    column of the ground-truth table and in no other cell: a year, the 0 and
    -10 read from a histogram bin "0-10", the 1 and 3 of "Q1" and "Q3". The
    metric counts them like any other number, so a system is credited for
    copying labels.
    """
    found = {}
    for record in json.loads(CHARTX_VAL_META.read_text()):
        if record["chart_type"] not in CHART_TYPES:
            continue
        rows = [
            [cell.strip() for cell in line.split("\t")]
            for line in parse_chartx_csv(record["csv"]).strip().split("\n")
            if line.strip()
        ]
        labels, data = set(), set()
        for cell in rows[0]:
            labels |= extract_numbers(cell)
        for row in rows[1:]:
            labels |= extract_numbers(row[0])
            for cell in row[1:]:
                data |= extract_numbers(cell)
        found[record["imgname"]] = (labels - data, labels | data)
    return found


def chartx_items(path, tolerance=0.05, deplot_title=False,
                 chart_types=CHART_TYPES, labels=None):
    """Return (model, {imgname: Item}) for the six reported chart types.

    VLM files store a flat list of rows; DePlot files store rows by chart
    type. Scores are recomputed from the stored number sets rather than read
    from the file: benchmark_deplot.py keeps duplicate values in its ground
    truth while the VLM runs store deduplicated values, and a repeated
    ground-truth value inflates the recall denominator. Scoring every system
    from deduplicated values is what makes the comparison like for like.

    DePlot's numbers are taken from its stored reply without the TITLE row
    (shared.deplot_rows). ``deplot_title=True`` keeps the title row, as the
    runner first scored it, to show what that convention costs.
    ``chart_types`` selects other ChartX types than the six reported.
    ``labels`` (from label_only_numbers) removes the numbers that occur only
    in labels from the ground truth and from the extraction.
    """
    data = json.loads(Path(path).read_text())
    is_deplot = data.get("model") == DEPLOT.key
    results = data["results"]
    if isinstance(results, dict):
        rows = [(t, row) for t, group in results.items() for row in group]
    else:
        rows = [(row["chart_type"], row) for row in results]
    items = {}
    for chart_type, row in rows:
        if chart_type not in chart_types:
            continue
        if is_deplot:
            extracted = extract_numbers_from_rows(
                deplot_rows(row["extracted_text"], keep_title=deplot_title))
        else:
            extracted = as_numbers(row["ex_nums"])
        truth = as_numbers(row["gt_nums"])
        if labels is not None:
            drop, everything = labels[row["imgname"]]
            if everything != truth:
                sys.exit(f"{path}: {row['imgname']} truth differs from the "
                         "annotation file")
            extracted, truth = extracted - drop, truth - drop
        # Image names are unique within the six reported types. Across all
        # 18 they are not, so other selections are keyed by type as well.
        name = row["imgname"]
        if chart_types is not CHART_TYPES:
            name = f"{chart_type}/{name}"
        if name in items:
            sys.exit(f"{path}: {name} occurs twice")
        items[name] = Item(
            chart_type,
            compute_numeric_f1(extracted, truth, tolerance),
            compute_recall(extracted, truth, tolerance),
        )
    return data.get("model"), items


def _vlm_runs(directory, tolerance=0.05, labels=None):
    """{system key: items} for the reported VLMs with a run in a directory.

    The directories also hold runs of systems the paper does not report, in
    other layouts, so a file is scored only once its model is recognised.
    """
    found = {}
    for path in sorted(Path(directory).glob("*.json")):
        model = json.loads(path.read_text()).get("model")
        if model in BY_KEY and model != DEPLOT.key:
            if model in found:
                sys.exit(f"Two result files for {model} in {directory}")
            found[model] = chartx_items(path, tolerance, labels=labels)[1]
    return found


def chartx_validation(tolerance=0.05, deplot_title=False, labels=None):
    """Per-item scores on the ChartX validation split, for every system."""
    found = _vlm_runs(FINAL_VAL_RESULTS_DIR, tolerance, labels)
    found[DEPLOT.key] = chartx_items(
        CHARTX_RESULTS_DIR / "chartx_deplot_val.json", tolerance, deplot_title,
        labels=labels)[1]
    missing = [s.name for s in VLMS if s.key not in found]
    if missing:
        sys.exit(f"No validation-split results for: {', '.join(missing)}")
    return found


def with_missing_as_zero(items, reference):
    """Add a zero score for every item of ``reference`` a run has no reply for.

    Empty replies were removed from the result files in April 2026 instead of
    being scored. This puts them back as failures.
    """
    filled = dict(items)
    for name, item in reference.items():
        filled.setdefault(name, Item(item.chart_type, 0.0, 0.0))
    return filled


def chartx_development():
    """Per-item scores on the development split, for the VLMs that have one."""
    return _vlm_runs(CHARTX_RESULTS_DIR)


def column(items, field, chart_types=CHART_TYPES):
    """{imgname: score} for one field, restricted to some chart types."""
    return {
        name: getattr(item, field)
        for name, item in items.items()
        if item.chart_type in chart_types
    }


def type_mean(items, chart_type, field="f1"):
    return statistics.mean(column(items, field, [chart_type]).values())


def cell_intervals(scores, field="f1"):
    """{(system key, chart type): Interval against DePlot} for every VLM."""
    deplot = scores[DEPLOT.key]
    return {
        (system.key, chart_type): paired_interval(
            column(scores[system.key], field, [chart_type]),
            column(deplot, field, [chart_type]),
        )
        for system in VLMS
        for chart_type in CHART_TYPES
    }


def ranked(scores):
    """The VLMs ordered by overall numeric F1, as Table 2 lists them."""
    return sorted(
        VLMS,
        key=lambda s: -statistics.mean(column(scores[s.key], "f1").values()),
    )


def overall(items, field="f1"):
    """Mean score over the six chart types, in per cent."""
    return statistics.mean(column(items, field).values()) * 100


def field_of(items, field, where=None):
    """{imgname: score} for one field of PlotQA Scores.

    ``where`` keeps the items for which it returns true, for example
    ``lambda s: s.horizontal``.
    """
    return {
        name: getattr(scores, field)
        for name, scores in items.items()
        if where is None or where(scores)
    }


# The subsets of the PlotQA items that the paper reports apart.
PLOTQA_SUBSETS = {
    "horizontal bar charts": lambda s: s.horizontal,
    "other charts": lambda s: not s.horizontal,
    "median value below 1e6": lambda s: not s.large,
    "median value 1e6 or more": lambda s: s.large,
}


def plotqa_order(plotqa):
    """PlotQA model keys, VLMs only, by best-series score, highest first."""
    return sorted(
        (key for key in plotqa if key != DEPLOT.plotqa_key),
        key=lambda k: -statistics.mean(field_of(plotqa[k], "best").values()),
    )


def flag(interval):
    return "" if interval.separable else "  spans zero"


def main():
    scores = chartx_validation()
    deplot = scores[DEPLOT.key]
    print(f"Paired bootstrap, {N_BOOT} draws, seed {SEED}, 95% intervals, "
          "unadjusted.\nDifferences are in percentage points of numeric F1 "
          "unless stated.\n")

    print("ChartX validation split: each VLM minus DePlot, six chart types")
    print(f"{'':22}{'n':>4}  {'numeric F1':24}{'recall':24}")
    for system in ranked(scores):
        f1 = paired_interval(column(scores[system.key], "f1"),
                             column(deplot, "f1"))
        recall = paired_interval(column(scores[system.key], "recall"),
                                 column(deplot, "recall"))
        print(f"{system.name:22}{f1.n:>4}  {f1!s:24}{recall!s:24}")

    non_box = [t for t in CHART_TYPES if t != "box"]
    print("\nThe same, over the five chart types other than box plots")
    for system in ranked(scores):
        interval = paired_interval(column(scores[system.key], "f1", non_box),
                                   column(deplot, "f1", non_box))
        print(f"{system.name:22}{interval.n:>4}  {interval}{flag(interval)}")

    for field, label in (("f1", "numeric F1"), ("recall", "recall")):
        cells = cell_intervals(scores, field)
        lead = sum(1 for c in cells.values() if c.low > 0)
        trail = sum(1 for c in cells.values() if c.high < 0)
        below = sum(1 for c in cells.values() if c.mean < 0)
        print(f"\nChartX per cell, {label}: each VLM minus DePlot")
        for system in ranked(scores):
            print(f"  {system.name}")
            for chart_type in CHART_TYPES:
                interval = cells[system.key, chart_type]
                print(f"    {TYPE_PROSE[chart_type]:30}{interval.n:>4}  "
                      f"{interval}{flag(interval)}")
        print(f"  {len(cells)} cells: {lead} separable in the VLM's favour, "
              f"{trail} separable in DePlot's, "
              f"{len(cells) - lead - trail} spanning zero; "
              f"{below} below DePlot by point estimate.")

    print("\nChartX: each row of Table 2 minus the row below it")
    for upper, lower in pairwise(ranked(scores)):
        interval = paired_interval(column(scores[upper.key], "f1"),
                                   column(scores[lower.key], "f1"))
        print(f"{upper.name:22}- {lower.name:22}{interval.n:>4}  "
              f"{interval}{flag(interval)}")

    titled = chartx_validation(deplot_title=True)[DEPLOT.key]
    print("\nChartX: each VLM minus DePlot if DePlot's title line is scored "
          f"(DePlot {overall(titled):.1f}% instead of {overall(deplot):.1f}%); "
          "six types, then the five other than box plots")
    for system in ranked(scores):
        interval = paired_interval(column(scores[system.key], "f1"),
                                   column(titled, "f1"))
        off_box = paired_interval(column(scores[system.key], "f1", non_box),
                                  column(titled, "f1", non_box))
        print(f"{system.name:22}{interval.n:>4}  {interval!s:24}{off_box}"
              f"{flag(off_box)}")

    plain = chartx_validation(labels=label_only_numbers())
    print("\nChartX: each VLM minus DePlot without the numbers that occur "
          f"only in labels (DePlot {overall(plain[DEPLOT.key]):.1f}%)")
    for system in ranked(scores):
        interval = paired_interval(column(plain[system.key], "f1"),
                                   column(plain[DEPLOT.key], "f1"))
        print(f"{system.name:22}{interval.n:>4}  {interval}{flag(interval)}")

    print("\nChartX: runs with a missing reply, the reply scored as zero")
    _, ministral = chartx_items(MINISTRAL_RUN)
    for name, items in [
        *((s.name, scores[s.key]) for s in ranked(scores)),
        ("ministral-3b-latest", ministral),
    ]:
        if len(items) == len(deplot):
            continue
        filled = with_missing_as_zero(items, deplot)
        interval = paired_interval(column(filled, "f1"), column(deplot, "f1"))
        print(f"{name:22}{interval.n:>4}  {overall(filled):.1f}% instead of "
              f"{overall(items):.1f}%  {interval}{flag(interval)}")

    alias_dir = MINISTRAL_RUN.parent
    print("\nChartX: the unreported alias runs minus DePlot")
    for path in sorted(alias_dir.glob("final_val_*.json")):
        model, items = chartx_items(path)
        for field, label in (("f1", "numeric F1"), ("recall", "recall")):
            interval = paired_interval(column(items, field),
                                       column(deplot, field))
            print(f"{model:24}{label:12}{interval.n:>4}  "
                  f"{interval}{flag(interval)}")

    plotqa = reported_items()
    deplot_qa = plotqa[DEPLOT.plotqa_key]
    print("\nPlotQA: each VLM minus DePlot")
    print(f"{'':22}{'n':>4}  {'best-series':24}{'whole-table':24}")
    for key in plotqa_order(plotqa):
        best = paired_interval(field_of(plotqa[key], "best"),
                               field_of(deplot_qa, "best"))
        whole = paired_interval(field_of(plotqa[key], "whole"),
                                field_of(deplot_qa, "whole"))
        print(f"{BY_PLOTQA_KEY[key].name:22}{best.n:>4}  "
              f"{best!s:24}{whole!s:24}")

    print("\nPlotQA best-series: the unreported alias runs minus DePlot")
    for path in sorted(alias_dir.glob("plotqa_*.json")):
        model, items = score_items(path)
        interval = paired_interval(field_of(items, "best"),
                                   field_of(deplot_qa, "best"))
        print(f"{model:24}{interval.n:>4}  {mean_of(items, 'best'):.1f}%  "
              f"{interval}{flag(interval)}")

    for label, where in PLOTQA_SUBSETS.items():
        size = len(field_of(deplot_qa, "best", where))
        print(f"\nPlotQA best-series, {label} (n={size}): each VLM minus "
              "DePlot")
        for key in plotqa_order(plotqa):
            interval = paired_interval(field_of(plotqa[key], "best", where),
                                       field_of(deplot_qa, "best", where))
            print(f"{BY_PLOTQA_KEY[key].name:22}{interval.n:>4}  "
                  f"{interval}{flag(interval)}")

    print("\nPlotQA best-series with repeated values kept: each VLM minus "
          "DePlot")
    for key in plotqa_order(plotqa):
        interval = paired_interval(field_of(plotqa[key], "repeats"),
                                   field_of(deplot_qa, "repeats"))
        print(f"{BY_PLOTQA_KEY[key].name:22}{interval.n:>4}  "
              f"{interval}{flag(interval)}")

    print("\nPlotQA best-series: detailed prompt minus simple prompt")
    for key, items in detailed_prompt_items().items():
        interval = paired_interval(field_of(items, "best"),
                                   field_of(plotqa[key], "best"))
        print(f"{BY_PLOTQA_KEY[key].name:22}{interval.n:>4}  "
              f"{interval.mean:+6.2f} [{interval.low:+6.2f}, "
              f"{interval.high:+6.2f}]{flag(interval)}")

    print("\nEvery system at four tolerances (per cent; in brackets the "
          "margin over DePlot)")
    header = "".join(f"{f'{t:.0%}':>16}" for t in TOLERANCES)
    print(f"{'ChartX numeric F1':22}{header}")
    by_tolerance = {t: chartx_validation(t) for t in TOLERANCES}
    for system in [*ranked(scores), DEPLOT]:
        cells = ""
        for t in TOLERANCES:
            value = overall(by_tolerance[t][system.key])
            base = overall(by_tolerance[t][DEPLOT.key])
            margin = "" if system is DEPLOT else f" ({value - base:+.1f})"
            cells += f"{f'{value:.1f}{margin}':>16}"
        print(f"{system.name:22}{cells}")
    print(f"{'PlotQA best-series':22}{header}")
    qa_by_tolerance = {t: reported_items(t) for t in TOLERANCES}
    for key in [*plotqa_order(plotqa), DEPLOT.plotqa_key]:
        cells = ""
        for t in TOLERANCES:
            best = field_of(qa_by_tolerance[t][key], "best")
            value = statistics.mean(best.values()) * 100
            base = statistics.mean(field_of(
                qa_by_tolerance[t][DEPLOT.plotqa_key], "best").values()) * 100
            margin = "" if key == DEPLOT.plotqa_key else f" ({value - base:+.1f})"
            cells += f"{f'{value:.1f}{margin}':>16}"
        print(f"{BY_PLOTQA_KEY[key].name:22}{cells}")


if __name__ == "__main__":
    main()
