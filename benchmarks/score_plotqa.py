"""Rescore the stored PlotQA results against the plotted values.

Each entry of external/plotqa_test_1000.json describes ONE series of one
chart (the first), while every model under test emits the whole table. Two
scores are computed from the stored reply text, with numeric F1
(compute_numeric_f1):

    best-series   each column of the reply (header row left out) and each row
                  (first cell left out) is scored against the annotated
                  series, and the best one is kept
    whole-table   every number in the reply against the annotated series

Best-series is optimistic: the ground truth chooses which series of the reply
is graded, so a reply that reads the values correctly but attaches them to the
wrong series still scores 100%. Whole-table charges a reply for correctly
extracting the series the annotation omits, which is why it sits far lower
for every system. Neither measures whether the right table was produced.

Both work on sets, so a value that a series repeats counts once. That
penalises rounding: a reply that writes 76 six times for a series running
from 75.7 to 76.7 matches one value. ``Scores.repeats`` is the best-series
score with repeated values kept on both sides, which removes that penalty,
and paired_bootstrap.py prints how the comparison with DePlot changes.

THE GROUND TRUTH. The runners stored ``gt_nums`` taken from ``y_values``. For
a horizontal bar chart those are the category labels, and the bar lengths are
in ``x_values``; 427 of the 529 scored items are horizontal bar charts whose
categories are years. Scores computed from the stored field (every PlotQA
figure in version 1 of the paper, and in drafts of version 2 before 5 October
2026) therefore mostly measured whether a reply contained a column of years.
This script ignores the stored field and rebuilds the truth from the
annotations with shared.plotqa_truth.

The 529 items are the entries of the 1000-item subset for which that
mistaken field was numeric, because the runners skipped the rest: 427
horizontal bar charts with years on the category axis and 102 other charts.

DePlot is scored like the VLMs, after its TITLE row is removed
(shared.deplot_rows explains why).

Paired intervals for the differences printed here come from
paired_bootstrap.py.

Usage:
    python benchmarks/score_plotqa.py
    python benchmarks/score_plotqa.py --json out/plotqa_scores.json
"""

import argparse
import io
import json
import statistics
import sys
from pathlib import Path
from typing import NamedTuple

if isinstance(sys.stdout, io.TextIOWrapper):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from paths import EXTERNAL_DIR, PLOTQA_RESULTS_DIR  # noqa: E402
from benchmarks.shared import (  # noqa: E402
    compute_numeric_f1,
    extract_number_list,
    extract_numbers,
    plotqa_truth,
    within_tolerance,
)
from benchmarks.systems import BY_PLOTQA_KEY, DEPLOT  # noqa: E402

ANNOTATIONS = EXTERNAL_DIR / "plotqa_test_1000.json"

# The two Claude runs made with the detailed prompt (prompts.md), and the
# reported simple-prompt run each is compared with.
DETAILED_PROMPT_RUNS = {
    "claude-sonnet-4-6": "detailed_prompt_sonnet.json",
    "claude-haiku-4-5-20251001": "detailed_prompt_haiku.json",
}


# Items whose annotated values have a median magnitude at or above this are
# reported apart; the split was made after the results were seen.
LARGE_VALUE = 1e6


class Scores(NamedTuple):
    """Scores of one reply against one annotated series.

    best        best-series numeric F1, on sets
    whole       whole-table numeric F1, on sets
    horizontal  whether the chart is a horizontal bar chart
    repeats     best-series numeric F1 with repeated values kept
    large       whether the median annotated magnitude reaches LARGE_VALUE
    errors      relative error of each annotated value that the best series
                reproduces within the tolerance
    rows        table rows after the header row
    values      annotated values, repeats kept
    """

    best: float
    whole: float
    horizontal: bool
    repeats: float
    large: bool
    errors: tuple
    rows: int
    values: int


def annotations():
    """{imgname: entry} for the PlotQA subset."""
    return {
        entry["imgname"]: entry
        for entry in json.loads(ANNOTATIONS.read_text(encoding="utf-8"))
    }


def parse_table(text):
    """Split a reply into rows of cells.

    DePlot writes ``<0x0A>`` between rows and ``|`` between cells, and starts
    with a TITLE row, which is dropped. The VLMs write newlines and tabs, and
    now and then a Markdown table, whose fence, outer pipes and rule row are
    removed. A table aligned with spaces is not split and counts as a single
    column; no reply of a reported system is laid out that way.
    """
    body = text.replace("<0x0A>", "\n").strip()
    if body.startswith("```"):
        body = body.split("\n", 1)[1] if "\n" in body else ""
        body = body.rsplit("```", 1)[0]
    lines = [line for line in body.split("\n") if line.strip()]
    tabbed = any("\t" in line for line in lines)
    grid = []
    for line in lines:
        if tabbed:
            cells = [cell.strip() for cell in line.split("\t")]
        else:
            cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if all(cell and set(cell) <= set("-: ") for cell in cells):
            continue
        grid.append(cells)
    if grid and grid[0][0].upper() == "TITLE":
        grid = grid[1:]
    return grid


def series_candidates(grid):
    """Every column (without the header row) and row (without its label).

    Each candidate is the list of its numbers in reading order, repeats kept;
    the set-based scores deduplicate it.
    """
    candidates = []
    if not grid:
        return candidates
    body = grid[1:] if len(grid) > 1 else grid
    for j in range(max(len(row) for row in grid)):
        numbers = []
        for row in body:
            if j < len(row):
                numbers += extract_number_list(row[j])
        if numbers:
            candidates.append(numbers)
    for row in grid:
        numbers = []
        for cell in row[1:]:
            numbers += extract_number_list(cell)
        if numbers:
            candidates.append(numbers)
    return candidates


def relative_errors(candidate, truth, tolerance):
    """Relative error of each truth value that the candidate reproduces."""
    errors = []
    for value in truth:
        if value == 0:
            continue
        nearest = min(candidate, key=lambda e: abs(e - value))
        if within_tolerance(nearest, value, tolerance):
            errors.append(abs(nearest - value) / abs(value))
    return tuple(errors)


def score_items(path, tolerance=0.05):
    """Return (model, {imgname: Scores}) for one result file."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    entries = annotations()
    items = {}
    for row in data["results"]:
        entry = entries[row["imgname"]]
        values = plotqa_truth(entry)
        truth = set(values)
        grid = parse_table(row.get("extracted_text") or "")
        everything = set()
        for cells in grid:
            for cell in cells:
                everything |= extract_numbers(cell)
        candidates = series_candidates(grid)
        best = [
            compute_numeric_f1(set(candidate), truth, tolerance)
            for candidate in candidates
        ]
        repeats = [
            compute_numeric_f1(candidate, values, tolerance)
            for candidate in candidates
        ]
        chosen = candidates[best.index(max(best))] if best else []
        items[row["imgname"]] = Scores(
            best=max(best) if best else 0.0,
            whole=compute_numeric_f1(everything, truth, tolerance),
            horizontal="<s_width>" in entry["raw_text"],
            repeats=max(repeats) if repeats else 0.0,
            large=statistics.median(map(abs, values)) >= LARGE_VALUE,
            errors=relative_errors(chosen, truth, tolerance) if chosen else (),
            rows=max(len(grid) - 1, 0),
            values=len(values),
        )
    return data["model"], items


def reported_items(tolerance=0.05):
    """Per-item scores of the reported systems, keyed by PlotQA model key."""
    found = {}
    for path in sorted(Path(PLOTQA_RESULTS_DIR).glob("plotqa_*.json")):
        model, items = score_items(path, tolerance)
        if model in BY_PLOTQA_KEY:
            if model in found:
                sys.exit(f"Two PlotQA result files for {model}")
            found[model] = items
    missing = [
        system.name for key, system in BY_PLOTQA_KEY.items() if key not in found
    ]
    if missing:
        sys.exit(f"No PlotQA results for: {', '.join(missing)}")
    return found


def detailed_prompt_items(tolerance=0.05):
    """Per-item scores of the detailed-prompt Claude runs, same keys."""
    return {
        model: score_items(Path(PLOTQA_RESULTS_DIR) / name, tolerance)[1]
        for model, name in DETAILED_PROMPT_RUNS.items()
    }


def mean_of(items, field, horizontal=None):
    """Mean of one score, over all items or one orientation, in per cent."""
    values = [
        getattr(scores, field)
        for scores in items.values()
        if horizontal is None or scores.horizontal == horizontal
    ]
    return statistics.mean(values) * 100


def median_error(items):
    """Median relative error of the reproduced values, in per cent."""
    errors = [e for scores in items.values() for e in scores.errors]
    return statistics.median(errors) * 100


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path, help="also write the table as JSON")
    args = parser.parse_args()

    scores = reported_items()
    rows = [
        {
            "model": model,
            "name": BY_PLOTQA_KEY[model].name,
            "n": len(items),
            "best_series_f1": mean_of(items, "best"),
            "best_series_f1_horizontal_bars": mean_of(items, "best", True),
            "best_series_f1_other_charts": mean_of(items, "best", False),
            "whole_table_f1": mean_of(items, "whole"),
        }
        for model, items in scores.items()
    ]
    rows.sort(key=lambda r: -r["best_series_f1"])
    deplot = next(r for r in rows if r["model"] == DEPLOT.plotqa_key)
    sample = scores[DEPLOT.plotqa_key]
    horizontal = sum(1 for s in sample.values() if s.horizontal)

    print("PlotQA, simple prompt, numeric F1 (5% relative tolerance), against "
          "the plotted values\n")
    print(f"{'system':24}{'n':>5}{'best-series':>13}{'vs DePlot':>11}"
          f"{'horiz. bars':>13}{'other':>8}{'whole-table':>13}")
    print("-" * 87)
    for r in rows:
        margin = r["best_series_f1"] - deplot["best_series_f1"]
        print(
            f"{r['name']:24}{r['n']:>5}{r['best_series_f1']:12.1f}%"
            f"{'' if r is deplot else f'{margin:+.1f}':>11}"
            f"{r['best_series_f1_horizontal_bars']:12.1f}%"
            f"{r['best_series_f1_other_charts']:7.1f}%"
            f"{r['whole_table_f1']:12.1f}%"
        )
    print(
        f"\n{horizontal} of the {len(sample)} items are horizontal bar charts, "
        f"{len(sample) - horizontal} are other charts."
        "\nBest-series lets the ground truth choose which column or row of "
        "the reply is graded,\nso it is optimistic; whole-table charges every "
        "system for the series the annotation\nomits. "
        "paired_bootstrap.py prints the intervals."
    )

    print("\nClaude, detailed prompt against simple prompt, best-series\n")
    print(f"{'model':24}{'n':>5}{'detailed':>11}{'simple':>9}{'difference':>12}")
    print("-" * 61)
    for model, items in detailed_prompt_items().items():
        detailed = mean_of(items, "best")
        simple = mean_of(scores[model], "best")
        print(
            f"{BY_PLOTQA_KEY[model].name:24}{len(items):>5}"
            f"{detailed:10.2f}%{simple:8.2f}%{detailed - simple:+12.2f}"
        )

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(rows, indent=2) + "\n")
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
