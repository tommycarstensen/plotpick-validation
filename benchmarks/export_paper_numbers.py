"""Generate the manuscript's numbers as LaTeX macros from the result files.

Writes numbers.tex, which main.tex pulls in with \\input{numbers}. Every
figure the paper quotes about its own results is defined here and computed
from results/, so a number cannot drift from the data it came from. When a
benchmark is re-run, re-run this and the paper follows.

Why this exists: over the course of preparing v2 several hardcoded figures
went stale as the underlying runs changed (a DePlot score, a per-type range,
a count of models beating a baseline), and each was caught only by reading
the tables against the data by hand.

LaTeX command names may not contain digits, so each system has a letters-only
suffix (systems.py): gemini-3-flash-preview -> GeminiFlash, and so on. The
metric the paper calls numeric F1 is "Fone" in macro names.

Counts below ten are written as words, for running text. Intervals come from
paired_bootstrap.py and are seeded, so two runs of this script write the same
file (for a given numpy version).

The paper also states things in words: that every VLM leads DePlot in
aggregate, that the trailing models are the lowest-ranked ones, that two
scorings order the systems alike. Each such statement is checked by require()
below, and the script stops if one no longer holds, so the wording cannot
outlive the data either.

Macros are written to out/ unless --out says otherwise.

Usage:
    python benchmarks/export_paper_numbers.py
    python benchmarks/export_paper_numbers.py --out <paper directory>/numbers.tex
"""

import argparse
import inspect
import json
import statistics
import sys
from itertools import pairwise
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from paths import (  # noqa: E402
    CHARTX_DIR,
    CHARTX_META,
    CHARTX_RESULTS_DIR,
    CHARTX_VAL_META,
    EXTERNAL_DIR,
    PMC_RESULTS_DIR,
    VALIDATION_ROOT,
)
from benchmarks.paired_bootstrap import (  # noqa: E402
    MINISTRAL_RUN,
    N_BOOT,
    SEED,
    TOLERANCES,
    cell_intervals,
    chartx_development,
    chartx_items,
    chartx_validation,
    column,
    field_of,
    overall,
    paired_interval,
    plotqa_order,
    ranked,
    type_mean,
)
from benchmarks.score_plotqa import (  # noqa: E402
    detailed_prompt_items,
    mean_of,
    reported_items,
)
from benchmarks.shared import (  # noqa: E402
    bootstrap_ci,
    deplot_rows,
    encode_image,
    extract_numbers,
    extract_numbers_from_rows,
)
from benchmarks.systems import (  # noqa: E402
    BY_KEY,
    BY_PLOTQA_KEY,
    CHART_TYPES,
    DEPLOT,
    HEATMAP_FLOOR,
    LABELLED_PAIRS,
    TYPE_PROSE,
    VLMS,
)

# Default inside the repo so a fresh clone cannot silently create a stray
# sibling directory and cannot overwrite a paper's committed numbers.tex.
DEFAULT_OUT = VALIDATION_ROOT / "out" / "numbers.tex"

# The relative tolerance compute_numeric_f1 and compute_recall default to.
TOLERANCE_PERCENT = 5

# A separable difference whose interval comes this close to zero (in points)
# flips with the bootstrap seed, so it is reported as borderline.
BORDERLINE = 0.1

# Macro infix for each tolerance of the sensitivity table.
TOLERANCE_MACRO = {0.01: "One", 0.02: "Two", 0.05: "Five", 0.10: "Ten"}

# The order in which the paper's tables list the systems.
TABLE_ORDER = [
    "GeminiFlash", "GeminiFlashLite", "GptMini", "Sonnet", "GptNano",
    "Haiku", "MistralMedium", "MistralSmall", "MistralLarge",
]
PLOTQA_TABLE_ORDER = [
    "GeminiFlash", "Sonnet", "DePlot", "GeminiFlashLite", "GptMini",
    "Haiku", "GptNano",
]

WORDS = {0: "no", 1: "one", 2: "two", 3: "three", 4: "four", 5: "five",
         6: "six", 7: "seven", 8: "eight", 9: "nine"}


def word(n):
    return WORDS.get(n, str(n))


def pct(value):
    return f"{value * 100:.1f}"


def minus(text):
    """Set a leading hyphen as a minus sign; LaTeX prints "-" as a hyphen."""
    return "$-$" + text[1:] if text.startswith("-") else text


def signed(value, digits=1):
    return minus(f"{value:+.{digits}f}")


def interval_text(interval, digits=1):
    return (f"[{signed(interval.low, digits)}, "
            f"{signed(interval.high, digits)}]")


def mean(values):
    return statistics.mean(values)


def join_words(parts):
    """["a", "b", "c"] -> "a, b and c"."""
    parts = list(parts)
    if len(parts) < 2:
        return "".join(parts)
    return ", ".join(parts[:-1]) + " and " + parts[-1]


def require(condition, statement):
    """Stop if a statement the paper makes in words has stopped holding."""
    if not condition:
        sys.exit(f"The paper says: {statement}. The data no longer do.")


def type_balanced_f1(items):
    """Mean of the six per-type means, so splits of unequal mix compare."""
    return mean(type_mean(items, t) for t in CHART_TYPES)


def unpaired_interval(a, b, n_boot=N_BOOT, seed=SEED):
    """95% interval for mean(a) - mean(b), two independent samples, points."""
    a = np.asarray(a, dtype=float) * 100
    b = np.asarray(b, dtype=float) * 100
    rng = np.random.default_rng(seed)
    draws = (
        rng.choice(a, size=(n_boot, a.size), replace=True).mean(axis=1)
        - rng.choice(b, size=(n_boot, b.size), replace=True).mean(axis=1)
    )
    low, high = np.percentile(draws, [2.5, 97.5])
    return float(low), float(high)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    out = [
        "% Generated by validation/benchmarks/export_paper_numbers.py.",
        "% Do not edit by hand: re-run that script after a benchmark changes.",
        "",
    ]

    def define(name, value):
        out.append(f"\\newcommand{{\\{name}}}{{{value}}}")

    def define_count(name, n, cap=False):
        define(name, word(n))
        if cap:
            define(name + "Cap", word(n).capitalize())

    def section(title):
        out.extend(["", f"% {title}"])

    scores = chartx_validation()
    deplot = scores[DEPLOT.key]
    order = ranked(scores)
    # The rows of Tables 2 to 4 are typed in the paper in this order.
    require([s.macro for s in order] == TABLE_ORDER,
            "Table 2 lists the VLMs by numeric F1")

    # --- Constants of the design -------------------------------------------
    out.append("% Design")
    define("Tolerance", str(TOLERANCE_PERCENT))
    example = 20.0
    define("ToleranceExample", f"{example:.1f}")
    define("ToleranceExampleLow",
           f"{example * (1 - TOLERANCE_PERCENT / 100):.1f}")
    define("ToleranceExampleHigh",
           f"{example * (1 + TOLERANCE_PERCENT / 100):.1f}")
    # What the tolerance can do to a standardised mean difference: two arm
    # means, one standard deviation, each mean off by the full tolerance in
    # the direction that shrinks or widens the difference.
    arm_a, arm_b, deviation = 20.0, 22.0, 5.0
    shift = TOLERANCE_PERCENT / 100
    define("SmdExampleMeans", f"{arm_a:.1f} and {arm_b:.1f}")
    define("SmdExampleSD", f"{deviation:.0f}")
    define("SmdExampleTrue", f"{(arm_b - arm_a) / deviation:.2f}")
    define("SmdExampleLow", minus(
        f"{(arm_b * (1 - shift) - arm_a * (1 + shift)) / deviation:.2f}"))
    define("SmdExampleHigh",
           f"{(arm_b * (1 + shift) - arm_a * (1 - shift)) / deviation:.2f}")
    define("ImageMaxSide", str(
        inspect.signature(encode_image).parameters["max_size"].default))
    define("BootstrapDraws", f"{N_BOOT:,}".replace(",", "{,}"))
    define("BorderlinePoints", f"{BORDERLINE:.1f}")
    define("HeatmapFloor", str(HEATMAP_FLOOR))
    define_count("ProviderCount", len({s.provider for s in VLMS}))
    mistral = sum(1 for s in VLMS if s.provider == "Mistral")
    define_count("MistralModelCount", mistral)
    define_count("NonMistralModelCount", len(VLMS) - mistral)
    chartx_meta = json.loads(CHARTX_VAL_META.read_text())
    types_total = len({r["chart_type"] for r in chartx_meta})
    define("ChartXTypesTotal", str(types_total))
    define("ChartXValItems", f"{len(chartx_meta):,}".replace(",", "{,}"))
    define("ChartXTestItems", f"{len(json.loads(CHARTX_META.read_text())):,}"
           .replace(",", "{,}"))
    define("ChartXTypesOther", str(types_total - len(CHART_TYPES)))
    define_count("ChartXTypeCount", len(CHART_TYPES))
    define_count("ChartXNonBoxTypeCount", len(CHART_TYPES) - 1)
    per_type = {len(column(deplot, "f1", [t])) for t in CHART_TYPES}
    require(len(per_type) == 1, "DePlot has the same number of items per type")
    define("ChartXPerType", str(per_type.pop()))
    corrections = json.loads((CHARTX_DIR / "chartx_prs.json").read_text())
    corrected = {c["imgname"] for c in corrections}
    require(not corrected & {r["imgname"] for r in chartx_meta},
            "no corrected item is in the validation split")
    define_count("ChartXCorrections", len(corrected))

    # --- ChartX, per system ------------------------------------------------
    section("ChartX validation split, per system")
    means = {}
    for system in [*order, DEPLOT]:
        f1 = list(column(scores[system.key], "f1").values())
        recall = list(column(scores[system.key], "recall").values())
        means[system.key] = mean(f1)
        low, high = bootstrap_ci(f1, n_boot=N_BOOT)
        define(f"ChartXFone{system.macro}", pct(mean(f1)))
        define(f"ChartXFoneCI{system.macro}",
               f"[{low * 100:.1f}, {high * 100:.1f}]")
        define(f"ChartXRec{system.macro}", pct(mean(recall)))
        define(f"ChartXN{system.macro}", str(len(f1)))
        if system is DEPLOT:
            continue
        delta = paired_interval(column(scores[system.key], "f1"),
                                column(deplot, "f1"))
        by_recall = paired_interval(column(scores[system.key], "recall"),
                                    column(deplot, "recall"))
        require(delta.low > 0 and by_recall.low > 0,
                f"{system.name} leads DePlot separably on numeric F1 and recall")
        define(f"ChartXDelta{system.macro}", signed(delta.mean))
        define(f"ChartXDeltaCI{system.macro}", interval_text(delta))

    # Rows with fewer items than DePlot: DePlot's score on the items they
    # share, so a reader can reproduce the paired margin by subtraction.
    short = [s for s in order if len(scores[s.key]) < len(deplot)]
    require([s.macro for s in short] == ["Haiku"],
            "only Claude Haiku 4.5 lacks an item")
    shared = {k: v for k, v in deplot.items() if k in scores[short[0].key]}
    define("ChartXFoneDePlotHaikuItems", f"{overall(shared):.1f}")

    # What scoring DePlot's title line as output would do.
    section("ChartX, DePlot's title line")
    reply = json.loads(
        (CHARTX_RESULTS_DIR / "chartx_deplot_val.json").read_text())["results"]
    titles = [
        row["extracted_text"].split("<0x0A>")[0]
        for chart_type in CHART_TYPES for row in reply[chart_type]
    ]
    require(all(t.strip().upper().startswith("TITLE") for t in titles),
            "every DePlot reply starts with a title line")
    define("ChartXDePlotTitlesWithNumber",
           str(sum(1 for t in titles if extract_numbers(t))))
    titled = chartx_validation(deplot_title=True)[DEPLOT.key]
    define("ChartXFoneDePlotTitled", f"{overall(titled):.1f}")

    # --- ChartX, aggregates the prose and abstract quote -------------------
    section("ChartX aggregates")
    lowest = order[-1]
    define("ChartXFoneVLMMin", pct(means[lowest.key]))
    define("ChartXFoneVLMMax", pct(means[order[0].key]))
    define("ChartXLowestVLMName", lowest.name)
    gap = paired_interval(column(scores[lowest.key], "f1"),
                          column(deplot, "f1"))
    define("ChartXGapLowestVLM", f"{gap.mean:.1f}")
    define("ChartXGapLowestVLMCI", interval_text(gap))
    highest = paired_interval(column(scores[order[0].key], "f1"),
                              column(deplot, "f1"))
    define("ChartXGapHighestVLM", f"{highest.mean:.1f}")
    define_count("ChartXModelCount", len(VLMS))

    f1_cells = cell_intervals(scores, "f1")
    recall_cells = cell_intervals(scores, "recall")
    trailing = {}
    for (key, chart_type), interval in f1_cells.items():
        if interval.mean < 0:
            trailing.setdefault(key, []).append(chart_type)
    require(set(trailing) == {s.key for s in order[-len(trailing):]},
            "the models that trail DePlot somewhere are the lowest-ranked")
    require(all("box" not in types for types in trailing.values()),
            "no VLM trails DePlot on box plots")
    define_count("ChartXSweepCount", len(VLMS) - len(trailing))
    define_count("ChartXTrailingModels", len(trailing))
    define_count("ChartXTrailingCells", sum(map(len, trailing.values())))
    define("ChartXTotalCells", str(len(f1_cells)))
    define("ChartXTrailingList", "; ".join(
        f"{BY_KEY[key].name} on "
        + join_words(TYPE_PROSE[t] for t in types)
        for key, types in trailing.items()))
    separable_trailing = [
        f"{BY_KEY[key].name} on {TYPE_PROSE[chart_type]}"
        for (key, chart_type), interval in f1_cells.items()
        if interval.high < 0
    ]
    define("ChartXTrailingSeparable", join_words(separable_trailing) or "none")
    define("ChartXCellsLeadSeparable",
           str(sum(1 for c in f1_cells.values() if c.low > 0)))
    define("ChartXCellsTrailSeparable", str(len(separable_trailing)))
    define("ChartXCellsSpanZero",
           str(sum(1 for c in f1_cells.values() if not c.separable)))
    define_count("ChartXTrailingCellsRecall",
                 sum(1 for c in recall_cells.values() if c.mean < 0))

    box = [type_mean(scores[s.key], "box") for s in VLMS]
    define("ChartXBoxDePlot", pct(type_mean(deplot, "box")))
    define("ChartXBoxVLMMin", pct(min(box)))
    define("ChartXBoxVLMMax", pct(max(box)))
    define("ChartXHistogramDePlot", pct(type_mean(deplot, "histogram")))

    # Margin over DePlot by chart type, from the paired cell differences:
    # averaged over the VLMs, and for the lowest-scoring VLM on that type.
    margin_mean, margin_lowest = {}, {}
    for chart_type in CHART_TYPES:
        cells = [f1_cells[s.key, chart_type].mean for s in VLMS]
        margin_mean[chart_type] = mean(cells)
        margin_lowest[chart_type] = min(cells)
    define("ChartXMarginBox", f"{margin_mean['box']:.1f}")
    non_box = [t for t in CHART_TYPES if t != "box"]
    require(margin_mean["box"] > 3 * max(margin_mean[t] for t in non_box),
            "the mean margin on box plots is several times any other")
    define("ChartXMarginMeanMin", f"{min(margin_mean[t] for t in non_box):.1f}")
    define("ChartXMarginMeanMax", f"{max(margin_mean[t] for t in non_box):.1f}")
    require(min(margin_lowest[t] for t in non_box) < 0
            < max(margin_lowest[t] for t in non_box),
            "the lowest VLM per non-box type is behind on some, ahead on others")
    define("ChartXMarginLowestMin",
           f"{abs(min(margin_lowest[t] for t in non_box)):.1f}")
    define("ChartXMarginLowestMax",
           f"{max(margin_lowest[t] for t in non_box):.1f}")

    # Pooled over the five non-box types, each VLM against DePlot.
    leads, level, behind = [], [], []
    for system in order:
        interval = paired_interval(column(scores[system.key], "f1", non_box),
                                   column(deplot, "f1", non_box))
        text = (f"{system.name} {signed(interval.mean)} "
                f"{interval_text(interval)}")
        if interval.low > 0:
            leads.append(interval.mean)
        elif interval.high < 0:
            behind.append(text)
        else:
            level.append(text)
    require(leads and len(leads) == len(VLMS) - len(trailing),
            "the models that never trail keep a separable lead off box plots")
    define("ChartXNonBoxLeadMin", f"{min(leads):.1f}")
    define("ChartXNonBoxLeadMax", f"{max(leads):.1f}")
    require(len(level) == 1 and len(behind) == 1,
            "off box plots one trailing model is level with DePlot, one behind")
    define("ChartXNonBoxLevel", level[0])
    define("ChartXNonBoxBehind", behind[0])

    # What DePlot returns for a box plot.
    box_rows = reply["box"]
    returned = [
        len(extract_numbers_from_rows(deplot_rows(row["extracted_text"])))
        for row in box_rows
    ]
    expected = [len(set(row["gt_nums"])) for row in box_rows]
    ceiling = [
        2 * min(r, e) / (r + e) if r + e else 1.0
        for r, e in zip(returned, expected, strict=True)
    ]
    define("ChartXBoxDePlotNumbers", f"{mean(returned):.1f}")
    define("ChartXBoxTruthNumbers", f"{mean(expected):.1f}")
    define("ChartXBoxDePlotCeiling", pct(mean(ceiling)))

    # Rows of Table 2 that a paired interval does not tell apart.
    not_separable, borderline = [], []
    for upper, lower in pairwise(order):
        interval = paired_interval(column(scores[upper.key], "f1"),
                                   column(scores[lower.key], "f1"))
        if not interval.separable:
            not_separable.append(
                f"{upper.name} and {lower.name} "
                f"({signed(interval.mean)} {interval_text(interval)})")
        elif min(abs(interval.low), abs(interval.high)) < BORDERLINE:
            borderline.append(
                f"{upper.name} and {lower.name} "
                f"({signed(interval.mean, 2)} {interval_text(interval, 2)})")
    define("ChartXAdjacentNotSeparable", "; ".join(not_separable) or "none")
    define("ChartXAdjacentBorderline", "; ".join(borderline) or "none")

    # Data labels: each labelled type against the same family without them.
    # The two are different charts, so the interval is unpaired.
    gains, spans_zero = [], 0
    for system in [*VLMS, DEPLOT]:
        for labelled, plain in LABELLED_PAIRS:
            with_labels = list(
                column(scores[system.key], "f1", [labelled]).values())
            without = list(column(scores[system.key], "f1", [plain]).values())
            gains.append((mean(with_labels) - mean(without)) * 100)
            low, high = unpaired_interval(with_labels, without)
            spans_zero += low <= 0 <= high
    require(min(gains) > 0, "every system scores higher with data labels")
    define("ChartXLabelGainMin", f"{min(gains):.1f}")
    define("ChartXLabelGainMax", f"{max(gains):.1f}")
    define("ChartXLabelComparisons", str(len(gains)))
    define("ChartXLabelSpanZero", str(spans_zero))

    # Development split, for the VLMs that were run on it.
    development = chartx_development()
    dev_gaps = [
        (type_balanced_f1(development[s.key])
         - type_balanced_f1(scores[s.key])) * 100
        for s in VLMS if s.key in development
    ]
    require(min(dev_gaps) > 0, "the development split scores higher")
    define_count("ChartXDevModels", len(dev_gaps))
    define("ChartXDevGapMin", f"{min(dev_gaps):.1f}")
    define("ChartXDevGapMax", f"{max(dev_gaps):.1f}")

    # The chart types the paper does not report: DePlot and the two Claude
    # models on the development split, the only runs that cover them.
    other_types = sorted(
        {r["chart_type"] for r in chartx_meta} - set(CHART_TYPES))
    _, dev_deplot = chartx_items(
        CHARTX_RESULTS_DIR / "chartx_deplot.json", chart_types=other_types)
    define("DevOtherN", str(len(dev_deplot)))
    define("DevOtherDePlot",
           f"{mean(v.f1 for v in dev_deplot.values()) * 100:.1f}")
    for key in ("haiku", "sonnet"):
        _, claude = chartx_items(
            CHARTX_RESULTS_DIR / f"chartx_{key}.json", chart_types=other_types)
        interval = paired_interval(
            {k: v.f1 for k, v in claude.items()},
            {k: v.f1 for k, v in dev_deplot.items()})
        require(interval.low > 0 and interval.n >= len(dev_deplot) - 1,
                f"{key} leads DePlot on the unreported chart types")
        define(f"DevOtherN{BY_KEY[key].macro}", str(interval.n))
        define(f"DevOtherDelta{BY_KEY[key].macro}", signed(interval.mean))
        define(f"DevOtherDeltaCI{BY_KEY[key].macro}", interval_text(interval))

    # The unreported run under a moving alias.
    section("ChartX, ministral-3b-latest (moving alias, not tabulated)")
    _, ministral = chartx_items(MINISTRAL_RUN)
    f1 = paired_interval(column(ministral, "f1"), column(deplot, "f1"))
    recall = paired_interval(column(ministral, "recall"),
                             column(deplot, "recall"))
    require(not f1.separable and not recall.separable and f1.mean > 0,
            "ministral-3b-latest is above DePlot but not separably")
    on_same_items = {k: v for k, v in deplot.items() if k in ministral}
    define("MinistralN", str(len(ministral)))
    define("MinistralFone", f"{overall(ministral):.1f}")
    define("MinistralDePlotFone", f"{overall(on_same_items):.1f}")
    define("MinistralDelta", signed(f1.mean))
    define("MinistralDeltaCI", interval_text(f1))
    define("MinistralRecDelta", signed(recall.mean))
    define("MinistralRecDeltaCI", interval_text(recall))
    define_count("MinistralTrailingTypes", sum(
        1 for t in CHART_TYPES
        if type_mean(ministral, t) < type_mean(deplot, t)))

    # --- PlotQA ------------------------------------------------------------
    section("PlotQA")
    plotqa = reported_items()
    deplot_qa = plotqa[DEPLOT.plotqa_key]
    qa_order = plotqa_order(plotqa)
    subset = json.loads((EXTERNAL_DIR / "plotqa_test_1000.json").read_text())
    define("PlotQASubsetN", str(len(subset)))
    define("PlotQAN", str(len(deplot_qa)))
    horizontal = sum(1 for s in deplot_qa.values() if s.horizontal)
    define("PlotQAHorizontalN", str(horizontal))
    define("PlotQAOtherN", str(len(deplot_qa) - horizontal))
    define_count("PlotQAModelCount", len(qa_order))

    above, level, below, wholes = [], [], [], {}
    for key in [*qa_order, DEPLOT.plotqa_key]:
        system = BY_PLOTQA_KEY[key]
        require(set(plotqa[key]) == set(deplot_qa),
                f"{system.name} has the same PlotQA items as DePlot")
        wholes[key] = mean_of(plotqa[key], "whole")
        define(f"PlotQABest{system.macro}",
               f"{mean_of(plotqa[key], 'best'):.1f}")
        define(f"PlotQAHorizontal{system.macro}",
               f"{mean_of(plotqa[key], 'best', True):.1f}")
        define(f"PlotQAOther{system.macro}",
               f"{mean_of(plotqa[key], 'best', False):.1f}")
        if system is DEPLOT:
            continue
        delta = paired_interval(field_of(plotqa[key], "best"),
                                field_of(deplot_qa, "best"))
        define(f"PlotQADelta{system.macro}", signed(delta.mean))
        define(f"PlotQADeltaCI{system.macro}", interval_text(delta))
        text = f"{system.name} ({signed(delta.mean)} {interval_text(delta)})"
        if delta.low > 0:
            above.append(text)
        elif delta.high < 0:
            below.append((system.name, -delta.mean))
        else:
            level.append(text)
    require(len(above) == 1 and len(level) == 1 and len(below) > 1,
            "on PlotQA one VLM is above DePlot, one level, the rest below")
    define("PlotQAAbove", above[0])
    define("PlotQALevel", level[0])
    define_count("PlotQABelowCount", len(below))
    define("PlotQABelowList", join_words(name for name, _ in below))
    define("PlotQABelowMin", f"{min(gap for _, gap in below):.1f}")
    define("PlotQABelowMax", f"{max(gap for _, gap in below):.1f}")
    by_whole = sorted(wholes, key=lambda k: -wholes[k])
    by_best = sorted(wholes, key=lambda k: -mean_of(plotqa[k], "best"))
    require(by_whole == by_best,
            "whole-table scoring orders the PlotQA systems as best-series does")
    require([BY_PLOTQA_KEY[k].macro for k in by_best] == PLOTQA_TABLE_ORDER,
            "Table 3 lists the systems by best-series numeric F1")
    define("PlotQAWholeMin", f"{min(wholes.values()):.0f}")
    define("PlotQAWholeMax", f"{max(wholes.values()):.0f}")
    require(all(
        mean_of(plotqa[k], "best", True) < mean_of(plotqa[k], "best", False)
        for k in plotqa),
        "every system scores lower on the horizontal bar charts")

    # Detailed prompt against simple prompt, the two Claude models.
    section("PlotQA, detailed prompt against simple prompt")
    for key, items in detailed_prompt_items().items():
        system = BY_PLOTQA_KEY[key]
        interval = paired_interval(field_of(items, "best"),
                                   field_of(plotqa[key], "best"))
        require(not interval.separable and interval.n == len(deplot_qa),
                f"the two prompts do not differ separably for {system.name}")
        define(f"PromptDetailed{system.macro}",
               f"{mean_of(items, 'best'):.1f}")
        define(f"PromptDelta{system.macro}", signed(interval.mean))
        define(f"PromptDeltaCI{system.macro}", interval_text(interval))

    # --- Every system at four tolerances -----------------------------------
    section("Sensitivity to the tolerance")
    chartx_above, plotqa_above = {}, {}
    for tolerance in TOLERANCES:
        infix = TOLERANCE_MACRO[tolerance]
        at = chartx_validation(tolerance)
        base = overall(at[DEPLOT.key])
        margins = {}
        for system in [*order, DEPLOT]:
            value = overall(at[system.key])
            define(f"TolChartX{infix}{system.macro}", f"{value:.1f}")
            if system is not DEPLOT:
                margins[system.name] = value - base
        chartx_above[tolerance] = sum(1 for m in margins.values() if m > 0)
        if tolerance == min(TOLERANCES):
            weakest = min(margins, key=lambda name: margins[name])
            require(margins[weakest] < 0,
                    "at the tightest tolerance one VLM is below DePlot")
            define("TolChartXLowestName", weakest)
            define("TolChartXLowestMargin", f"{-margins[weakest]:.1f}")

        at = reported_items(tolerance)
        base = mean_of(at[DEPLOT.plotqa_key], "best")
        ahead = 0
        for key in [*qa_order, DEPLOT.plotqa_key]:
            value = mean_of(at[key], "best")
            define(f"TolPlotQA{infix}{BY_PLOTQA_KEY[key].macro}",
                   f"{value:.1f}")
            ahead += key != DEPLOT.plotqa_key and value > base
        plotqa_above[tolerance] = ahead
    # The sentences these fill are worded for exactly this pattern.
    require(chartx_above[0.01] == len(VLMS) - 1
            and all(chartx_above[t] == len(VLMS) for t in (0.02, 0.05, 0.10)),
            "on ChartX all VLMs but one are above DePlot at 1%, all at the rest")
    require([plotqa_above[t] for t in TOLERANCES] == [0, 1, 2, 2],
            "on PlotQA no, one, two and two VLMs are above DePlot")
    define_count("TolChartXAboveOne", chartx_above[0.01])
    define_count("TolChartXAboveRest", chartx_above[0.02])
    define_count("TolPlotQAAboveOne", plotqa_above[0.01])
    define_count("TolPlotQAAboveTwo", plotqa_above[0.02])
    define_count("TolPlotQAAboveWide", plotqa_above[0.10])

    # Tightening the tolerance costs every VLM more than it costs DePlot.
    loose, tight = reported_items(max(TOLERANCES)), reported_items(
        min(TOLERANCES))
    drops = {
        key: mean_of(loose[key], "best") - mean_of(tight[key], "best")
        for key in loose
    }
    require(all(drops[key] > drops[DEPLOT.plotqa_key] for key in qa_order),
            "on PlotQA every VLM loses more than DePlot as the tolerance tightens")
    define_count("TolPlotQADropDoubled", sum(
        drops[key] > 2 * drops[DEPLOT.plotqa_key] for key in qa_order
    ))

    # The exploratory PubMed Central run. Its ground truth is each figure's
    # companion table, which holds far more cells than the figure plots, so
    # the recall below is not an accuracy; the paper says so where it
    # quotes it.
    section("PubMed Central figure-table pairs (exploratory, confounded)")
    cells = {}
    for model, suffix in (("haiku", "Haiku"), ("sonnet", "Sonnet")):
        paths = sorted((PMC_RESULTS_DIR / model).glob("*.json"))
        metrics = [json.loads(path.read_text())["metrics"] for path in paths]
        define(f"PMCPairs{suffix}", str(len(metrics)))
        define(f"PMCRecall{suffix}", pct(mean(m["recall"] for m in metrics)))
        for path, m in zip(paths, metrics, strict=True):
            cells[path.name] = m["n_total"]
    define("PMCMedianCells", f"{statistics.median(cells.values()):.0f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(out) + "\n")
    defined = sum(1 for line in out if line.startswith("\\newcommand"))
    print(f"Wrote {defined} macros to {args.out}")
    print(f"  ChartX: DePlot {pct(means[DEPLOT.key])}%, "
          f"VLMs {pct(means[lowest.key])}-{pct(means[order[0].key])}%; "
          f"{len(trailing)} models trail in "
          f"{sum(map(len, trailing.values()))} of {len(f1_cells)} cells")
    print(f"  PlotQA: DePlot {mean_of(deplot_qa, 'best'):.1f}%, "
          f"{len(above)} above, {len(below)} below, "
          f"{len(level)} not separable")


if __name__ == "__main__":
    main()
