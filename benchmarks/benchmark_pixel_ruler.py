"""Benchmark pixel-ruler extraction on ChartX dataset.

Runs pixel_ruler.extract_values on chart images and compares against
ground truth. Reports recall per chart type, comparable to LLM benchmarks.

Usage:
    python benchmark_pixel_ruler.py             # all supported types
    python benchmark_pixel_ruler.py --n 10      # first 10 per type
    python benchmark_pixel_ruler.py --type box  # only box plots
"""

import argparse
import contextlib
import io
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

if isinstance(sys.stdout, io.TextIOWrapper):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from paths import CHARTX_IMG_ROOT, CHARTX_META, CHARTX_RESULTS_DIR  # noqa: E402
from benchmarks.pixel_ruler import extract_values, find_best_scale  # noqa: E402
from benchmarks.shared import (  # noqa: E402
    NUM_RE,
    compute_recall,
    parse_chartx_csv,
    require_chartx_images,
)

SUPPORTED_TYPES = ["histogram", "bar_chart", "bar_chart_num", "box"]


def extract_numbers(rows):
    """Extract all numeric values from a parsed TSV table."""
    nums = set()
    for row in rows:
        for cell in row:
            for m in NUM_RE.finditer(cell):
                with contextlib.suppress(ValueError):
                    nums.add(float(m.group().replace(",", "")))
    return nums


def parse_gt_rows(csv_str):
    """Parse ChartX ground-truth CSV field into row lists."""
    text = parse_chartx_csv(csv_str)
    rows = []
    for line in text.strip().split("\n"):
        line = line.strip()
        if line:
            rows.append([c.strip() for c in line.split("\t")])
    return rows


def detect_stacked(row):
    """Check if bar chart uses stacking (bottom= in redrawing code)."""
    code = row.get("redrawing", {}).get("output", "")
    return "bottom=" in code or "bottom =" in code


def detect_orientation(row):
    """Detect if chart uses horizontal bars (barh)."""
    code = row.get("redrawing", {}).get("output", "")
    return "barh" in code


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--n", type=int, default=0,
        help="Max figures per chart type (0=all)",
    )
    parser.add_argument("--type", default="", help="Only run this chart type")
    args = parser.parse_args()
    require_chartx_images()

    meta = json.loads(CHARTX_META.read_text("utf-8"))

    by_type = defaultdict(list)
    for row in meta:
        by_type[row["chart_type"]].append(row)

    target_types = [args.type] if args.type else SUPPORTED_TYPES

    results = defaultdict(list)
    summary = {}

    for chart_type in target_types:
        if chart_type not in by_type:
            print(f"No charts of type '{chart_type}' in test set")
            continue

        charts = by_type[chart_type]
        if args.n > 0:
            charts = charts[:args.n]

        recalls = []
        processed = 0
        errors = 0

        print(f"\n{'=' * 60}")
        print(f"  {chart_type} ({len(charts)} charts)")
        print(f"{'=' * 60}")

        for i, row in enumerate(charts):
            imgname = row["imgname"]
            img_path = (
                CHARTX_IMG_ROOT / row["chart_type"] / "png"
                / f"{imgname}.png"
            )
            if not img_path.exists():
                continue

            gt_rows = parse_gt_rows(row["csv"])
            gt_nums = extract_numbers(gt_rows)
            if not gt_nums:
                continue

            stacked = detect_stacked(row)
            is_hbar = detect_orientation(row)

            effective_type = chart_type
            if chart_type == "histogram" and is_hbar:
                effective_type = "histogram_hbar"

            try:
                ex_vals = extract_values(
                    str(img_path),
                    chart_type=effective_type,
                    stacked_hint=stacked,
                )
            except Exception as e:  # noqa: BLE001
                errors += 1
                if i < 5:
                    print(f"  {imgname}: ERROR {e}")
                continue

            if not ex_vals:
                recalls.append(0.0)
                processed += 1
                if i < 20 or i % 50 == 0:
                    print(f"  {imgname}: no values extracted")
                continue

            k = find_best_scale(ex_vals, gt_nums)
            scaled = {v * k for v in ex_vals}

            recall = compute_recall(scaled, gt_nums)
            recalls.append(recall)
            processed += 1

            results[chart_type].append({
                "imgname": imgname,
                "chart_type": chart_type,
                "recall": recall,
                "scale_factor": k,
                "n_gt": len(gt_nums),
                "n_extracted": len(ex_vals),
                "gt_nums": sorted(gt_nums),
                "ex_nums_raw": sorted(ex_vals),
                "ex_nums_scaled": sorted(scaled),
            })

            if i < 20 or i % 50 == 0 or recall < 0.5:
                print(
                    f"  {imgname}: recall={recall:.0%} "
                    f"({len(ex_vals)} vals, k={k:.2f})"
                    f"{'  <-- LOW' if recall < 0.5 else ''}"
                )

        if recalls:
            avg = np.mean(recalls)
            med = np.median(recalls)
            perfect = sum(1 for r in recalls if r >= 1.0)
            summary[chart_type] = {
                "n": processed,
                "errors": errors,
                "avg_recall": float(avg),
                "median_recall": float(med),
                "perfect": perfect,
            }
            print(
                f"\n  {chart_type} summary: "
                f"n={processed}, avg={avg:.1%}, median={med:.1%}, "
                f"perfect={perfect}/{processed}, errors={errors}"
            )

    # Save results
    CHARTX_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = CHARTX_RESULTS_DIR / "chartx_pixel_ruler.json"
    out_data = {
        "summary": summary,
        "results": dict(results),
    }
    out_path.write_text(
        json.dumps(out_data, indent=2, ensure_ascii=False), encoding="utf-8",
    )
    print(f"\nSaved to {out_path}")

    # Overall summary
    print(f"\n{'=' * 60}")
    print("  OVERALL SUMMARY")
    print(f"{'=' * 60}")
    for ct, s in summary.items():
        print(
            f"  {ct:<20} avg={s['avg_recall']:.1%}  "
            f"median={s['median_recall']:.1%}  "
            f"n={s['n']}  perfect={s['perfect']}"
        )


if __name__ == "__main__":
    main()
