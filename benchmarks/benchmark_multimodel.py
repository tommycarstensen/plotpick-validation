"""Benchmark chart-to-table extraction across multiple VLM backends.

Tests GPT, Gemini, and Mistral models on ChartX life-science types
for comparison with Claude results. Resume-safe with incremental saves.

Usage:
    python benchmark_multimodel.py --model gpt-5.4-nano
    python benchmark_multimodel.py --model gemini-3.1-flash-lite
    python benchmark_multimodel.py --model mistral-small-2603
"""

import argparse
import functools
import io
import json
import sys
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

if isinstance(sys.stdout, io.TextIOWrapper):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from paths import CHARTX_IMG_ROOT, CHARTX_META, CHARTX_RESULTS_DIR  # noqa: E402
from shared import (  # noqa: E402
    LIFESCI_TYPES,
    call_model,
    compute_recall,
    encode_image,
    extract_numbers,
    load_checkpoint,
    parse_chartx_csv,
    pinned_model,
    rate_limit_delay,
    require_chartx_images,
    retry_on_rate_limit,
    save_checkpoint,
)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model", required=True, type=pinned_model,
                        help="Model name (e.g. gpt-5.4-nano)")
    parser.add_argument("--n", type=int, default=100,
                        help="Figures per chart type")
    args = parser.parse_args()
    require_chartx_images()

    safe_name = args.model.replace(".", "_").replace("-", "_")

    meta = json.loads(CHARTX_META.read_text("utf-8"))
    by_type = defaultdict(list)
    for row in meta:
        if row["chart_type"] in LIFESCI_TYPES:
            by_type[row["chart_type"]].append(row)

    out_path = CHARTX_RESULTS_DIR / f"chartx_{safe_name}.json"
    results, done = load_checkpoint(out_path)
    results = defaultdict(list, results)
    total = len(done)

    for chart_type, rows in sorted(by_type.items()):
        for row in rows[:args.n]:
            if row["imgname"] in done:
                continue

            img_path = (CHARTX_IMG_ROOT / row["chart_type"]
                        / "png" / f"{row['imgname']}.png")
            if not img_path.exists():
                continue

            gt_text = parse_chartx_csv(row["csv"])
            gt_nums = extract_numbers(gt_text)
            if not gt_nums:
                continue

            img, b64 = encode_image(img_path)

            try:
                extracted = retry_on_rate_limit(
                    functools.partial(
                        call_model, args.model, args.model, b64=b64, img=img)
                )
                ex_nums = extract_numbers(extracted)
                recall = compute_recall(ex_nums, gt_nums)
            except Exception as e:  # noqa: BLE001
                print(f"  [error] {row['imgname']}: {e}")
                recall, ex_nums, extracted = 0.0, set(), ""

            results[chart_type].append({
                "imgname": row["imgname"],
                "chart_type": chart_type,
                "recall": recall,
                "gt_csv": row["csv"],
                "extracted_text": extracted if isinstance(extracted, str) else "",
                "gt_nums": sorted(gt_nums),
                "ex_nums": sorted(ex_nums),
            })
            total += 1
            print(f"  [{total:3d}] {chart_type:20s} {row['imgname']:15s} "
                  f"recall={recall:.0%}")

            save_checkpoint(
                out_path, results,
                model=args.model, n_per_type=args.n,
            )
            rate_limit_delay(args.model)

    # Summary
    all_recalls = []
    print(f"\n{'=' * 50}")
    print(f"Model: {args.model}")
    for ct in sorted(results):
        vals = [r["recall"] for r in results[ct]]
        all_recalls.extend(vals)
        print(f"  {ct:20s}: n={len(vals):3d}  "
              f"recall={sum(vals) / len(vals):.1%}")
    if all_recalls:
        print(f"  {'OVERALL':20s}: n={len(all_recalls):3d}  "
              f"recall={sum(all_recalls) / len(all_recalls):.1%}")


if __name__ == "__main__":
    main()
