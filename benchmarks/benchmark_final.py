"""Final benchmark on ChartX VALIDATION split (held out from development).

Tests all model families (Claude, GPT, Gemini, Mistral) on the
validation split with 95% bootstrap confidence intervals.

Usage:
    python benchmark_final.py --model haiku --n 50
    python benchmark_final.py --model sonnet-5-5 --n 50
    python benchmark_final.py --model gpt-5.4-nano --n 50
    python benchmark_final.py --model gemini-3.1-flash-lite-preview --n 50
    python benchmark_final.py --model mistral-small-2603 --n 50
"""

import argparse
import functools
import io
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

load_dotenv()

if isinstance(sys.stdout, io.TextIOWrapper):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from paths import CHARTX_IMG_ROOT, CHARTX_VAL_META, FINAL_VAL_RESULTS_DIR  # noqa: E402
from shared import (  # noqa: E402
    LIFESCI_TYPES,
    bootstrap_ci,
    call_model,
    compute_recall,
    encode_image,
    extract_numbers,
    load_checkpoint,
    parse_chartx_csv,
    pinned_model,
    rate_limit_delay,
    require_api_key,
    require_chartx_images,
    retry_on_rate_limit,
    save_checkpoint,
)

CLAUDE_MODELS = {
    "haiku": "claude-haiku-4-5-20251001",
    "sonnet": "claude-sonnet-4-6",
    "sonnet-5-5": "claude-sonnet-5-5",
    "opus-5-5": "claude-opus-5-5",
}


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model", required=True, type=pinned_model)
    parser.add_argument("--n", type=int, default=50,
                        help="Figures per type")
    args = parser.parse_args()
    require_chartx_images()
    require_api_key(args.model)

    model_id = CLAUDE_MODELS.get(args.model, args.model)
    safe_name = args.model.replace(".", "_").replace("-", "_")

    meta = json.loads(CHARTX_VAL_META.read_text("utf-8"))
    by_type = defaultdict(list)
    for row in meta:
        if row["chart_type"] in LIFESCI_TYPES:
            by_type[row["chart_type"]].append(row)

    out_path = FINAL_VAL_RESULTS_DIR / f"final_val_{safe_name}.json"
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

            error = None
            try:
                extracted = retry_on_rate_limit(
                    functools.partial(
                        call_model, args.model, model_id, b64=b64, img=img)
                )
                ex_nums = extract_numbers(extracted)
                recall = compute_recall(ex_nums, gt_nums)
            except Exception as e:  # noqa: BLE001
                print(f"  [error] {row['imgname']}: {e}")
                recall, ex_nums, error = 0.0, set(), str(e)

            entry = {
                "imgname": row["imgname"],
                "chart_type": chart_type,
                "recall": recall,
                "gt_nums": sorted(gt_nums),
                "ex_nums": sorted(ex_nums),
            }
            if error:
                entry["error"] = error
            results[chart_type].append(entry)
            total += 1
            print(f"  [{total:3d}] {chart_type:20s} {row['imgname']:15s} "
                  f"recall={recall:.0%}")

            save_checkpoint(
                out_path, results,
                model=args.model, model_id=model_id,
                split="validation", n_per_type=args.n,
            )
            rate_limit_delay(args.model)

    # Summary with CI
    print(f"\n{'=' * 60}")
    print(f"Model: {args.model} (validation split)")
    print(f"{'Type':25s} {'N':>3s} {'Recall':>8s} {'95% CI':>15s}")
    print("-" * 55)

    all_vals = []
    for ct in sorted(results):
        vals = [r["recall"] for r in results[ct]]
        all_vals.extend(vals)
        mean = np.mean(vals)
        lo, hi = bootstrap_ci(vals)
        print(f"  {ct:23s} {len(vals):3d} "
              f"{mean:7.1%} [{lo:.1%}, {hi:.1%}]")

    if all_vals:
        mean = np.mean(all_vals)
        lo, hi = bootstrap_ci(all_vals)
        print("-" * 55)
        print(f"  {'OVERALL':23s} {len(all_vals):3d} "
              f"{mean:7.1%} [{lo:.1%}, {hi:.1%}]")
        n_err = sum("error" in r for ct in results for r in results[ct])
        if n_err:
            print(f"  {n_err} figure(s) scored 0 after an API error or refusal"
                  " -- see the 'error' field")
        # A finished run is hours of API calls living in one untracked file.
        # One was lost to a stray rm before it was ever committed.
        print(f"\nResults: {out_path}")
        print("Commit them now, before anything else touches the file:")
        print(f"    git add {os.path.relpath(out_path)} && "
              f"git commit -m 'Benchmark {args.model} on ChartX validation' "
              f"-- {os.path.relpath(out_path)}")


if __name__ == "__main__":
    main()
