"""Benchmark Claude chart-to-table extraction on ChartX test set.

Sends chart images to Claude and compares extracted values against
the ground-truth CSV. Reports recall per chart type. Resume-safe
with incremental saves.

Usage:
    python benchmark_chartx.py                  # 5 per type, haiku
    python benchmark_chartx.py --n 20           # 20 per type
    python benchmark_chartx.py --model sonnet   # use sonnet
    python benchmark_chartx.py --model sonnet-5-5 --n 100 --lifesci
"""

import argparse
import io
import json
import sys
from collections import defaultdict
from pathlib import Path

import anthropic

if isinstance(sys.stdout, io.TextIOWrapper):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from paths import CHARTX_IMG_ROOT, CHARTX_META, CHARTX_RESULTS_DIR  # noqa: E402
from shared import (  # noqa: E402
    CLAUDE_MAX_TOKENS,
    EXTRACT_PROMPT_DETAILED,
    LIFESCI_TYPES,
    claude_reply_text,
    compute_mape,
    compute_recall,
    encode_image,
    extract_numbers,
    extract_numbers_from_rows,
    load_anthropic_key,
    load_checkpoint,
    parse_chartx_csv,
    parse_tsv,
    rate_limit_delay,
    require_chartx_images,
    save_checkpoint,
)

CLAUDE_MODELS = {
    "haiku": "claude-haiku-4-5-20251001",
    "sonnet": "claude-sonnet-4-6",
    "opus": "claude-opus-4-6",
    "sonnet-5-5": "claude-sonnet-5-5",
    "opus-5-5": "claude-opus-5-5",
}


def extract_from_image(client, img_path, model):
    """Send a chart image to Claude and return extracted text."""
    _, b64 = encode_image(img_path)
    resp = client.messages.create(
        model=model, max_tokens=CLAUDE_MAX_TOKENS,
        messages=[{"role": "user", "content": [
            {"type": "image", "source": {
                "type": "base64", "media_type": "image/png", "data": b64}},
            {"type": "text", "text": EXTRACT_PROMPT_DETAILED},
        ]}],
    )
    return claude_reply_text(resp)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--n", type=int, default=5,
                        help="Figures per chart type (default: 5)")
    parser.add_argument("--model", default="haiku",
                        choices=list(CLAUDE_MODELS))
    parser.add_argument("--lifesci", action="store_true",
                        help="Only the six chart types the paper reports "
                             "(CHART_TYPES in systems.py)")
    args = parser.parse_args()
    require_chartx_images()

    model_id = CLAUDE_MODELS[args.model]
    client = anthropic.Anthropic(api_key=load_anthropic_key())

    meta = json.loads(CHARTX_META.read_text("utf-8"))
    by_type = defaultdict(list)
    for row in meta:
        if args.lifesci and row["chart_type"] not in LIFESCI_TYPES:
            continue
        by_type[row["chart_type"]].append(row)

    safe_name = args.model.replace("-", "_")
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
                print(f"  [skip] {img_path} not found")
                continue

            gt_text = parse_chartx_csv(row["csv"])
            gt_nums = extract_numbers(gt_text)
            if not gt_nums:
                continue

            error = None
            try:
                extracted_text = extract_from_image(client, img_path, model_id)
                ex_rows = parse_tsv(extracted_text)
                ex_nums = extract_numbers_from_rows(ex_rows)
                recall = compute_recall(ex_nums, gt_nums)
                mape = compute_mape(ex_nums, gt_nums)
            except Exception as e:  # noqa: BLE001
                print(f"  [error] {row['imgname']}: {e}")
                recall, mape, ex_nums, extracted_text = 0.0, 1.0, set(), ""
                error = str(e)

            entry = {
                "imgname": row["imgname"],
                "chart_type": chart_type,
                "recall": recall,
                "mape": mape,
                "gt_csv": row["csv"],
                "extracted_text": extracted_text,
                "gt_nums": sorted(gt_nums),
                "ex_nums": sorted(ex_nums),
                "title": row.get("title", ""),
                "img_rel": f"{row['chart_type']}/png/{row['imgname']}.png",
            }
            if error:
                entry["error"] = error
            results[chart_type].append(entry)
            total += 1
            print(f"  [{total:3d}] {chart_type:20s} {row['imgname']:15s} "
                  f"recall={recall:.0%}  ({len(ex_nums)}/{len(gt_nums)} values)")

            save_checkpoint(
                out_path, results,
                model=args.model, model_id=model_id, n_per_type=args.n,
            )
            rate_limit_delay(args.model)

    # Summary
    print(f"\n{'=' * 60}")
    print(f"Model: {args.model} ({model_id}), {args.n} per type")
    print(f"{'Type':25s} {'N':>3s} {'Mean Recall':>12s} {'Median':>8s}")
    print("-" * 52)

    all_recalls = []
    for chart_type in sorted(results):
        recalls = [r["recall"] for r in results[chart_type]]
        mean_r = sum(recalls) / len(recalls) if recalls else 0
        sorted_r = sorted(recalls)
        median_r = sorted_r[len(sorted_r) // 2] if sorted_r else 0
        all_recalls.extend(recalls)
        print(f"  {chart_type:23s} {len(recalls):3d} "
              f"{mean_r:11.1%} {median_r:7.1%}")

    if all_recalls:
        overall = sum(all_recalls) / len(all_recalls)
        print("-" * 52)
        print(f"  {'OVERALL':23s} {len(all_recalls):3d} {overall:11.1%}")
        n_err = sum("error" in r for ct in results for r in results[ct])
        if n_err:
            print(f"  {n_err} figure(s) scored 0 after an API error or refusal"
                  " -- see the 'error' field")


if __name__ == "__main__":
    main()
