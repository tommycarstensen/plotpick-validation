"""Benchmark Claude chart-to-table extraction on PlotQA with the detailed prompt.

The metric is this repository's numeric F1 (shared.compute_numeric_f1). It is
not the RMS F1 that the DePlot and TinyChart papers report, and its scores
must not be compared with theirs.

The score printed during a run and stored as ``rmsf1`` is the whole-table
numeric F1 against the one annotated series. The scores the paper reports
come from score_plotqa.py, which rescores the stored replies.

Usage:
    python benchmark_plotqa.py              # 100 figures, haiku
    python benchmark_plotqa.py --n 50       # 50 figures
    python benchmark_plotqa.py --model sonnet
"""

import argparse
import contextlib
import io
import json
import sys
import time
from pathlib import Path

if isinstance(sys.stdout, io.TextIOWrapper):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from paths import PLOTQA_DIR, PLOTQA_RESULTS_DIR, EXTERNAL_DIR  # noqa: E402
from benchmarks.shared import (  # noqa: E402
    NUM_RE,
    EXTRACT_PROMPT_DETAILED,
    compute_rmsf1,
    encode_image,
    load_anthropic_key,
    call_claude,
    plotqa_truth,
)

MODELS = {
    "haiku": "claude-haiku-4-5-20251001",
    "sonnet": "claude-sonnet-4-6",
}


def extract_numbers(text):
    """Extract all numeric values from extracted text."""
    nums = []
    for m in NUM_RE.finditer(text):
        with contextlib.suppress(ValueError):
            nums.append(float(m.group().replace(",", "")))
    return nums


def _save(model_name, model_id, results, all_f1):
    """Write incremental checkpoint."""
    PLOTQA_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PLOTQA_RESULTS_DIR / f"plotqa_{model_name}.json"
    out_data = {
        "model": model_name,
        "model_id": model_id,
        "n": len(results),
        "mean_rmsf1": sum(all_f1) / len(all_f1) if all_f1 else 0,
        "results": results,
    }
    out_path.write_text(
        json.dumps(out_data, indent=2, ensure_ascii=False), encoding="utf-8",
    )


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--n", type=int, default=100,
        help="Number of figures to evaluate (default: 100)",
    )
    parser.add_argument("--model", default="haiku", choices=list(MODELS.keys()))
    args = parser.parse_args()

    meta_path = EXTERNAL_DIR / "plotqa_test_100.json"
    meta = json.loads(meta_path.read_text("utf-8"))[:args.n]

    # Load API key once (used implicitly by call_claude)
    load_anthropic_key()
    model_id = MODELS[args.model]
    img_dir = PLOTQA_DIR / "images"

    results = []
    all_f1 = []

    for i, row in enumerate(meta):
        img_path = img_dir / f"{row['imgname']}.png"
        if not img_path.exists():
            print(f"  [skip] {img_path}")
            continue

        gt_nums = plotqa_truth(row)
        if not gt_nums:
            continue

        _, b64 = encode_image(img_path)

        try:
            extracted_text = call_claude(
                model_id, b64, prompt=EXTRACT_PROMPT_DETAILED,
            )
            ex_nums = extract_numbers(extracted_text)
            f1 = compute_rmsf1(ex_nums, gt_nums)
        except Exception as e:  # noqa: BLE001
            print(f"  [error] {row['imgname']}: {e}")
            f1 = 0.0
            ex_nums = []
            extracted_text = ""

        results.append({
            "imgname": row["imgname"],
            "rmsf1": f1,
            "gt_nums": gt_nums,
            "ex_nums": ex_nums,
            "extracted_text": extracted_text,
            "series_name": row.get("series_name", ""),
        })
        all_f1.append(f1)
        print(
            f"  [{i + 1:3d}/{len(meta)}] {row['imgname']}  "
            f"RMSF1={f1:.0%}  ({len(ex_nums)}/{len(gt_nums)} values)"
        )

        if (i + 1) % 20 == 0:
            _save(args.model, model_id, results, all_f1)

        time.sleep(0.3)

    _save(args.model, model_id, results, all_f1)

    # Summary
    print(f"\n{'=' * 60}")
    print(f"Model: {args.model} ({model_id})")
    print(f"Figures evaluated: {len(all_f1)}")
    if all_f1:
        print(f"Mean RMSF1: {sum(all_f1) / len(all_f1):.1%}")
        print(f"Median RMSF1: {sorted(all_f1)[len(all_f1) // 2]:.1%}")


if __name__ == "__main__":
    main()
