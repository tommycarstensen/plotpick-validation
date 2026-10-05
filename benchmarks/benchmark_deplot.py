"""Benchmark DePlot on ChartX dataset for comparison with Claude models.

Downloads the DePlot model from HuggingFace and runs chart-to-table
extraction on the same ChartX figures used in benchmark_chartx.py.
Reports RMSF1 per chart type.

Usage:
    python benchmark_deplot.py                    # test split, 10 per type
    python benchmark_deplot.py --n 5              # test split, 5 per type
    python benchmark_deplot.py --split val --n 50 # validation split

The VLM numbers reported in the paper come from the validation split, so
``--split val --n 50`` is the run that makes the DePlot comparison
like-for-like on identical items. The default stays on the test split,
which the paper uses as its development split.

The stored chartx_deplot*.json files were written before DePlot's title row
was left out of the scoring, so their ``ex_nums`` and ``rmsf1`` fields
include the title's numbers; a rerun would not. The paper's scores are
rescored from ``extracted_text`` by paired_bootstrap.py.
"""

import argparse
import contextlib
import io
import json
import sys
from collections import defaultdict
from pathlib import Path

from PIL import Image

try:
    import torch  # noqa: F401  (imported for the availability check only)
    import torchvision  # noqa: F401
    from transformers import (
        Pix2StructForConditionalGeneration,
        Pix2StructProcessor,
    )
except ImportError as exc:  # pragma: no cover - environment guard
    sys.exit(
        f"DePlot needs torch, torchvision and transformers, and this "
        f"interpreter ({sys.executable}) is missing at least one "
        f"({exc.name}).\n"
        "transformers selects a torchvision-backed image processor for "
        "Pix2Struct, so torchvision is required even though no other "
        "script here uses it.\n"
        "Install them for THIS interpreter with:\n"
        f"    {sys.executable} -m pip install -r requirements.txt\n"
        "A machine often has several Pythons and only some carry the "
        "stack, so check a candidate before using it:\n"
        "    <python> -c 'import torch, torchvision, transformers'"
    )

if isinstance(sys.stdout, io.TextIOWrapper):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from paths import (  # noqa: E402
    CHARTX_IMG_ROOT,
    CHARTX_META,
    CHARTX_RESULTS_DIR,
    CHARTX_VAL_META,
)
from benchmarks.shared import (  # noqa: E402
    DEPLOT_INSTRUCTION,
    DEPLOT_MAX_NEW_TOKENS,
    NUM_RE,
    compute_rmsf1,
    deplot_rows,
    parse_chartx_csv,
    require_chartx_images,
)

MODEL_NAME = "google/deplot"


def extract_numbers_from_rows(rows):
    """Extract all numeric values from a list of row lists (as a list)."""
    nums = []
    for row in rows:
        for cell in row:
            for m in NUM_RE.finditer(cell):
                with contextlib.suppress(ValueError):
                    nums.append(float(m.group().replace(",", "")))
    return nums


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument(
        "--split",
        choices=("test", "val"),
        default="test",
        help=(
            "Which ChartX split to score. The VLM results reported in the "
            "paper use the held-out validation split, so --split val is "
            "what makes the DePlot comparison like-for-like; the default "
            "stays 'test' so existing result files reproduce."
        ),
    )
    args = parser.parse_args()
    require_chartx_images()

    meta_path = CHARTX_META if args.split == "test" else CHARTX_VAL_META
    out_name = (
        "chartx_deplot.json"
        if args.split == "test"
        else "chartx_deplot_val.json"
    )

    print("Loading DePlot model...")
    processor = Pix2StructProcessor.from_pretrained(MODEL_NAME)
    model = Pix2StructForConditionalGeneration.from_pretrained(MODEL_NAME)
    print("Model loaded.")

    meta = json.loads(meta_path.read_text("utf-8"))
    by_type = defaultdict(list)
    for row in meta:
        by_type[row["chart_type"]].append(row)

    results = defaultdict(list)
    total = 0

    for chart_type, rows in sorted(by_type.items()):
        for row in rows[:args.n]:
            img_path = (
                CHARTX_IMG_ROOT / row["chart_type"] / "png"
                / f"{row['imgname']}.png"
            )
            if not img_path.exists():
                continue

            gt_text = parse_chartx_csv(row["csv"])
            gt_rows = [
                line.split("\t")
                for line in gt_text.strip().split("\n") if line.strip()
            ]
            gt_nums = extract_numbers_from_rows(gt_rows)
            if not gt_nums:
                continue

            try:
                img = Image.open(img_path).convert("RGB")
                # transformers' type hints wrongly reject both of these calls.
                inputs = processor(
                    images=img,
                    text=DEPLOT_INSTRUCTION,
                    return_tensors="pt",  # pyright: ignore[reportCallIssue]
                )
                preds = model.generate(  # pyright: ignore[reportAttributeAccessIssue]
                    **inputs, max_new_tokens=DEPLOT_MAX_NEW_TOKENS
                )
                extracted_text = processor.decode(
                    preds[0], skip_special_tokens=True,
                )
                # Without the TITLE row: shared.deplot_rows says why.
                ex_rows = deplot_rows(extracted_text)
                ex_nums = extract_numbers_from_rows(ex_rows)
                f1 = compute_rmsf1(ex_nums, gt_nums)
            except Exception as e:  # noqa: BLE001
                print(f"  [error] {row['imgname']}: {e}")
                f1 = 0.0
                ex_nums = []
                extracted_text = ""

            results[chart_type].append({
                "imgname": row["imgname"],
                "chart_type": chart_type,
                "rmsf1": f1,
                "gt_csv": row["csv"],
                "extracted_text": extracted_text,
                "gt_nums": gt_nums,
                "ex_nums": ex_nums,
                "title": row.get("title", ""),
                "img_rel": f"{row['chart_type']}/png/{row['imgname']}.png",
            })
            total += 1
            print(
                f"  [{total:3d}] {chart_type:20s} {row['imgname']:15s} "
                f"RMSF1={f1:.0%}  ({len(ex_nums)}/{len(gt_nums)} values)"
            )

    # Summary
    print(f"\n{'=' * 60}")
    print(f"Model: DePlot ({MODEL_NAME})")
    print(f"Figures per type: {args.n}")
    print(f"Total evaluated: {total}")
    print(f"\n{'Type':25s} {'N':>3s} {'Mean RMSF1':>11s}")
    print("-" * 42)

    all_f1 = []
    for chart_type in sorted(results.keys()):
        f1s = [r["rmsf1"] for r in results[chart_type]]
        mean_f1 = sum(f1s) / len(f1s) if f1s else 0
        all_f1.extend(f1s)
        print(f"  {chart_type:23s} {len(f1s):3d} {mean_f1:10.1%}")

    if all_f1:
        overall = sum(all_f1) / len(all_f1)
        print("-" * 42)
        print(f"  {'OVERALL':23s} {len(all_f1):3d} {overall:10.1%}")

    # Save
    CHARTX_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = CHARTX_RESULTS_DIR / out_name
    out_data = {
        "model": "deplot",
        "model_id": MODEL_NAME,
        "n_per_type": args.n,
        "split": args.split,
        "results": dict(results),
    }
    out_path.write_text(
        json.dumps(out_data, indent=2, ensure_ascii=False), encoding="utf-8",
    )
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
