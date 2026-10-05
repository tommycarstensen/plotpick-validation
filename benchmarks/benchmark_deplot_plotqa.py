"""Run DePlot on PlotQA test set. Incremental saves, resume-safe.

The score printed during a run and stored as ``rmsf1`` is the whole-table
numeric F1 against the one annotated series. The scores the paper reports
come from score_plotqa.py, which rescores the stored replies.

Usage:
    python benchmark_deplot_plotqa.py          # all 1000
    python benchmark_deplot_plotqa.py --n 100  # first 100
"""

import argparse
import io
import json
import sys
from pathlib import Path

if isinstance(sys.stdout, io.TextIOWrapper):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from paths import PLOTQA_DIR, PLOTQA_RESULTS_DIR, EXTERNAL_DIR  # noqa: E402
from benchmarks.shared import NUM_RE, compute_rmsf1, plotqa_truth  # noqa: E402


def extract_numbers(text):
    """Extract all numeric values from a string (as list, preserving dupes)."""
    return [
        float(m.group().replace(",", ""))
        for m in NUM_RE.finditer(str(text))
    ]


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--n", type=int, default=1000)
    args = parser.parse_args()

    meta_path = EXTERNAL_DIR / "plotqa_test_1000.json"
    meta = json.loads(meta_path.read_text("utf-8"))[:args.n]
    img_dir = PLOTQA_DIR / "images"

    PLOTQA_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PLOTQA_RESULTS_DIR / "plotqa_deplot.json"

    # Resume
    done = set()
    results = []
    if out_path.exists():
        existing = json.loads(out_path.read_text("utf-8"))
        results = existing.get("results", [])
        done = {r["imgname"] for r in results}
    if done:
        print(f"Resuming: {len(done)} already done")

    print("Loading DePlot model...")
    import torch
    torch.set_num_threads(4)
    from PIL import Image
    from transformers import Pix2StructProcessor, Pix2StructForConditionalGeneration
    processor = Pix2StructProcessor.from_pretrained("google/deplot")
    model = Pix2StructForConditionalGeneration.from_pretrained("google/deplot")
    print("Model loaded.")

    for row in meta:
        if row["imgname"] in done:
            continue

        img_path = img_dir / f"{row['imgname']}.png"
        if not img_path.exists():
            continue

        gt_nums = plotqa_truth(row)
        if not gt_nums:
            continue

        try:
            img = Image.open(img_path).convert("RGB")
            # transformers' type hints wrongly reject both of these calls.
            inputs = processor(
                images=img,
                text="Generate underlying data table of the figure below:",
                return_tensors="pt",  # pyright: ignore[reportCallIssue]
            )
            preds = model.generate(  # pyright: ignore[reportAttributeAccessIssue]
                **inputs, max_new_tokens=512
            )
            extracted = processor.decode(preds[0], skip_special_tokens=True)
            ex_nums = extract_numbers(extracted)
            f1 = compute_rmsf1(ex_nums, gt_nums)
        except Exception as e:  # noqa: BLE001
            print(f"  [error] {row['imgname']}: {e}")
            f1 = 0.0
            ex_nums = []
            extracted = ""

        results.append({
            "imgname": row["imgname"],
            "rmsf1": f1,
            "gt_nums": gt_nums,
            "ex_nums": ex_nums,
            "extracted_text": extracted,
        })

        total = len(results)
        print(
            f"  [{total:3d}] {row['imgname']}  RMSF1={f1:.0%}  "
            f"({len(ex_nums)}/{len(gt_nums)} values)"
        )

        # Save after every figure
        out_data = {
            "model": "deplot",
            "n": total,
            "mean_rmsf1": sum(r["rmsf1"] for r in results) / total,
            "results": results,
        }
        out_path.write_text(
            json.dumps(out_data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    total = len(results)
    mean = sum(r["rmsf1"] for r in results) / total if total else 0
    print(f"\nDone: {total} figures, mean RMSF1 = {mean:.1%}")


if __name__ == "__main__":
    main()
