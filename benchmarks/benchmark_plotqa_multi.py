"""Run PlotQA chart-to-table extraction across all VLM backends.

Supports GPT, Gemini, and Mistral models. Resume-safe with incremental
saves after every figure.

The score printed during a run and stored as ``rmsf1`` is the whole-table
numeric F1 against the one annotated series. The scores the paper reports
come from score_plotqa.py, which rescores the stored replies.

Usage:
    python benchmark_plotqa_multi.py --model gemini-3.1-flash-lite-preview
    python benchmark_plotqa_multi.py --model gpt-5.4-nano
    python benchmark_plotqa_multi.py --model mistral-small-2603
"""

import argparse
import functools
import io
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

if isinstance(sys.stdout, io.TextIOWrapper):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

load_dotenv()

from dotenv import load_dotenv  # noqa: E402
from paths import PLOTQA_DIR, PLOTQA_RESULTS_DIR, EXTERNAL_DIR  # noqa: E402

# benchmark_final.py does this too; without it a key in .env is ignored
# and every call fails, scoring every figure 0.
load_dotenv()
from benchmarks.shared import (  # noqa: E402
    EXTRACT_PROMPT,
    NUM_RE,
    call_model,
    compute_rmsf1,
    encode_image,
    pinned_model,
    plotqa_truth,
    rate_limit_delay,
    require_api_key,
    retry_on_rate_limit,
)


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
    parser.add_argument("--model", required=True, type=pinned_model)
    parser.add_argument("--n", type=int, default=1000)
    args = parser.parse_args()
    require_api_key(args.model)

    safe_name = args.model.replace(".", "_").replace("-", "_")
    meta = json.loads(
        (EXTERNAL_DIR / "plotqa_test_1000.json").read_text("utf-8"),
    )[:args.n]
    img_dir = PLOTQA_DIR / "images"
    if not img_dir.is_dir():
        sys.exit(
            f"PlotQA images not found at {img_dir}.\n"
            "Every figure would be skipped and the run would report\n"
            "'0 figures, mean RMSF1 = 0.0%' as if it were a result.\n"
            "Fetch the PlotQA images from "
            "https://github.com/NiteshMethani/PlotQA first."
        )

    # Resume
    PLOTQA_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PLOTQA_RESULTS_DIR / f"plotqa_{safe_name}.json"
    results = []
    done = set()
    if out_path.exists():
        existing = json.loads(out_path.read_text("utf-8"))
        results = existing.get("results", [])
        done = {r["imgname"] for r in results}
    if done:
        print(f"Resuming: {len(done)} already done")

    # Determine model family for call_model dispatch
    model_name = args.model
    if model_name.startswith("gpt"):
        family = "gpt"
    elif model_name.startswith("gemini"):
        family = "gemini"
    elif model_name.startswith("mistral") or model_name.startswith("ministral"):
        family = "mistral"
    else:
        family = model_name

    for row in meta:
        if row["imgname"] in done:
            continue
        img_path = img_dir / f"{row['imgname']}.png"
        if not img_path.exists():
            continue

        gt_nums = plotqa_truth(row)
        if not gt_nums:
            continue

        img, b64 = encode_image(img_path)

        try:
            extracted = retry_on_rate_limit(
                functools.partial(
                    call_model, family, model_name,
                    b64=b64, img=img, prompt=EXTRACT_PROMPT,
                ),
            )
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
        print(f"  [{total:3d}] {row['imgname']}  RMSF1={f1:.0%}")

        # Incremental save
        out_data = {
            "model": model_name,
            "n": total,
            "mean_rmsf1": sum(r["rmsf1"] for r in results) / total,
            "results": results,
        }
        out_path.write_text(
            json.dumps(out_data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        rate_limit_delay(model_name)

    total = len(results)
    mean = sum(r["rmsf1"] for r in results) / total if total else 0
    print(f"\nDone: {total} figures, mean RMSF1 = {mean:.1%}")


if __name__ == "__main__":
    main()
