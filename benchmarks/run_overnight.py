"""Run all remaining benchmarks. Parallelized where possible.

CPU models (DePlot, TinyChart) use multiprocessing for 4x speedup.
API benchmarks run concurrently via threading.
All results save incrementally — safe to kill and resume.

Usage:
    python run_overnight.py              # run everything
    python run_overnight.py --skip-cpu   # API benchmarks only
    python run_overnight.py --skip-api   # CPU models only
"""

import argparse
import base64
import contextlib
import functools
import io
import json
import os
import re
import sys
import time
import threading
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv

if isinstance(sys.stdout, io.TextIOWrapper):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

load_dotenv()

from paths import (  # noqa: E402
    CHARTX_IMG_ROOT,
    CHARTX_RESULTS_DIR,
    EXTERNAL_DIR,
    PLOTQA_DIR,
    PLOTQA_RESULTS_DIR,
    RESULTS_DIR,
    SECRETS_PATH,
)

RESULTS_DIR.mkdir(exist_ok=True)
CHARTX_RESULTS_DIR.mkdir(exist_ok=True)
PLOTQA_RESULTS_DIR.mkdir(exist_ok=True)

NUM_RE = re.compile(r"-?\d[\d,]*\.?\d*(?:[eE][+-]?\d+)?")

EXTRACT_PROMPT = """\
Extract the UNDERLYING data values from this chart into a tab-separated table.

Rules:
- For stacked bar charts, extract the individual segment values (NOT the cumulative totals)
- Read values from axis scales as precisely as possible
- Only include columns that are in the original data -- do NOT add computed columns like totals
- Return ONLY the table, no explanation
- Use \\t between columns and \\n between rows
- First row should be column headers"""  # noqa: E501 -- prompt kept verbatim


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_api_key():
    key = os.environ.get("ANTHROPIC_API_KEY")
    if key:
        return key
    if SECRETS_PATH.exists():
        for line in SECRETS_PATH.read_text("utf-8").splitlines():
            if line.strip().startswith("ANTHROPIC_API_KEY"):
                return line.split("=", 1)[1].strip().strip('"')
    return None


def extract_numbers(text):
    return [float(m.group().replace(",", "")) for m in NUM_RE.finditer(str(text))]


def extract_numbers_from_rows(rows):
    nums = []
    for row in rows:
        for cell in row:
            nums.extend(
                float(m.group().replace(",", "")) for m in NUM_RE.finditer(str(cell))
            )
    return nums


def compute_rmsf1(ex_nums, gt_nums, tol=0.05):
    if not gt_nums and not ex_nums:
        return 1.0
    if not gt_nums or not ex_nums:
        return 0.0
    gt_m, ex_m = set(), set()
    for i, gt in enumerate(gt_nums):
        for j, ex in enumerate(ex_nums):
            if j in ex_m:
                continue
            if gt == 0:
                if abs(ex) < 0.01:
                    gt_m.add(i)
                    ex_m.add(j)
                    break
            elif abs(ex - gt) / abs(gt) <= tol:
                gt_m.add(i)
                ex_m.add(j)
                break
    p = len(ex_m) / len(ex_nums) if ex_nums else 0
    r = len(gt_m) / len(gt_nums) if gt_nums else 0
    return 2 * p * r / (p + r) if (p + r) else 0.0


def relaxed_match(predicted, gold, tol=0.05):
    predicted = predicted.strip().lower().split("\n")[0]
    predicted = re.sub(r"\s*\(.*", "", predicted).strip()
    predicted = re.sub(r"\s*,\s*(which|because|since|as).*", "", predicted).strip()
    gold = gold.strip().lower()
    if predicted == gold or predicted.startswith(gold):
        return True
    try:
        p_m = re.search(r"-?\d[\d,]*\.?\d*", predicted)
        p = float(p_m.group().replace(",", "")) if p_m else float("nan")
        g = float(gold.replace(",", "").replace("%", ""))
        return abs(p - g) / abs(g) <= tol if g != 0 else abs(p) < 0.01
    except (ValueError, ZeroDivisionError):
        return False


def img_to_b64(img_path):
    from PIL import Image
    img = Image.open(img_path)
    img.thumbnail((1024, 1024))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def save_json(path, data):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def load_json(path):
    if path.exists():
        return json.loads(path.read_text("utf-8"))
    return None


# ---------------------------------------------------------------------------
# DePlot: single-figure worker (for multiprocessing)
# ---------------------------------------------------------------------------
@functools.cache
def _load_deplot():
    """Load the DePlot processor and model once per worker process.

    Heavy, but unavoidable with a process pool: each worker needs its own
    copy. The cache stops it being reloaded for every figure.
    """
    from transformers import Pix2StructForConditionalGeneration, Pix2StructProcessor

    processor = Pix2StructProcessor.from_pretrained("google/deplot")
    model = Pix2StructForConditionalGeneration.from_pretrained("google/deplot")
    return processor, model


def _deplot_one(args_tuple):
    """Process one figure with DePlot. Called in subprocess."""
    img_path_str, gt_csv, imgname, chart_type = args_tuple
    from PIL import Image

    processor, model = _load_deplot()

    gt_text = gt_csv.replace("\\t", "\t").replace("\\n", "\n")
    gt_rows = [line.split("\t") for line in gt_text.strip().split("\n") if line.strip()]
    gt_nums = extract_numbers_from_rows(gt_rows)

    try:
        img = Image.open(img_path_str).convert("RGB")
        # transformers' type hints wrongly reject both of these calls.
        inputs = processor(images=img,
                           text="Generate underlying data table of the figure below:",
                           return_tensors="pt")  # pyright: ignore[reportCallIssue]
        preds = model.generate(  # pyright: ignore[reportAttributeAccessIssue]
            **inputs, max_new_tokens=512
        )
        extracted = processor.decode(preds[0], skip_special_tokens=True)
        ex_rows = [
            row.strip().split("|")
            for row in extracted.split("<0x0A>")
            if row.strip()
        ]
        ex_nums = extract_numbers_from_rows(ex_rows)
        f1 = compute_rmsf1(ex_nums, gt_nums)
    except Exception as e:  # noqa: BLE001
        extracted = f"ERROR: {e}"
        ex_nums = []
        f1 = 0.0

    return {
        "imgname": imgname, "chart_type": chart_type,
        "rmsf1": f1, "gt_csv": gt_csv,
        "extracted_text": extracted,
        "gt_nums": gt_nums, "ex_nums": ex_nums,
    }


# ---------------------------------------------------------------------------
# CPU model runner (DePlot / TinyChart) with multiprocessing
# ---------------------------------------------------------------------------
def run_cpu_model(model_name, n=10, workers=4):
    out_path = CHARTX_RESULTS_DIR / f"chartx_{model_name}.json"
    existing = load_json(out_path)
    done = set()
    results_by_type = defaultdict(list)
    if existing and "results" in existing:
        for ct, items in existing["results"].items():
            for item in items:
                done.add(item["imgname"])
                results_by_type[ct].append(item)

    if len(done) >= 18 * n:
        print(f"[{model_name} ChartX] Already done ({len(done)}). Skipping.")
        return

    meta = json.loads((EXTERNAL_DIR / "chartx" / "chartx_test.json").read_text("utf-8"))
    by_type = defaultdict(list)
    for row in meta:
        by_type[row["chart_type"]].append(row)

    # Build work queue
    work = []
    for chart_type, rows in sorted(by_type.items()):
        for row in rows[:n]:
            if row["imgname"] in done:
                continue
            img_path = (
                CHARTX_IMG_ROOT / row["chart_type"] / "png" / f"{row['imgname']}.png"
            )
            if not img_path.exists():
                continue
            work.append((str(img_path), row["csv"], row["imgname"], chart_type))

    if not work:
        print(f"[{model_name} ChartX] Nothing to do.")
        return

    print(f"[{model_name} ChartX] {len(work)} figures to process with "
          f"{workers} workers ({len(done)} already done)")

    if model_name != "deplot":
        print(f"[{model_name}] Unknown model. Skipping.")
        return

    completed = len(done)
    # Use sequential processing (model can't be shared across processes easily)
    # But use torch threading for internal parallelism
    import torch
    torch.set_num_threads(workers)

    from PIL import Image
    from transformers import Pix2StructProcessor, Pix2StructForConditionalGeneration
    print(f"[{model_name}] Loading model...")
    processor = Pix2StructProcessor.from_pretrained("google/deplot")
    model = Pix2StructForConditionalGeneration.from_pretrained("google/deplot")
    print(f"[{model_name}] Loaded. Using {workers} threads for torch ops.")

    for img_path_str, gt_csv, imgname, chart_type in work:
        gt_text = gt_csv.replace("\\t", "\t").replace("\\n", "\n")
        gt_rows = [
            line.split("\t") for line in gt_text.strip().split("\n") if line.strip()
        ]
        gt_nums = extract_numbers_from_rows(gt_rows)
        if not gt_nums:
            continue

        try:
            img = Image.open(img_path_str).convert("RGB")
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
            ex_rows = [
                row.strip().split("|")
                for row in extracted.split("<0x0A>")
                if row.strip()
            ]
            ex_nums = extract_numbers_from_rows(ex_rows)
            f1 = compute_rmsf1(ex_nums, gt_nums)
        except Exception as e:  # noqa: BLE001
            print(f"  [error] {imgname}: {e}")
            f1 = 0.0
            ex_nums = []
            extracted = ""

        results_by_type[chart_type].append({
            "imgname": imgname, "chart_type": chart_type,
            "rmsf1": f1, "gt_csv": gt_csv,
            "extracted_text": extracted,
            "gt_nums": gt_nums, "ex_nums": ex_nums,
        })
        completed += 1
        print(f"  [{completed:3d}/{len(done)+len(work)}] {chart_type:20s} "
              f"{imgname:15s} RMSF1={f1:.0%}")
        save_json(out_path, {"model": model_name, "n_per_type": n,
                             "results": dict(results_by_type)})

    print(f"[{model_name} ChartX] Done: {completed} figures total.")


# ---------------------------------------------------------------------------
# API benchmark runner
# ---------------------------------------------------------------------------
def run_api_benchmark(dataset_name, meta, img_dir, task, models, out_prefix):
    import anthropic
    api_key = load_api_key()
    if not api_key:
        print(f"[{dataset_name}] No API key. Skipping.")
        return

    MODEL_IDS = {
        "haiku": "claude-haiku-4-5-20251001",
        "sonnet": "claude-sonnet-4-6",
    }
    client = anthropic.Anthropic(api_key=api_key)

    for model_name in models:
        model_id = MODEL_IDS[model_name]
        # Route to subdirectory based on prefix
        if out_prefix.startswith("plotqa"):
            out_dir = PLOTQA_RESULTS_DIR
        elif out_prefix.startswith("chartx"):
            out_dir = CHARTX_RESULTS_DIR
        else:
            out_dir = RESULTS_DIR / "archive"
            out_dir.mkdir(exist_ok=True)
        out_path = out_dir / f"{out_prefix}_{model_name}.json"
        existing = load_json(out_path)
        done = set()
        results_list = []
        if existing and "results" in existing:
            results_list = existing["results"]
            done = {r.get("imgname", "") for r in results_list}

        if len(done) >= len(meta):
            print(f"[{dataset_name} {model_name}] Already done ({len(done)}). "
                  "Skipping.")
            continue

        print(f"[{dataset_name} {model_name}] {len(meta)-len(done)} to do...")
        correct = sum(1 for r in results_list if r.get("correct"))
        all_f1 = [r.get("rmsf1", 0) for r in results_list if "rmsf1" in r]

        for i, row in enumerate(meta):
            key = row.get("imgname", str(i))
            if key in done:
                continue
            img_path = img_dir / f"{key}.png"
            if not img_path.exists():
                continue

            b64 = img_to_b64(img_path)

            if task == "table":
                gt_nums = []
                for v in row.get("y_values", []):
                    with contextlib.suppress(ValueError):
                        gt_nums.append(float(v.strip()))
                if not gt_nums:
                    continue
                try:
                    resp = client.messages.create(
                        model=model_id, max_tokens=2000,
                        messages=[{"role": "user", "content": [
                            {"type": "image", "source": {
                                "type": "base64",
                                "media_type": "image/png", "data": b64}},
                            {"type": "text", "text": EXTRACT_PROMPT}]}])
                    extracted = "".join(
                        b.text for b in resp.content if b.type == "text"
                    )
                    ex_nums = extract_numbers(extracted)
                    f1 = compute_rmsf1(ex_nums, gt_nums)
                except Exception as e:  # noqa: BLE001
                    print(f"  [error] {key}: {e}")
                    f1 = 0.0
                    ex_nums = []
                    extracted = ""
                all_f1.append(f1)
                results_list.append({
                    "imgname": key, "rmsf1": f1,
                    "gt_nums": gt_nums, "ex_nums": ex_nums,
                    "extracted_text": extracted,
                })
                print(f"  [{len(results_list):3d}/{len(meta)}] {key} RMSF1={f1:.0%}")

            elif task == "qa":
                q = row.get("questions", [row.get("question", "")])
                a = row.get("answers", [row.get("answer", "")])
                if isinstance(q, list):
                    q = q[0] if q else ""
                if isinstance(a, list):
                    a = a[0] if a else ""
                try:
                    resp = client.messages.create(
                        model=model_id, max_tokens=100,
                        messages=[{"role": "user", "content": [
                            {"type": "image", "source": {
                                "type": "base64",
                                "media_type": "image/png", "data": b64}},
                            {"type": "text", "text":
                                f"Answer this question about the chart. "
                                f"Give ONLY the answer, no explanation.\n\n"
                                f"Question: {q}"}]}])
                    predicted = "".join(
                        b.text for b in resp.content if b.type == "text"
                    ).strip()
                except Exception as e:  # noqa: BLE001
                    print(f"  [error] {key}: {e}")
                    predicted = ""
                is_correct = relaxed_match(predicted, str(a))
                if is_correct:
                    correct += 1
                results_list.append({
                    "imgname": key, "question": str(q),
                    "predicted": predicted, "gold": str(a), "correct": is_correct,
                })
                total = len(results_list)
                tag = "OK" if is_correct else "WRONG"
                print(f"  [{total:3d}/{len(meta)}] {tag} pred=\"{predicted[:25]}\" "
                      f"gold=\"{str(a)[:25]}\"")

            # Save after every figure
            out_data = {"model": model_name, "model_id": model_id,
                        "n": len(results_list), "results": results_list}
            if task == "qa":
                t = len(results_list)
                out_data["accuracy"] = correct / t if t else 0
                out_data["correct"] = correct
            elif task == "table" and all_f1:
                out_data["mean_rmsf1"] = sum(all_f1) / len(all_f1)
            save_json(out_path, out_data)
            time.sleep(0.3)

        print(f"[{dataset_name} {model_name}] Done: {len(results_list)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--skip-cpu", action="store_true")
    parser.add_argument("--skip-api", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    print("=" * 60)
    print("BENCHMARK RUN")
    print("=" * 60)

    # Start API benchmarks in a thread (they don't use CPU)
    api_thread = None
    if not args.skip_api:
        def run_all_api():
            from paths import LOCAL_DATA
            datasets = LOCAL_DATA / "datasets"

            # PlotQA Sonnet
            print("\n--- API: Sonnet on PlotQA ---")
            plotqa_meta = json.loads(
                (EXTERNAL_DIR / "plotqa_test_100.json").read_text("utf-8"))
            run_api_benchmark("PlotQA", plotqa_meta,
                              PLOTQA_DIR / "images", "table", ["sonnet"], "plotqa")

            # ChartQAPro
            print("\n--- API: ChartQAPro Haiku + Sonnet ---")
            cqp_meta = json.loads(
                (EXTERNAL_DIR / "chartqapro_test_100.json").read_text("utf-8"))
            cqp_img = datasets / "chartqapro" / "images"
            run_api_benchmark("ChartQAPro", cqp_meta,
                              cqp_img, "qa", ["haiku", "sonnet"], "chartqapro")

            # ChartMuseum
            print("\n--- API: ChartMuseum Haiku + Sonnet ---")
            cm_meta = json.loads(
                (EXTERNAL_DIR / "chartmuseum_test_100.json").read_text("utf-8"))
            cm_img = datasets / "chartmuseum" / "images"
            run_api_benchmark("ChartMuseum", cm_meta,
                              cm_img, "qa", ["haiku", "sonnet"], "chartmuseum")

        api_thread = threading.Thread(target=run_all_api, name="api-benchmarks")
        api_thread.start()
        print("[Started API benchmarks in background thread]")

    # Run CPU models in main thread
    if not args.skip_cpu:
        print("\n--- CPU: DePlot on ChartX ---")
        run_cpu_model("deplot", n=10, workers=args.workers)

    # Wait for API thread
    if api_thread:
        print("\n[Waiting for API benchmarks to finish...]")
        api_thread.join()

    print("\n" + "=" * 60)
    print("ALL DONE")
    print("=" * 60)


if __name__ == "__main__":
    main()
