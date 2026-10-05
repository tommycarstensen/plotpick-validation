"""Benchmark Claude on ChartQA (question answering, not chart-to-table).

Uses relaxed accuracy with 5% numerical tolerance.

This is exploratory -- output goes to results/archive/.

Usage:
    python benchmark_chartqa.py              # 200 figures, haiku
    python benchmark_chartqa.py --model sonnet
"""

import argparse
import io
import json
import re
import sys
import time
from pathlib import Path

if isinstance(sys.stdout, io.TextIOWrapper):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from paths import LOCAL_DATA, RESULTS_DIR, EXTERNAL_DIR  # noqa: E402
from benchmarks.shared import encode_image, load_anthropic_key, call_claude  # noqa: E402

CHARTQA_IMG_DIR = LOCAL_DATA / "datasets" / "chartqa" / "images"
ARCHIVE_DIR = RESULTS_DIR / "archive"

MODELS = {
    "haiku": "claude-haiku-4-5-20251001",
    "sonnet": "claude-sonnet-4-6",
}


def relaxed_match(predicted, gold, tol=0.05):
    """ChartQA relaxed accuracy: exact string or within 5% for numbers."""
    predicted = predicted.strip().lower()
    gold = gold.strip().lower()
    # Strip explanations: take first line, remove parentheticals
    predicted = predicted.split("\n")[0].strip()
    predicted = re.sub(r"\s*\(.*", "", predicted).strip()
    predicted = re.sub(
        r"\s*,\s*(which|because|since|as).*", "", predicted,
    ).strip()
    if predicted == gold:
        return True
    if predicted.startswith(gold):
        return True
    try:
        p_match = re.search(r"-?\d[\d,]*\.?\d*", predicted)
        p = (
            float(p_match.group().replace(",", ""))
            if p_match
            else float(predicted.replace(",", "").replace("%", ""))
        )
        g = float(gold.replace(",", "").replace("%", ""))
        if g == 0:
            return abs(p) < 0.01
        return abs(p - g) / abs(g) <= tol
    except ValueError:
        return False


def _save(model_name, model_id, results, correct, total):
    """Write incremental checkpoint."""
    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    out_path = ARCHIVE_DIR / f"chartqa_{model_name}.json"
    out_data = {
        "model": model_name,
        "model_id": model_id,
        "n": len(results),
        "accuracy": correct / total if total else 0,
        "correct": correct,
        "total": total,
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
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--model", default="haiku", choices=list(MODELS.keys()))
    args = parser.parse_args()

    meta = json.loads(
        (EXTERNAL_DIR / "chartqa_test_200.json").read_text("utf-8"),
    )[:args.n]

    load_anthropic_key()
    model_id = MODELS[args.model]

    results = []
    correct = 0
    total = 0

    for i, row in enumerate(meta):
        img_path = CHARTQA_IMG_DIR / f"{row['imgname']}.png"
        if not img_path.exists():
            continue

        _, b64 = encode_image(img_path)

        prompt = (
            "Answer this question about the chart. "
            "Give ONLY the answer, no explanation.\n\n"
            f"Question: {row['query']}"
        )

        try:
            predicted = call_claude(model_id, b64, prompt=prompt).strip()
        except Exception as e:  # noqa: BLE001
            print(f"  [error] {row['imgname']}: {e}")
            predicted = ""

        gold_labels = (
            row["label"] if isinstance(row["label"], list) else [row["label"]]
        )
        is_correct = any(relaxed_match(predicted, g) for g in gold_labels)
        if is_correct:
            correct += 1
        total += 1

        results.append({
            "imgname": row["imgname"],
            "query": row["query"],
            "predicted": predicted,
            "gold": gold_labels,
            "correct": is_correct,
            "human_or_machine": row["human_or_machine"],
        })

        tag = "OK" if is_correct else "WRONG"
        print(
            f"  [{i + 1:3d}/{len(meta)}] {tag}  pred=\"{predicted[:30]}\" "
            f"gold=\"{gold_labels[0][:30]}\""
        )

        if (i + 1) % 50 == 0:
            _save(args.model, model_id, results, correct, total)
        time.sleep(0.3)

    _save(args.model, model_id, results, correct, total)

    # Summary
    print(f"\n{'=' * 60}")
    print(f"Model: {args.model} ({model_id})")
    print(f"Total: {total}, Correct: {correct}")
    print(f"Relaxed Accuracy: {correct / total:.1%}")

    human = [r for r in results if r["human_or_machine"] == 0]
    machine = [r for r in results if r["human_or_machine"] == 1]
    if human:
        h_acc = sum(r["correct"] for r in human) / len(human)
        print(f"  Human-written: {h_acc:.1%} ({len(human)} questions)")
    if machine:
        m_acc = sum(r["correct"] for r in machine) / len(machine)
        print(f"  Machine-generated: {m_acc:.1%} ({len(machine)} questions)")


if __name__ == "__main__":
    main()
