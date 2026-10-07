"""Write the scored benchmark items as a Hugging Face dataset, in hf_dataset/.

One row per item and system: the item's ground truth, what the system
returned, and its scores, for the systems and items the paper reports:

    chartx/validation.jsonl   six ChartX chart types of the validation split,
                              nine VLMs and DePlot (2,999 rows: the Claude
                              Haiku 4.5 run has no result for one item)
    plotqa/test.jsonl         529 items of the PlotQA test split, six VLMs
                              and DePlot

The scores are the paper's: they come from paired_bootstrap.chartx_validation
and score_plotqa.reported_items, the loaders export_paper_numbers.py uses,
and every number set written here is rescored and checked against them. The
dataset card (README.md) and the licence notes are written alongside, and the
per-system means are printed for comparison with the paper. No chart images
are written; each row names the image in its source dataset.

Usage, from the repository root:
    python -m benchmarks.export_hf_dataset
"""

import json
import statistics
from pathlib import Path

from benchmarks.paired_bootstrap import as_numbers, chartx_validation
from benchmarks.score_plotqa import annotations, reported_items
from benchmarks.shared import (
    compute_numeric_f1,
    deplot_rows,
    extract_numbers_from_rows,
    plotqa_truth,
)
from benchmarks.systems import BY_KEY, BY_PLOTQA_KEY, CHART_TYPES, DEPLOT
from paths import (
    CHARTX_RESULTS_DIR,
    CHARTX_VAL_META,
    FINAL_VAL_RESULTS_DIR,
    PLOTQA_RESULTS_DIR,
    VALIDATION_ROOT,
)

OUT = VALIDATION_ROOT / "hf_dataset"
TOLERANCE = 0.05


def _rows(data):
    """The result rows of one file, flat, whatever its layout."""
    results = data["results"]
    if isinstance(results, dict):
        return [row for group in results.values() for row in group]
    return list(results)


def _chartx_files():
    """{system key: result file} for the systems the paper reports."""
    files = {DEPLOT.key: CHARTX_RESULTS_DIR / "chartx_deplot_val.json"}
    for path in sorted(FINAL_VAL_RESULTS_DIR.glob("*.json")):
        model = json.loads(path.read_text()).get("model")
        if model in BY_KEY and model != DEPLOT.key:
            files[model] = path
    return files


def chartx_records():
    """One record per reported ChartX item and system."""
    scores = chartx_validation(TOLERANCE)
    meta = {
        entry["imgname"]: entry
        for entry in json.loads(CHARTX_VAL_META.read_text())
        if entry["chart_type"] in CHART_TYPES
    }
    records = []
    for key, path in _chartx_files().items():
        system = BY_KEY[key]
        data = json.loads(path.read_text())
        for row in _rows(data):
            if row["chart_type"] not in CHART_TYPES:
                continue
            name = row["imgname"]
            truth = as_numbers(row["gt_nums"])
            if key == DEPLOT.key:
                extracted = extract_numbers_from_rows(
                    deplot_rows(row["extracted_text"], keep_title=False))
            else:
                extracted = as_numbers(row["ex_nums"])
            item = scores[key][name]
            if compute_numeric_f1(extracted, truth, TOLERANCE) != item.f1:
                raise SystemExit(f"{system.name} {name}: rescoring differs")
            records.append({
                "item_id": name,
                "chart_type": row["chart_type"],
                "chartx_image": f"{row['chart_type']}/png/{name}.png",
                "title": meta[name]["title"].strip(),
                "ground_truth_csv": meta[name]["csv"],
                "system": system.name,
                "provider": system.provider,
                "model_id": system.api_id,
                "truth_numbers": sorted(truth),
                "extracted_numbers": sorted(extracted),
                "reply": row.get("extracted_text") if key == DEPLOT.key else None,
                "numeric_f1": item.f1,
                "recall": item.recall,
            })
    return records


def plotqa_records():
    """One record per reported PlotQA item and system."""
    scores = reported_items(TOLERANCE)
    entries = annotations()
    records = []
    for path in sorted(PLOTQA_RESULTS_DIR.glob("plotqa_*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        key = data["model"]
        if key not in BY_PLOTQA_KEY:
            continue
        system = BY_PLOTQA_KEY[key]
        for row in data["results"]:
            name = row["imgname"]
            entry = entries[name]
            item = scores[key][name]
            records.append({
                "item_id": name,
                "plotqa_test_index": int(entry["idx"]),
                "chart_kind": (
                    "horizontal bar" if item.horizontal else "line or dot-line"
                ),
                "series_name": entry.get("series_name"),
                "truth_values": plotqa_truth(entry),
                "system": system.name,
                "provider": system.provider,
                "model_id": system.api_id,
                "reply": row.get("extracted_text") or "",
                "best_series_f1": item.best,
                "whole_table_f1": item.whole,
            })
    return records


def _write_jsonl(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as out:
        for record in records:
            out.write(json.dumps(record, ensure_ascii=False) + "\n")


def _means(records, field):
    by_system = {}
    for record in records:
        by_system.setdefault(record["system"], []).append(record[field])
    return {
        system: (len(values), statistics.mean(values) * 100)
        for system, values in by_system.items()
    }


def main():
    print(f"Writing the scored benchmark items to {OUT}")
    chartx = chartx_records()
    plotqa = plotqa_records()
    _write_jsonl(OUT / "chartx" / "validation.jsonl", chartx)
    _write_jsonl(OUT / "plotqa" / "test.jsonl", plotqa)
    card = Path(__file__).with_name("hf_dataset_card.md")
    (OUT / "README.md").write_text(card.read_text(encoding="utf-8"),
                                   encoding="utf-8")
    print(f"ChartX: {len(chartx)} rows; mean numeric F1 by system:")
    for system, (n, mean) in sorted(_means(chartx, "numeric_f1").items(),
                                    key=lambda kv: -kv[1][1]):
        print(f"  {system:24s} n={n:3d}  {mean:5.1f}")
    print(f"PlotQA: {len(plotqa)} rows; mean best-series numeric F1 by system:")
    for system, (n, mean) in sorted(_means(plotqa, "best_series_f1").items(),
                                    key=lambda kv: -kv[1][1]):
        print(f"  {system:24s} n={n:3d}  {mean:5.1f}")
    print("Done.")


if __name__ == "__main__":
    main()
