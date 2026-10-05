# Extraction prompts

## Simple prompt (every result in the paper's tables and figures)

```
Extract the data from this chart as a tab-separated table.
Return ONLY the table, no explanation.
```

Used by `benchmark_final.py` (ChartX validation split), `benchmark_multimodel.py` (ChartX development split, non-Claude models) and `benchmark_plotqa_multi.py` (PlotQA). It is `EXTRACT_PROMPT` in `benchmarks/shared.py`.

## Detailed prompt

```
Extract the UNDERLYING data values from this chart into a tab-separated table.

Rules:
- For stacked bar charts, extract the individual segment values (NOT the cumulative totals)
- Read values from axis scales as precisely as possible
- Only include columns that are in the original data -- do NOT add computed columns like totals
- Return ONLY the table, no explanation
- Use \t between columns and \n between rows
- First row should be column headers
```

This is the text `benchmark_plotqa.py` carried when the runs below were made (17 to 19 April 2026). `EXTRACT_PROMPT_DETAILED` in `benchmarks/shared.py` has since gained rules on axis-scale multipliers and box plots.

Results attributed to a detailed prompt:

- `results/plotqa/detailed_prompt_sonnet.json` and `detailed_prompt_haiku.json`: Claude Sonnet 4.6 and Claude Haiku 4.5 on the 529 PlotQA items. These files were called `plotqa1000_*.json` until October 2026.
- `results/chartx/chartx_sonnet.json` and `chartx_haiku.json`: the two Claude runs on the ChartX development split (`benchmark_chartx.py`).

No result in the paper's tables or figures uses it.

## How sure the attribution is

No result file records the prompt it was made with. The attribution of the two PlotQA files rests on the repository's history: they were first committed on 19 April 2026, when the only PlotQA runner was `benchmark_plotqa.py` and its only prompt was the detailed one; a commit of 21 April archived copies of them as "detailed prompt results"; and the simple-prompt Claude runs (`plotqa_claude_*.json`, written by `benchmark_plotqa_multi.py`) were first committed on 4 May 2026 under a message ending "add simple-prompt PlotQA results". That history belongs to the development repository; the public release is a snapshot without it, so a reader of the release cannot check these dates. The two runs of each model were therefore also made on different dates. The replies of the two runs differ in text on 457 (Sonnet) and 476 (Haiku) of the 529 items. One loose end: `benchmark_plotqa.py` as committed reads the 100-item subset, so the 529-item run was made from a copy pointed at the 1000-item file that was never committed.

## What the comparison shows

`python benchmarks/score_plotqa.py` scores the two Claude runs on the same 529 PlotQA items with the same code, and `python benchmarks/paired_bootstrap.py` gives the paired intervals:

| Model | Detailed | Simple | Difference (paired 95% interval) |
|---|---|---|---|
| Claude Sonnet 4.6 | 88.6% | 89.0% | -0.5 [-1.2, +0.3] |
| Claude Haiku 4.5 | 69.7% | 70.2% | -0.4 [-2.0, +1.1] |

Best-series numeric F1 against the plotted values. Neither difference is separable from zero, and the intervals exclude a gain of more than 0.3 points for Sonnet and 1.1 points for Haiku. Each prompt was run once per model.

Version 1 of the paper reported the detailed prompt as 1 to 3 points better (Haiku 96.3% against 97.6%, Sonnet 98.8% against 99.1%). Those figures were computed against the wrong axis for 427 of the 529 items (see the README), so they measured whether a reply contained a column of years and say nothing about the prompt.

An earlier 20-figure comparison on ChartX (88.5% against 91.6%) has no per-item outputs behind it and is not used.

## Corrections to this file

A revision of 5 October 2026 said that the files archived under `results/plotqa/detailed_prompt/` were byte-identical copies of the simple-prompt run, and that the ablation had therefore been withdrawn. That was wrong. They were byte-identical copies of `plotqa1000_*.json`, the files now named `detailed_prompt_*.json`. The duplicates were removed and no run was lost.

The next revision, also of 5 October, gave the comparison as 98.84% against 98.80% and 96.52% against 96.56%. Those were computed against the wrong axis too.
