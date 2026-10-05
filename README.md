# PlotPick validation

Benchmark code, per-item results and analysis scripts for the PlotPick paper: *PlotPick: AI-powered batch extraction of numerical data from scientific figures*.

Every number, table and figure in the paper that describes its own results is regenerated from the per-item result files in `results/` by the four commands below. Rescoring needs no API key, no model download and no network access, so the tables can be checked in seconds.

Companion repository: the PlotPick application lives at <https://github.com/tommycarstensen/plotpick>.

## Reproducing the paper

None of these commands calls a model.

```
pip install -r requirements-core.txt   # matplotlib, numpy, Pillow only

# Table 3 (PlotQA) and the prompt comparison
python benchmarks/score_plotqa.py

# Every paired interval the paper quotes: each VLM against DePlot, overall
# and per chart type; adjacent rows of Table 2; PlotQA; the two prompts;
# and Table 4, every system at four tolerances
python benchmarks/paired_bootstrap.py

# Figures 1 and 2 (ChartX validation split), written to out/
python benchmarks/plot_chartx_validation.py

# Every macro the paper uses, written to out/numbers.tex. main.tex reads
# them via \input{numbers}, so no result in the paper is typed by hand.
# Table 2 (numeric F1, its interval, recall, margin over DePlot) is here.
python benchmarks/export_paper_numbers.py
```

`requirements-core.txt` is deliberately small. The full `requirements.txt` additionally pulls torch, torchvision and the provider SDKs, which are needed only to re-run the benchmarks.

The bootstraps are seeded, so each command writes the same output every time on a given numpy version. `export_paper_numbers.py` exits with an error if a statement the paper makes in words (for example "every VLM leads DePlot in aggregate") stops holding for the data.

## Scoring errors corrected in October 2026

Version 1 of the paper, and this repository until 5 October 2026, scored two things wrongly. Both were found by review, not by the author, and both are corrected in the scripts above.

**PlotQA was scored against the wrong axis.** Each entry of `external/plotqa_test_1000.json` describes one series of a chart, with `x_values` and `y_values`. The runners took the series from `y_values` for every chart. For a horizontal bar chart those are the category labels and the bar lengths are in `x_values`. Of the 529 scored items, 427 are horizontal bar charts whose categories are years, so their "ground truth" was a list of years, and a score near 100% meant only that the reply contained a column of years. Version 1 reported 86-99% for the VLMs on that basis. Scored against the plotted values the picture is different: DePlot 87.0%, two VLMs at 89%, and the other four between 83% and 57%. `shared.plotqa_truth` now builds the truth, the runners use it, and `score_plotqa.py` ignores the `gt_nums` and `rmsf1` fields stored in `results/plotqa/*.json`, which date from before the fix.

The 529 items are the entries for which the mistaken field happened to be numeric, because the runners skipped the rest. They are not a random sample of PlotQA: 427 horizontal bar charts with years on the category axis and 102 other charts. A rerun with the corrected runners would cover all 1000 entries.

**DePlot's title line was scored as output.** DePlot begins every reply with `TITLE | <chart title>`, and 220 of its 300 ChartX titles contain a number, usually a year, which was counted as an extracted value that matches nothing. The VLMs were told to return only the table. DePlot is now scored without that row (`shared.deplot_rows`): 74.3% numeric F1 on ChartX instead of 72.8%.

A third, smaller correction: the tolerance test rejected a few values lying exactly on the 5% bound through floating-point rounding (`shared.within_tolerance`); five ChartX scores moved by 0.1.

## The metrics, and what they do not measure

**Numeric F1** is the F1 between the set of numbers in a model's table and the set in the ground truth, matched one to one within 5% relative tolerance. **Recall** is the share of ground-truth numbers matched, with no precision term. Both ignore row and column labels: a value read correctly but attached to the wrong series, group or timepoint still counts. Both work on sets, so a value that occurs twice in a table counts once. Numbers in headers and row labels are counted like any other.

The code, the result files and version 1 of the paper call numeric F1 `rmsf1`. It is not the Relative Mapping Similarity F1 of the DePlot paper (Liu et al. 2023), which matches (row header, column header, value) triples with partial credit, and the two must not be compared. `compute_numeric_f1` in `benchmarks/shared.py` is the definition; `compute_rmsf1` is kept as an alias for the benchmark runners.

Numbers are found with one regular expression (`NUM_RE` in `shared.py`). It reads a hyphen directly before a digit as a minus sign, so a range such as `10-20` yields 10 and -20, and it does not expand unit suffixes such as `3.2M`. This applies to every system alike. The ChartX validation runs of the VLMs stored only the extracted numbers, not the reply text, so they cannot be re-parsed.

For PlotQA, `score_plotqa.py` reports two numbers per system. **Best-series numeric F1** scores each column and each row of the reply against the one annotated series and keeps the best. It is optimistic: the ground truth chooses which series of the reply is graded, so a reply that reads every value correctly but assigns it to the wrong series still scores 100%. **Whole-table numeric F1** (25-38% for every system) scores the entire reply against that one series, so it charges a system for correctly extracting the series the annotation omits. The two order the systems the same way.

## Re-running the benchmarks

These do call models, cost money and take hours. The DePlot run is CPU inference over 300 figures and took roughly seven hours on a laptop.

```
# Fetch the ChartX images (~440 MB); annotations are already in external/
python pipeline/fetch_chartx_images.py

# DePlot on the same validation split the VLMs were scored on
python benchmarks/benchmark_deplot.py --split val --n 50

# VLM runs: ANTHROPIC_API_KEY, KEY_API_OPENAI, KEY_API_GOOGLE or
# KEY_API_MISTRAL, read from the environment or a .env file
python benchmarks/benchmark_final.py --model haiku
```

Large data (images) is kept outside the repository, under `~/plotpick_data` or the directory named by `PLOTPICK_DATA`. The PlotQA images are not fetched by any script here; `benchmark_plotqa_multi.py` says where it expects them.

Model calls are made at each provider's defaults, with no seed, so a rerun will not reproduce the stored outputs exactly. The stored per-item results are the reproducible artefact; the scoring on top of them is deterministic. The runners were corrected in October 2026 as described above and have not been re-run since.

## Layout

| Path | What it holds |
|---|---|
| `benchmarks/` | Benchmark runners, metrics (`shared.py`), the list of reported systems (`systems.py`) and the four scoring scripts above |
| `pipeline/` | Dataset fetching and PubMed Central pair construction. `match_pairs.py` imports `pdf_figures` from the application repository and expects it cloned beside this one |
| `external/` | Benchmark annotations (ChartX, PlotQA). Images are not stored here |
| `results/final_val/` | ChartX validation split, the nine VLMs. **The paper's headline numbers** |
| `results/chartx/chartx_deplot_val.json` | DePlot on the same validation split |
| `results/chartx/` (other files) | ChartX development split: runs of the reported models and of others |
| `results/plotqa/plotqa_*.json` | PlotQA, six VLMs and our own DePlot run, simple prompt |
| `results/plotqa/detailed_prompt_*.json` | PlotQA, the two Claude models with the detailed prompt (see `prompts.md`) |
| `results/pmc/`, `reports/` | PubMed Central figure/table pairs and their summary reports (see the caveat below) |
| `results/archive/` | Runs not reported in the paper (next section) |
| `pairs.json`, `candidates.csv`, `excluded_pairs.json` | The PubMed Central pairs and how they were selected |
| `prompts.md` | The extraction prompts, and which results used which |
| `paths.py`, `ruff.toml`, `setup.cfg`, `pyrightconfig.json` | Shared paths and lint configuration |

## Runs that are in this repository but not in the paper's tables

- **Moving-alias runs** (`results/archive/alias_runs/`). Early Mistral runs used the `-latest` aliases, which Mistral repoints when a new version ships, so they cannot be tied to a model version. The three reported Mistral models were re-run under dated identifiers. `ministral-3b-latest` was not re-run; the paper states its result in the text (76.7% numeric F1 on 295 items, 2.8 points above DePlot with an interval that includes zero) without tabulating it. `plotqa_mistral_medium_latest.json` cannot be scored as stored: most of its tables are aligned with spaces, which `score_plotqa.py` reads as a single column.
- **Development split** (`results/chartx/`). Runs of the six non-Mistral models, of Gemini 2.5 Flash, Ministral 3B and 8B, the Mistral aliases, a pixel-ruler computer-vision baseline, and a small DePlot run (10 items per chart type). The Claude runs there cover all 18 ChartX chart types and used the detailed prompt. The paper's tables and figures use the validation split only; one sentence of the paper compares the two splits.
- **Chart types.** ChartX has 18; the paper reports the six listed in `benchmarks/systems.py`. `chartx_deplot_val.json` holds DePlot on all 18.
- **PubMed Central, Opus** (`results/pmc/opus/`, 13 pairs). The paper quotes the Haiku and Sonnet runs only.
- **Early and exploratory runs** (`results/archive/`): about 130 earlier Sonnet extractions of PubMed Central pairs (`sonnet_old/`, `sonnet_old2/`, `PMC*.json`), partial PlotQA runs and one-off comparisons.

## Two things to know before using these results

**The PMC results are not a fair accuracy measure.** `results/pmc/` scores extractions from real biomedical figures against the article's companion table, asking the model to fill in the table's layout from numbers printed in the figure. The ground truth has a median of 71 cells per figure and a maximum of 1060, far more than any single figure plots, so recall is structurally capped: the median figure has about 5 extracted values against those cells. Mean recall is around 13%, but restricting to in-scope chart types does not raise it, and the confound is in the denominator rather than the chart types. The paper states this run and draws no conclusion from it; the numbers should not be quoted as PlotPick's accuracy on biomedical figures. Answering that question properly needs ground truth limited to values actually plotted; that work has not been done.

**The ChartX annotations here are modified.** Eight figures had errors in the upstream ground truth, all proposed upstream as pull requests and all in the development split. See `external/chartx/CORRECTIONS.md` for the full list with links. No validation-split item was changed, so the paper's ChartX tables and figures run on uncorrected upstream data.

## Datasets and licensing

`NOTICE` records the attribution that the redistributed annotations require: ChartX is Apache-2.0 and the copies here are marked as modified; PlotQA is CC-BY-4.0. Licence texts are in `LICENSES/`. No chart images are redistributed; they are fetched from the original sources. Note that the result files do embed ground-truth values from both datasets.

Subsets of ChartQA, ChartQAPro and ChartMuseum were used in exploratory work. They are not included here because their redistribution terms were not established, and none is reported in the paper. `benchmarks/benchmark_chartqa.py` and `benchmarks/run_overnight.py` still reference them and will not run without fetching that data yourself.

`results/pmc/`, `reports/`, `pairs.json`, `candidates.csv` and `excluded_pairs.json` contain values and metadata from figures and tables in PubMed Central Open Access Subset articles, identified by PMCID. Each source article carries its own licence. No patient-identifiable data is included; all values are aggregate figures as published.

## Licence

MIT, see `LICENSE`, matching the PlotPick application repository.

The MIT grant covers the code here. It does not extend to the third-party benchmark data this repository redistributes or embeds: ChartX is Apache-2.0 and PlotQA is CC-BY-4.0, and `NOTICE` records their terms.
