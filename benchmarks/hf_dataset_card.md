---
pretty_name: PlotPick validation (scored ChartX and PlotQA items)
license: other
license_name: mixed
license_link: https://github.com/tommycarstensen/plotpick-validation/blob/main/NOTICE
task_categories:
- image-to-text
tags:
- chart-to-table
- chart-understanding
- benchmark
- evaluation
- vision-language-models
size_categories:
- 1K<n<10K
configs:
- config_name: chartx
  data_files:
  - split: validation
    path: chartx/validation.jsonl
- config_name: plotqa
  data_files:
  - split: test
    path: plotqa/test.jsonl
---

# PlotPick validation: scored ChartX and PlotQA items

The per-item results behind the benchmark in "PlotPick: AI-powered batch extraction of numerical data from scientific figures" (arXiv:2605.06021, version 2), which compares nine vision-language models (VLMs) with DePlot, a dedicated chart-to-table model, at reading the numbers in chart images. Each row is one chart and one system: the chart's ground truth, what the system returned, and its scores. The code that produced the results, and the script that wrote this dataset (`benchmarks/export_hf_dataset.py`), are at <https://github.com/tommycarstensen/plotpick-validation>.

No chart images are included. Each row names its image in the source dataset (below).

## Configurations

**`chartx`** (split `validation`): six chart types of the validation split of [ChartX](https://huggingface.co/datasets/InternScience/ChartX) (plain bar charts, bar charts with data labels, plain line charts, line charts with data labels, box plots and histograms), 50 charts each, for nine VLMs and DePlot. The Claude Haiku 4.5 run has no result for one chart.

| Field | Meaning |
|---|---|
| `item_id`, `chart_type` | the ChartX image name and chart type |
| `chartx_image` | the image's path inside `ChartX_png.zip` of the ChartX dataset |
| `title`, `ground_truth_csv` | the chart's title and CSV table, as ChartX gives them |
| `system`, `provider`, `model_id` | the system and the identifier it was called with |
| `truth_numbers` | the numbers of the ground-truth table, deduplicated, as scored |
| `extracted_numbers` | the numbers in the system's reply, deduplicated, as scored |
| `reply` | DePlot's raw reply; the VLM runs stored only the numbers |
| `numeric_f1`, `recall` | the scores at 5% relative tolerance |

**`plotqa`** (split `test`): 529 charts from the first 1,000 entries of the [PlotQA](https://github.com/NiteshMethani/PlotQA) test split, in the order of the conversion at [achang/plot_qa](https://huggingface.co/datasets/achang/plot_qa), for six of the VLMs and DePlot. The subset is not representative of PlotQA: 427 of the 529 are horizontal bar charts.

| Field | Meaning |
|---|---|
| `item_id`, `plotqa_test_index` | the item, and its index in the test split of achang/plot_qa |
| `chart_kind` | horizontal bar, or line or dot-line |
| `series_name`, `truth_values` | the annotated series (the first series of the chart) and its plotted values |
| `system`, `provider`, `model_id` | the system and the identifier it was called with |
| `reply` | the system's reply |
| `best_series_f1` | numeric F1 of the reply's best-matching row or column against the annotated series |
| `whole_table_f1` | numeric F1 of every number in the reply against the annotated series |

## How the items were scored

Numeric F1 compares the set of numbers a system returned with the set in the ground truth: a number counts as correct when it is within 5% of a true value, whatever row, column or label it is attached to. It does not check that a value is assigned to the right group or series. The VLMs were asked, with a two-sentence prompt, for the chart's data as a tab-separated table; DePlot was given its standard instruction, and its title row is left out before scoring.

For PlotQA, the ground truth of a horizontal bar chart is taken from the annotation's `x_values` (the bar lengths). Version 1 of the paper took it from `y_values`, which for those charts are the category labels; that was wrong, and the scores here use the corrected truth. Best-series scoring lets the annotation choose which series of the reply is graded, so it is lenient; whole-table scoring charges a reply for the series the annotation leaves out.

## Relation to the ChartX corrections

Ground-truth errors were found in eight ChartX figures while building the benchmark and were proposed upstream as pull requests #4 to #11 on [InternScience/ChartX](https://huggingface.co/datasets/InternScience/ChartX/discussions). All eight are in ChartX's test split, which the paper used for development. None is in the validation split, so the `chartx` configuration here holds the upstream ground truth unchanged. The corrections are listed in `external/chartx/CORRECTIONS.md` of the GitHub repository.

## Limitations

Both benchmarks are synthetic charts. The prompt is not the PlotPick application's, and the application's default model was not benchmarked. Accuracy on published biomedical figures has not been established. See the paper for the results and their uncertainty.

## Licences

The rows reproduce annotations of both source datasets: ChartX (Apache-2.0, copyright the ChartX authors) and PlotQA (CC BY 4.0, Methani, Ganguly, Khapra and Kumar, WACV 2020). The scores and the code are MIT-licensed. The replies are outputs of the systems named in each row. Attribution and the licence texts are in `NOTICE` and `LICENSES/` of the GitHub repository.

## Citation

```bibtex
@misc{carstensen2026plotpick,
  author        = {Carstensen, Tommy},
  title         = {{PlotPick}: {AI}-powered batch extraction of numerical data from scientific figures},
  year          = {2026},
  eprint        = {2605.06021},
  archivePrefix = {arXiv},
  doi           = {10.48550/arXiv.2605.06021},
}
```

Please also cite ChartX (Xia et al., IEEE Transactions on Image Processing 34:7436-7447, 2025) and PlotQA (Methani et al., WACV 2020) when you use their data.
