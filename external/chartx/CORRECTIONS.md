# ChartX ground-truth corrections

The ChartX annotation files in this directory are modified copies of the upstream dataset. This file records every change, so that the modifications are identifiable as Apache-2.0 section 4(b) requires, and so that the effect of the corrections on the reported results can be checked.

## Summary

Eight figures were found to have errors in the upstream ground truth during evaluation. All eight were proposed upstream as pull requests on the Hugging Face dataset repository (nine pull requests, #3 to #11; #3 is closed in favour of #11, and the other eight were open and unmerged on 5 October 2026). `chartx_prs.json` in this directory holds the machine-readable record with their URLs.

**All eight are in the development (test) split. None is in the validation split.** Every ChartX result in the paper's tables and figures, including the DePlot comparison, is computed on the validation split, so none of them is affected by these corrections. The paper's development-split results (the twelve unreported chart types, the gap between the two splits, the margins of the two Claude models on the six reported types, and, in its Section 7, the figures of version 1) are the only results the corrections can touch.

## How the errors were found

The errors surfaced while reviewing figures on which several models disagreed with the ground truth. Each was then checked by eye against the rendered chart, and only cases where the ground truth was demonstrably wrong about the chart as drawn were changed. No change was made because a model's answer looked preferable. The bug classes below are all failures of the upstream data rather than judgement calls: a CSV column that is not plotted, a redrawing script with a typo or an undefined variable, or stacked-bar annotations that report totals where the segments are the data.

Because the search was prompted by model disagreement, the process could in principle miss ground-truth errors that every model happened to reproduce. The corrections are confined to the development split, so this cannot affect the validation-split results reported in the paper.

## The corrections

| Figure | Chart type | Bug class | What was wrong | Upstream pull request |
|---|---|---|---|---|
| `bar_263` | bar_chart | redrawing_stacking | Oxford enrolment hardcoded as 1 instead of 10; ratio stacked on enrolment | [#11](https://huggingface.co/datasets/InternScience/ChartX/discussions/11) (supersedes [#3](https://huggingface.co/datasets/InternScience/ChartX/discussions/3)) |
| `bar_455` | bar_chart | redrawing_stacking | Incorrect stacking order: bottom = solar instead of solar + hydro | [#6](https://huggingface.co/datasets/InternScience/ChartX/discussions/6) |
| `bar_num_83` | bar_chart_num | misleading_annotation | Stacked-bar annotations show totals instead of segment values | [#5](https://huggingface.co/datasets/InternScience/ChartX/discussions/5) |
| `bar_num_297` | bar_chart_num | redrawing_wrong_data | Redrawing code has undefined variables and wrong annotations | [#9](https://huggingface.co/datasets/InternScience/ChartX/discussions/9) |
| `bar_num_363` | bar_chart_num | csv_extra_column | CSV has a Carbon Emissions column not depicted in the chart | [#4](https://huggingface.co/datasets/InternScience/ChartX/discussions/4) |
| `bar_num_454` | bar_chart_num | misleading_annotation | Stacked-bar annotations show totals instead of segment values | [#10](https://huggingface.co/datasets/InternScience/ChartX/discussions/10) |
| `line_num_315` | line_chart_num | csv_extra_column | CSV has a Donation Amount column not depicted in the chart | [#7](https://huggingface.co/datasets/InternScience/ChartX/discussions/7) |
| `line_num_361` | line_chart_num | csv_extra_column | CSV has an Employees column not depicted in the chart | [#8](https://huggingface.co/datasets/InternScience/ChartX/discussions/8) |

## What was changed where

- **CSV annotation edits** (`chartx_test.json`): `bar_num_363`, `line_num_315` and `line_num_361` had the undepicted column removed from the ground-truth table. Recall was recomputed for all three across all models.
- **Image regenerations**: `bar_263`, `bar_num_297` and `bar_num_454` were re-rendered from corrected redrawing code, and the affected models were re-run on the corrected images. Neither the corrected code nor the regenerated images are stored here: the `redrawing` fields of `chartx_test.json` are the upstream ones, and the pull requests linked above describe each fix.
- **Reported upstream only**: `bar_455` and `bar_num_83` were reported upstream and left as they are here. Their ground-truth tables in `chartx_test.json` are the upstream ones and their images were not regenerated.
- **Exclusions**: entries covered by pull requests #3-#6 were removed from some result files where they had survived an earlier pass, so that aggregate metrics are computed over a consistent item set.

Only the three CSV edits change a file in this repository: the images are not stored here.

## Reproducing against pristine upstream data

To check the effect of the corrections, fetch the upstream annotations from <https://huggingface.co/datasets/InternScience/ChartX> and rerun the scoring. The validation-split numbers should be identical, since no validation item was touched. The validation items were not audited for errors of this kind.
