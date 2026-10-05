"""The systems the paper reports, in one place.

Every script that turns stored results into a table, a figure or a macro
reads this list, so a model cannot be called one thing in a table and another
in a figure legend, and a result file that is not listed here cannot reach
the paper by being picked up by a glob.
"""

from typing import NamedTuple


class System(NamedTuple):
    """One benchmarked system.

    key         ``model`` field of its ChartX validation-split result file
    name        name printed in the paper's tables and prose
    macro       letters-only suffix for LaTeX macros (no digits allowed)
    provider    organisation that serves or released the model
    api_id      identifier sent to the provider's API
    plotqa_key  ``model`` field of its PlotQA result file, None if not run
    colour      bar colour in Figure 1
    """

    key: str
    name: str
    macro: str
    provider: str
    api_id: str
    plotqa_key: str | None
    colour: str


# Colours: the Region Hovedstaden palette supplies navy, blue and grey, used
# here for Anthropic and for the DePlot baseline. It has no further hues, so
# the other three providers take one ColorBrewer hue each, darker for the
# model that scored higher. Lightness separates the models of one provider;
# across providers some pairs are close in greyscale and under red-green
# colour blindness, so the figure also keeps the bars in legend order.
VLMS = [
    System("gemini-3-flash-preview", "Gemini 3 Flash", "GeminiFlash",
           "Google", "gemini-3-flash-preview",
           "gemini-3-flash-preview", "#d95f02"),
    System("gemini-3.1-flash-lite-preview", "Gemini 3.1 Flash-Lite",
           "GeminiFlashLite", "Google", "gemini-3.1-flash-lite-preview",
           "gemini-3.1-flash-lite-preview", "#fdb863"),
    System("gpt-5.4-mini", "GPT-5.4 mini", "GptMini",
           "OpenAI", "gpt-5.4-mini", "gpt-5.4-mini", "#1b7837"),
    System("sonnet", "Claude Sonnet 4.6", "Sonnet",
           "Anthropic", "claude-sonnet-4-6", "claude-sonnet-4-6", "#002555"),
    System("gpt-5.4-nano", "GPT-5.4 nano", "GptNano",
           "OpenAI", "gpt-5.4-nano", "gpt-5.4-nano", "#a6dba0"),
    System("haiku", "Claude Haiku 4.5", "Haiku",
           "Anthropic", "claude-haiku-4-5-20251001",
           "claude-haiku-4-5-20251001", "#007dbb"),
    System("mistral-medium-2604", "Mistral Medium 3.5", "MistralMedium",
           "Mistral", "mistral-medium-2604", None, "#3f007d"),
    System("mistral-small-2603", "Mistral Small 4", "MistralSmall",
           "Mistral", "mistral-small-2603", None, "#807dba"),
    System("mistral-large-2512", "Mistral Large 3", "MistralLarge",
           "Mistral", "mistral-large-2512", None, "#cbc9e2"),
]

DEPLOT = System("deplot", "DePlot", "DePlot", "Google", "google/deplot",
                "deplot", "#333333")

SYSTEMS = [*VLMS, DEPLOT]

BY_KEY = {system.key: system for system in SYSTEMS}
BY_PLOTQA_KEY = {
    system.plotqa_key: system for system in SYSTEMS if system.plotqa_key
}

# The six ChartX chart types the paper reports, in the order of its figures.
CHART_TYPES = [
    "bar_chart", "bar_chart_num", "line_chart", "line_chart_num",
    "box", "histogram",
]

# Chart-type names for prose ("... on plain bar charts").
TYPE_PROSE = {
    "bar_chart": "plain bar charts",
    "bar_chart_num": "bar charts with data labels",
    "line_chart": "plain line charts",
    "line_chart_num": "line charts with data labels",
    "box": "box plots",
    "histogram": "histograms",
}

# Axis tick labels for the figures.
TYPE_TICKS = {
    "bar_chart": "Bar",
    "bar_chart_num": "Bar\n(labelled)",
    "line_chart": "Line",
    "line_chart_num": "Line\n(labelled)",
    "box": "Box",
    "histogram": "Histogram",
}

# Where the heatmap's colour scale starts, in per cent. The one cell below
# it (DePlot on box plots) takes the lowest colour and still prints its
# value; the paper's caption quotes this number.
HEATMAP_FLOOR = 50

# Each data-labelled type and the same chart family without labels.
LABELLED_PAIRS = [
    ("bar_chart_num", "bar_chart"),
    ("line_chart_num", "line_chart"),
]
