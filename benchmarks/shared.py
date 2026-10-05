"""Shared utilities for benchmark scripts.

Consolidates metrics, API callers, image encoding, and checkpoint
logic used across all benchmarks.
"""

import argparse
import base64
import io
import json
import os
import re
import sys
import time
from pathlib import Path

from PIL import Image

# Every number in a reply or a ground-truth table. Known limits, which apply
# to every system alike: a hyphen directly before a digit is read as a minus
# sign, so "10-20" yields 10 and -20 and a date yields negative parts; a
# leading-dot decimal (".5") loses its point; unit suffixes ("3.2M") are not
# expanded. The ChartX validation runs of the VLMs stored only the numbers
# extracted this way, not the reply text, so they cannot be re-parsed.
NUM_RE = re.compile(r"-?\d[\d,]*\.?\d*(?:[eE][+-]?\d+)?")

LIFESCI_TYPES = {
    "bar_chart", "bar_chart_num", "line_chart", "line_chart_num",
    "box", "histogram",
}

# How DePlot is called: the instruction of its model card, greedy decoding
# and this cap on the reply. The paper quotes both.
DEPLOT_INSTRUCTION = "Generate underlying data table of the figure below:"
DEPLOT_MAX_NEW_TOKENS = 512

EXTRACT_PROMPT = (
    "Extract the data from this chart as a tab-separated table. "
    "Return ONLY the table, no explanation."
)

EXTRACT_PROMPT_DETAILED = """\
Extract the UNDERLYING data values from this chart into a tab-separated table.

Rules:
- For stacked bar charts, extract the individual segment values \
(NOT the cumulative totals)
- Read values from axis scales as precisely as possible
- CRITICAL: check for axis scale multipliers before reading ANY values. \
Matplotlib often puts a small "1e6", "1e5", etc. in the corner of an axis -- \
this means ALL tick values must be multiplied by that factor. \
For example, if the y-axis corner says "1e6" and a tick reads 0.2, the actual \
value is 0.2 * 1e6 = 200000. Also watch for labels like "x10^3", \
"(thousands)", "(millions)", etc. Always report the FULL actual value, \
never the raw tick label
- For boxplots: report min, Q1, median, Q3, max, and any outlier values
- Only include columns that are in the original data -- \
do NOT add computed columns like totals
- Return ONLY the table, no explanation
- Use \\t between columns and \\n between rows
- First row should be column headers\
"""


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def extract_numbers(text):
    """Extract all numeric values from a string."""
    return {
        float(m.group().replace(",", ""))
        for m in NUM_RE.finditer(text)
    }


def extract_number_list(text):
    """Every numeric value in a string, in order, repeats kept."""
    return [float(m.group().replace(",", "")) for m in NUM_RE.finditer(text)]


def extract_numbers_from_rows(rows):
    """Extract all numeric values from a list of row lists."""
    nums = set()
    for row in rows:
        for cell in row:
            nums |= extract_numbers(cell)
    return nums


def within_tolerance(extracted, truth, tolerance=0.05):
    """Whether an extracted value matches a ground-truth value.

    The match is relative and inclusive: 21.0 matches 20.0 at 5%. The bound
    is compared as a product with a little slack because the quotient
    form, abs(e - g) / abs(g) <= 0.05, rejects some values that sit exactly
    on the bound (1.05 against 1.0 gives 0.05000000000000004). A ground-truth
    zero matches anything below 0.01 in magnitude.
    """
    if truth == 0:
        return abs(extracted) < 0.01
    return abs(extracted - truth) <= tolerance * abs(truth) * (1 + 1e-9)


def compute_recall(extracted, ground_truth, tolerance=0.05):
    """Fraction of ground-truth values found in extraction (within tolerance)."""
    if not ground_truth:
        return 0.0
    matched = sum(
        1 for gt in ground_truth
        if any(within_tolerance(ex, gt, tolerance) for ex in extracted)
    )
    return matched / len(ground_truth)


def compute_mape(extracted, ground_truth):
    """Mean absolute percentage error (best-match per GT value)."""
    if not ground_truth:
        return 0.0
    errors = []
    for gt in ground_truth:
        best = None
        for ex in extracted:
            err = abs(ex - gt) / abs(gt) if gt != 0 else abs(ex)
            if best is None or err < best:
                best = err
        if best is not None:
            errors.append(best)
    return sum(errors) / len(errors) if errors else 0.0


def compute_numeric_f1(extracted, ground_truth, tolerance=0.05):
    """F1 between two sets of numbers, matched one to one within a tolerance.

    Each ground-truth value is paired with at most one extracted value that
    lies within ``tolerance`` of it (see within_tolerance). Precision is the
    matched share of the extraction, recall the matched share of the ground
    truth. Callers pass sets, so a value that occurs twice in a table counts
    once. Two empty sets score 1.0; one empty set scores 0.0.

    Values are paired greedily in ascending order. On every item of the
    stored results this gives the same score as an optimal assignment.

    The paper calls this "numeric F1". It ignores row and column labels, so
    it does not check that a value sits in the right series. The first
    version of the paper, the result files (``rmsf1``, ``mean_rmsf1``) and
    the benchmark runners call it RMSF1; it is NOT the Relative Mapping
    Similarity F1 of Liu et al. (DePlot, 2023), which matches (row header,
    column header, value) triples with partial credit, and scores from the
    two must not be compared.
    """
    if not ground_truth and not extracted:
        return 1.0
    if not ground_truth or not extracted:
        return 0.0
    ex_list = sorted(extracted)
    gt_list = sorted(ground_truth)
    gt_matched = set()
    ex_matched = set()
    for i, g in enumerate(gt_list):
        for j, e in enumerate(ex_list):
            if j in ex_matched:
                continue
            if within_tolerance(e, g, tolerance):
                gt_matched.add(i)
                ex_matched.add(j)
                break
    precision = len(ex_matched) / len(ex_list)
    recall = len(gt_matched) / len(gt_list)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


# The name the benchmark runners and stored result files use; see the
# docstring above for why it is misleading and kept only for them.
compute_rmsf1 = compute_numeric_f1


def bootstrap_ci(values, n_boot=2000, ci=95, seed=0):
    """Bootstrap confidence interval for the mean.

    Seeded by default so a regeneration of numbers.tex is byte-stable; two
    unseeded runs differed in 8 of 9 ChartXCI* macros by up to 0.2 points.
    Pass seed=None for a non-deterministic draw.
    """
    import numpy as np
    arr = np.array(values)
    rng = np.random.default_rng(seed)
    means = rng.choice(arr, size=(n_boot, len(arr)), replace=True).mean(axis=1)
    lo = float(np.percentile(means, (100 - ci) / 2))
    hi = float(np.percentile(means, 100 - (100 - ci) / 2))
    return lo, hi


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def parse_tsv(text):
    """Parse a TSV string (possibly with markdown fences) into rows."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
    text = text.removesuffix("```")
    rows = []
    for line in text.strip().split("\n"):
        line = line.strip()
        if line:
            rows.append([c.strip() for c in line.split("\t")])
    return rows


def parse_chartx_csv(csv_field):
    """Convert a ChartX CSV field (escaped \\t / \\n) to plain text."""
    return csv_field.replace("\\t", "\t").replace("\\n", "\n")


def deplot_rows(text, keep_title=False):
    """Parse DePlot's linearised table into rows of cells, without its title.

    DePlot writes ``TITLE | <chart title> <0x0A> header | ... <0x0A> row``.
    The title is not part of the table, and a year in it used to be scored as
    an extracted value that matches nothing, which lowered DePlot's precision
    on 220 of the 300 ChartX validation items. The VLMs are told to return
    only the table, so the title row is dropped to compare like with like.
    ``keep_title=True`` returns the reply as the runners first scored it, to
    show what that convention costs.
    """
    rows = []
    for line in text.split("<0x0A>"):
        line = line.strip()
        if line:
            rows.append([c.strip() for c in line.split("|")])
    if not keep_title and rows and rows[0][0].strip().upper() == "TITLE":
        rows = rows[1:]
    return rows


def plotqa_truth(entry):
    """The plotted values of a PlotQA subset entry, as a list of floats.

    Each entry of external/plotqa_test_*.json describes the first series of
    one chart. For a horizontal bar chart (its annotation carries
    ``<s_width>``) the bar lengths are in ``x_values`` and ``y_values`` holds
    the category labels; for every other chart the values are in
    ``y_values``.

    The ``<s_width>`` test is sound for these files only because of what
    they hold: all 898 entries with the tag are horizontal bar charts and the
    other 102 are line or dot-line plots. A vertical bar chart would carry
    the same tag with the axes the other way round, and the files have none.
    Do not use this function on other PlotQA data without checking that.

    Until October 2026 the runners read ``y_values`` for every chart. For the
    427 horizontal bar charts among the 529 scored items that made the
    "ground truth" the category labels, which there are years, and a score
    near 100% meant only that the reply had a column of years. The ``gt_nums``
    and ``rmsf1`` fields stored in results/plotqa/*.json date from then and
    must not be used; score_plotqa.py rebuilds the truth with this function.
    """
    horizontal = "<s_width>" in entry.get("raw_text", "")
    values = []
    for value in entry.get("x_values" if horizontal else "y_values", []):
        try:
            values.append(float(value.strip()))
        except ValueError:
            continue
    return values


# ---------------------------------------------------------------------------
# Image encoding
# ---------------------------------------------------------------------------

def encode_image(img_path, max_size=1024):
    """Load an image, thumbnail it, and return (PIL image, base64 string)."""
    img = Image.open(img_path)
    img.thumbnail((max_size, max_size))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    return img, b64


# ---------------------------------------------------------------------------
# API callers
# ---------------------------------------------------------------------------

def load_anthropic_key():
    """Load Anthropic API key from env or secrets file."""
    key = os.environ.get("ANTHROPIC_API_KEY")
    if key:
        return key
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from paths import SECRETS_PATH
    if SECRETS_PATH.exists():
        for line in SECRETS_PATH.read_text(encoding="utf-8").splitlines():
            if line.strip().startswith("ANTHROPIC_API_KEY"):
                return line.split("=", 1)[1].strip().strip('"')
    sys.exit("No Anthropic API key found.")


# Same as the app.  Sonnet 5.5 and Opus 5.5 think by default and thinking
# counts against max_tokens, so the old 2000 would truncate their tables.
CLAUDE_MAX_TOKENS = 16384


PROVIDER_KEYS = {
    "claude": ("ANTHROPIC_API_KEY", "Anthropic"),
    "haiku": ("ANTHROPIC_API_KEY", "Anthropic"),
    "sonnet": ("ANTHROPIC_API_KEY", "Anthropic"),
    "opus": ("ANTHROPIC_API_KEY", "Anthropic"),
    "gpt": ("KEY_API_OPENAI", "OpenAI"),
    "gemini": ("KEY_API_GOOGLE", "Google"),
    "mistral": ("KEY_API_MISTRAL", "Mistral"),
    "ministral": ("KEY_API_MISTRAL", "Mistral"),
}


def require_api_key(model_name):
    """Exit loudly when the API key this model needs is not set.

    Every provider call is wrapped in `except Exception`, which scores the
    figure 0 and continues. A missing key therefore used to produce a
    complete result file of zeros that looks like a real measurement: a run
    with no key reported "0 figures, mean RMSF1 = 0.0%" rather than failing.
    Checking once at startup turns that into an immediate error.
    """
    for prefix, (env_var, provider) in PROVIDER_KEYS.items():
        if not model_name.startswith(prefix):
            continue
        if prefix in ("claude", "haiku", "sonnet", "opus"):
            if load_anthropic_key():
                return
        elif os.environ.get(env_var):
            return
        sys.exit(
            f"No {provider} API key for model {model_name}.\n"
            f"Set {env_var} and re-run. Without it every call fails and the "
            "run scores every figure 0."
        )
    sys.exit(f"Unknown model: {model_name}")


def require_chartx_images():
    """Exit loudly when the ChartX images are missing.

    The benchmark loops skip any figure whose PNG is absent, so on a machine
    without the images a run used to finish with no results and no error.
    The download hint is printed relative to the current directory, so it
    can be pasted wherever the benchmark was launched from.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from paths import CHARTX_IMG_ROOT, VALIDATION_ROOT
    if not CHARTX_IMG_ROOT.is_dir():
        fetch = VALIDATION_ROOT / "pipeline" / "fetch_chartx_images.py"
        sys.exit(
            f"ChartX images not found at {CHARTX_IMG_ROOT}.\n"
            f"Download them with: {Path(sys.executable).name} "
            f"{os.path.relpath(fetch)}"
        )


def claude_reply_text(resp):
    """Return the text of a Claude reply, raising when there is none.

    Sonnet 5.5 and Opus 5.5 think by default, so ``content[0]`` can be a
    thinking block; text blocks are read by type instead.  A refusal or a
    truncated reply raises, and the benchmark scores the figure as a failure,
    which is what an app user would get.
    """
    if resp.stop_reason == "refusal":
        category = getattr(getattr(resp, "stop_details", None), "category", None)
        raise RuntimeError(f"refusal (category: {category})")
    if resp.stop_reason == "max_tokens":
        raise RuntimeError(f"hit max_tokens={CLAUDE_MAX_TOKENS}")
    text = "".join(b.text for b in resp.content if b.type == "text")
    if not text.strip():
        raise RuntimeError("empty reply")
    return text


def call_claude(model, b64, prompt=EXTRACT_PROMPT):
    """Call Claude with a base64 image and a text prompt.

    No thinking or effort parameters are sent, as in the app, so each model
    runs at its API defaults (Sonnet 5.5: adaptive thinking at effort high;
    Opus 5.5: adaptive thinking at effort medium).
    """
    import anthropic
    client = anthropic.Anthropic(api_key=load_anthropic_key())
    resp = client.messages.create(
        model=model, max_tokens=CLAUDE_MAX_TOKENS,
        messages=[{"role": "user", "content": [
            {"type": "image", "source": {
                "type": "base64", "media_type": "image/png", "data": b64}},
            {"type": "text", "text": prompt},
        ]}],
    )
    return claude_reply_text(resp)


def call_gpt(model, b64, prompt=EXTRACT_PROMPT):
    """Call GPT with a base64 image and a text prompt."""
    from openai import OpenAI
    client = OpenAI(api_key=os.environ.get("KEY_API_OPENAI"))
    resp = client.chat.completions.create(
        model=model, max_completion_tokens=2000,
        messages=[{"role": "user", "content": [
            {"type": "image_url",
             "image_url": {"url": f"data:image/png;base64,{b64}"}},
            {"type": "text", "text": prompt},
        ]}],
    )
    return resp.choices[0].message.content


def call_gemini(model, img, prompt=EXTRACT_PROMPT):
    """Call Gemini with a PIL image and a text prompt."""
    # Imported from the defining modules: the package's __init__ re-imports
    # these names without declaring them public, which pyright rejects.
    from google.generativeai.client import configure
    from google.generativeai.generative_models import GenerativeModel
    configure(api_key=os.environ.get("KEY_API_GOOGLE"))
    m = GenerativeModel(model)
    resp = m.generate_content([img, prompt])
    return resp.text


def call_mistral(model, b64, prompt=EXTRACT_PROMPT):
    """Call Mistral with a base64 image and a text prompt."""
    # mistralai 3.x moved the client class out of the top-level package.
    from mistralai.client import Mistral
    client = Mistral(api_key=os.environ.get("KEY_API_MISTRAL"))
    resp = client.chat.complete(
        model=model,
        messages=[{"role": "user", "content": [
            {"type": "image_url",
             "image_url": f"data:image/png;base64,{b64}"},
            {"type": "text", "text": prompt},
        ]}],
    )
    # The SDK types the message as optional; an empty reply is returned as
    # None, the same as a GPT reply with no content.
    message = resp.choices[0].message
    return message.content if message else None


def call_model(model_name, model_id, *, b64=None, img=None, prompt=EXTRACT_PROMPT):
    """Dispatch to the right API caller based on model name prefix."""
    if model_name.startswith(("haiku", "sonnet", "opus", "claude")):
        return call_claude(model_id, b64, prompt)
    elif model_name.startswith("gpt"):
        return call_gpt(model_id, b64, prompt)
    elif model_name.startswith("gemini"):
        return call_gemini(model_id, img, prompt)
    elif model_name.startswith("mistral") or model_name.startswith("ministral"):
        return call_mistral(model_id, b64, prompt)
    raise ValueError(f"Unknown model: {model_name}")


def pinned_model(name):
    """Argparse type for --model: refuse a provider's moving "-latest" alias.

    Mistral repoints mistral-small-latest and mistral-medium-latest when a new
    version ships, so a result stored under the alias cannot be tied to a
    model, and resuming such a run mixes two models in one file. The dated
    identifier names one model for good.
    """
    if name.endswith("-latest"):
        raise argparse.ArgumentTypeError(
            f"{name} is a moving alias; pass the dated model id instead "
            "(e.g. mistral-small-2603, mistral-medium-2604)"
        )
    return name


def rate_limit_delay(model_name):
    """Sleep appropriate amount for a model's rate limit."""
    if "medium" in model_name or "magistral" in model_name:
        time.sleep(3.0)
    elif "mistral" in model_name or "ministral" in model_name:
        time.sleep(1.0)
    else:
        time.sleep(0.3)


def retry_on_rate_limit(func, max_attempts=5):
    """Call func(), retrying on 429 errors with exponential backoff."""
    for attempt in range(max_attempts):
        try:
            return func()
        except Exception as e:
            if "429" in str(e) and attempt < max_attempts - 1:
                wait = 2 ** attempt * 5
                print(f"  [rate-limit] retry in {wait}s")
                time.sleep(wait)
            else:
                raise


# ---------------------------------------------------------------------------
# Checkpoint (save / resume)
# ---------------------------------------------------------------------------

def load_checkpoint(path):
    """Load existing results for resume. Returns (results_dict, done_set)."""
    results = {}
    done = set()
    if path.exists():
        data = json.loads(path.read_text("utf-8"))
        for ct, items in data.get("results", {}).items():
            results[ct] = []
            for item in items:
                if isinstance(item, dict) and "imgname" in item:
                    done.add(item["imgname"])
                    results[ct].append(item)
    if done:
        print(f"  Resuming: {len(done)} already done")
    return results, done


def save_checkpoint(path, results, **metadata):
    """Save results incrementally."""
    data = {**metadata, "results": dict(results)}
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
    )
