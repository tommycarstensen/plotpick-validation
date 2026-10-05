"""Pixel-ruler: extract numeric values from chart images without LLMs.

Measures pixel distances, gridline spacing, and color boundaries to
extract values from bar charts, histograms, and box plots.

Usage as module:
    from pixel_ruler import extract_values
    result = extract_values("path/to/chart.png", chart_type="histogram")

Usage standalone:
    python pixel_ruler.py path/to/image.png [--type histogram]
"""

import sys

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _find_bar_regions_h(hsv, min_height_frac=0.02):
    """Find horizontal bands of saturated pixels (= horizontal bars)."""
    h, _w = hsv.shape[:2]
    sat = hsv[:, :, 1]
    row_sat_count = np.sum(sat > 15, axis=1)
    regions = []
    in_bar = False
    start = 0
    for y in range(h):
        if row_sat_count[y] > 10 and not in_bar:
            start = y
            in_bar = True
        elif row_sat_count[y] <= 10 and in_bar:
            if y - start > h * min_height_frac:
                regions.append((start, y))
            in_bar = False
    if in_bar and h - start > h * min_height_frac:
        regions.append((start, h))
    return regions


def _find_gap_rows(hsv):
    """Find rows between bars with no saturation (for gridline detection)."""
    sat = hsv[:, :, 1]
    return np.where(np.sum(sat > 15, axis=1) <= 5)[0]


def _find_origin_from_bars(gray, hsv):
    """Find x=0 origin by detecting where bars start (left non-white edge)."""
    bar_regions = _find_bar_regions_h(hsv)
    left_edges = []
    for y_start, y_end in bar_regions:
        margin = max(1, (y_end - y_start) // 4)
        for y in range(y_start + margin, y_end - margin):
            row = gray[y, :]
            non_white = np.where(row < 240)[0]
            if len(non_white) > 0:
                left_edges.append(non_white[0])
    if not left_edges:
        return None
    return int(np.median(left_edges))


def _find_gridlines_and_origin_h(gray, hsv):
    """Find vertical gridlines and origin for horizontal bar charts."""
    bar_regions = _find_bar_regions_h(hsv)
    gap_rows = _find_gap_rows(hsv)
    if bar_regions:
        bar_y_min = bar_regions[0][0]
        bar_y_max = bar_regions[-1][1]
        gap_rows = gap_rows[(gap_rows > bar_y_min) & (gap_rows < bar_y_max)]
    origin_x = _find_origin_from_bars(gray, hsv)
    if len(gap_rows) == 0 or origin_x is None:
        return origin_x, [], None
    avg_gray = np.mean(gray[gap_rows, :], axis=0)
    gl_cols = np.where(avg_gray < 235)[0]
    if origin_x is not None:
        gl_cols = gl_cols[gl_cols > origin_x + 10]
    gridlines = _cluster_1d(gl_cols, gap=5)
    spacings = np.diff(gridlines) if len(gridlines) >= 2 else []
    ppu = float(np.median(spacings)) if len(spacings) > 0 else None
    return origin_x, gridlines, ppu


def _cluster_1d(arr, gap=5):
    """Cluster sorted 1D positions into centers."""
    if len(arr) == 0:
        return []
    groups = []
    cluster = [arr[0]]
    for i in range(1, len(arr)):
        if arr[i] - arr[i - 1] <= gap:
            cluster.append(arr[i])
        else:
            groups.append(int(np.mean(cluster)))
            cluster = [arr[i]]
    groups.append(int(np.mean(cluster)))
    return groups


def _cluster_runs(arr, gap=10):
    """Cluster sorted values into (start, end) ranges."""
    if len(arr) == 0:
        return []
    groups = []
    start = arr[0]
    prev = arr[0]
    for v in arr[1:]:
        if v - prev > gap:
            groups.append((int(start), int(prev)))
            start = v
        prev = v
    groups.append((int(start), int(prev)))
    return groups


def _find_label_centers(gray, x_lo, x_hi, y_lo, y_hi, dark_thresh=100,
                        min_dark=3, cluster_gap=8):
    """Find y-centers of dark text clusters in a vertical label zone."""
    centers = []
    in_cluster = False
    start = 0
    for y in range(y_lo, y_hi):
        dark = np.sum(gray[y, x_lo:x_hi] < dark_thresh)
        if dark >= min_dark and not in_cluster:
            start = y
            in_cluster = True
        elif dark < min_dark and in_cluster:
            centers.append((start + y) // 2)
            in_cluster = False
    return centers


def _calibrate_from_labels(label_ys, label_vals, include_baseline=None):
    """Fit linear y->value mapping from label positions."""
    ys = list(label_ys)
    vs = list(label_vals)
    if include_baseline is not None:
        ys.append(include_baseline[0])
        vs.append(include_baseline[1])
    a, b = np.polyfit(ys, vs, 1)
    return a, b


def _find_x_axis(gray, h, w):
    """Find y-position of x-axis (bottommost row with many dark pixels)."""
    for y in range(h - 1, h // 3, -1):
        dark = np.sum(gray[y, :] < 100)
        if dark > w * 0.3:
            return y
    return h - 1


def _find_y_axis(gray, h, w):
    """Find x-position of y-axis (leftmost column with many dark pixels)."""
    for x in range(0, w // 3):
        dark = np.sum(gray[:, x] < 100)
        if dark > h * 0.3:
            return x
    return 0


# ---------------------------------------------------------------------------
# Horizontal bar charts (histograms)
# ---------------------------------------------------------------------------

def extract_hbar(img):
    """Extract values from a horizontal bar chart using saturation counting."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]

    bar_regions = _find_bar_regions_h(hsv)
    _, _gridlines, pixels_per_unit = _find_gridlines_and_origin_h(gray, hsv)

    if pixels_per_unit is None or len(bar_regions) == 0:
        return []

    values = []
    for y_start, y_end in bar_regions:
        margin = max(1, (y_end - y_start) // 4)
        widths = []
        for y in range(y_start + margin, y_end - margin):
            count = np.sum(sat[y, :] > 15)
            if count > 0:
                widths.append(count)
        if widths:
            bar_width_px = np.median(widths)
            # axis_step unknown without OCR; use pixel_units for now
            values.append(float(bar_width_px / pixels_per_unit))
    return values


# ---------------------------------------------------------------------------
# Vertical bar charts (simple and stacked)
# ---------------------------------------------------------------------------

def _find_bar_cols(hsv, y_lo, y_hi, sat_thresh=30, min_count=20):
    """Find x-ranges of vertical bar groups with adaptive gap detection."""
    sat = hsv[:, :, 1]
    # Exclude legend area (top 15% of plot)
    plot_h = y_hi - y_lo
    legend_cutoff = y_lo + int(plot_h * 0.15)
    col_sat = np.sum(sat[legend_cutoff:y_hi, :] > sat_thresh, axis=0)
    bar_cols = np.where(col_sat > min_count)[0]
    if len(bar_cols) < 2:
        return _cluster_runs(bar_cols, gap=10)
    # Adaptive gap: find the natural break size between bars
    # Gaps between consecutive bar columns
    gaps = np.diff(bar_cols)
    big_gaps = gaps[gaps > 3]  # ignore 1-2px noise
    if len(big_gaps) == 0:
        return _cluster_runs(bar_cols, gap=10)
    # Use half the median big gap as the clustering threshold
    med_gap = np.median(big_gaps)
    cluster_gap = max(5, int(med_gap * 0.6))
    return _cluster_runs(bar_cols, gap=cluster_gap)


def _find_tick_marks(gray, axis_x, x_axis_y, y_start=20):
    """Find y-axis tick mark positions by detecting short horizontal dark lines
    near the y-axis. More robust than scanning for gridlines in the plot area."""
    _h, w = gray.shape
    # Tick marks are short horizontal dark segments just left or right of axis
    # Scan a narrow strip around the axis
    strip_lo = max(0, axis_x - 8)
    strip_hi = min(w, axis_x + 8)

    tick_rows = []
    for y in range(y_start, x_axis_y):
        dark = np.sum(gray[y, strip_lo:strip_hi] < 100)
        if dark >= 4:  # tick mark has several dark pixels
            tick_rows.append(y)

    return _cluster_1d(tick_rows, gap=5)


def _vbar_calibrate(gray, x_axis_y, bar_groups):
    """Calibrate y->value for vertical bar charts.

    Uses multiple strategies in priority order:
    1. Y-axis tick marks (most reliable)
    2. Y-axis label text positions
    3. Plot height with estimated tick count
    Returns (a, b) for value = a*y + b, or None.
    """
    h, w = gray.shape
    y_axis_x = _find_y_axis(gray, h, w)

    # Strategy 1: tick marks near y-axis
    ticks = _find_tick_marks(gray, y_axis_x, x_axis_y)
    # Filter to evenly spaced ticks
    if len(ticks) >= 3:
        spacings = np.diff(ticks)
        med_sp = np.median(spacings)
        if med_sp > 5:  # meaningful spacing
            filtered = [ticks[0]]
            for t in ticks[1:]:
                dist = t - filtered[-1]
                if abs(dist - med_sp) < med_sp * 0.35:
                    filtered.append(t)
                elif abs(dist - 2 * med_sp) < med_sp * 0.35:
                    # Skipped one tick (e.g., obscured by y-axis title)
                    filtered.append(t)
            if len(filtered) >= 3:
                ticks = filtered

    # Strategy 2: label text positions
    label_centers = _find_label_centers(gray, 5, max(5, y_axis_x - 5),
                                        20, min(x_axis_y + 20, h - 1))
    if len(label_centers) >= 3:
        spacings = np.diff(label_centers)
        med_sp = np.median(spacings)
        if med_sp > 5:
            filtered = [label_centers[0]]
            for lc in label_centers[1:]:
                dist = lc - filtered[-1]
                if (abs(dist - med_sp) < med_sp * 0.4
                        or abs(dist - 2 * med_sp) < med_sp * 0.4):
                    filtered.append(lc)
            label_centers = filtered

    # Pick best calibration source
    cal_points = ticks if len(ticks) >= len(label_centers) else label_centers

    # Strategy 3: fallback — use plot area height and estimate ticks
    if len(cal_points) < 2:
        # Assume ~5 evenly spaced ticks across the plot area
        plot_top = 30
        for y in range(20, x_axis_y):
            if np.sum(gray[y, y_axis_x:y_axis_x + 50] < 200) > 3:
                plot_top = y
                break
        n_ticks = 5
        step = (x_axis_y - plot_top) / n_ticks
        cal_points = [int(plot_top + i * step) for i in range(n_ticks + 1)]

    if len(cal_points) < 2:
        return None

    # Assign unit values: top = N, bottom = 0
    n = len(cal_points)
    cal_ys = [*reversed(cal_points), x_axis_y]
    cal_vs = [*range(n, 0, -1), 0]

    a, b = np.polyfit(cal_ys, cal_vs, 1)
    return a, b


def _split_grouped_bar(hsv, x1, x2, x_axis_y):
    """Split a bar group into sub-bars if it contains multiple colors side by side.

    For grouped bar charts, each category has 2-4 bars of different colors.
    Detect by looking at hue changes across the x-range near the bar bottom.
    Returns list of (sub_x1, sub_x2) ranges.
    """
    width = x2 - x1
    if width < 10:
        return [(x1, x2)]

    # Sample hues near the bottom of the bar (where all sub-bars are present)
    sample_y = x_axis_y - max(5, (x_axis_y - 30) // 10)
    strip_hue = hsv[sample_y, x1:x2, 0]
    strip_sat = hsv[sample_y, x1:x2, 1]

    # Only look at saturated pixels
    colored = strip_sat > 40
    if np.sum(colored) < 5:
        return [(x1, x2)]

    # Find hue transitions: where hue changes by >15 between adjacent colored pixels
    colored_xs = np.where(colored)[0]
    hue_vals = strip_hue[colored_xs]

    # Detect boundaries where hue changes significantly
    boundaries = [0]
    for i in range(1, len(hue_vals)):
        if abs(int(hue_vals[i]) - int(hue_vals[i - 1])) > 15:
            boundaries.append(i)
    boundaries.append(len(hue_vals))

    if len(boundaries) <= 2:
        # No hue transitions — single bar
        return [(x1, x2)]

    # Convert boundary indices back to x coordinates
    sub_bars = []
    for j in range(len(boundaries) - 1):
        start_idx = boundaries[j]
        end_idx = boundaries[j + 1] - 1
        sx1 = x1 + colored_xs[start_idx]
        sx2 = x1 + colored_xs[end_idx]
        if sx2 - sx1 >= 3:  # minimum sub-bar width
            sub_bars.append((int(sx1), int(sx2)))

    return sub_bars or [(x1, x2)]


def extract_vbar(img, stacked=False):
    """Extract values from a vertical bar chart."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, w = img.shape[:2]

    x_axis_y = _find_x_axis(gray, h, w)
    bar_groups = _find_bar_cols(hsv, 30, x_axis_y)

    if not bar_groups:
        return []

    cal = _vbar_calibrate(gray, x_axis_y, bar_groups)
    if cal is None:
        return []
    a, b = cal

    def y_to_val(y):
        return a * y + b

    if not stacked:
        # Detect sub-bars within each group (grouped bar charts)
        # Check if groups contain multiple distinct color bars side by side
        all_values = []
        for x1, x2 in bar_groups:
            sub_bars = _split_grouped_bar(hsv, x1, x2, x_axis_y)
            for sx1, sx2 in sub_bars:
                margin = max(1, (sx2 - sx1) // 4)
                for y in range(30, x_axis_y):
                    strip_sat = hsv[y, sx1 + margin:sx2 - margin, 1]
                    if np.sum(strip_sat > 50) >= 2:
                        all_values.append(float(y_to_val(y)))
                        break
        return all_values
    else:
        return _extract_vbar_stacked(img, hsv, gray, bar_groups,
                                     x_axis_y, a, b)


def _extract_vbar_stacked(img, hsv, gray, bar_groups, x_axis_y, a, b):
    """Extract values from a stacked vertical bar chart."""

    def y_to_val(y):
        return a * y + b

    # Detect distinct hues in the bars to identify segments
    # Sample hues from all bars to find the reference hues
    all_hues = []
    for x1, x2 in bar_groups:
        margin = max(3, (x2 - x1) // 4)
        for y in range(x_axis_y - 1, 30, -10):
            strip_sat = hsv[y, x1 + margin:x2 - margin, 1]
            colored = strip_sat > 50
            if np.sum(colored) < 3:
                continue
            strip_hue = hsv[y, x1 + margin:x2 - margin, 0]
            all_hues.append(float(np.median(strip_hue[colored])))

    if not all_hues:
        return []

    # Cluster hues to find distinct segment colors
    all_hues_arr = np.array(sorted(all_hues))
    ref_hues = []
    cluster = [all_hues_arr[0]]
    for h_val in all_hues_arr[1:]:
        if h_val - cluster[-1] > 12:
            ref_hues.append(float(np.median(cluster)))
            cluster = [h_val]
        else:
            cluster.append(h_val)
    ref_hues.append(float(np.median(cluster)))

    values = []
    for x1, x2 in bar_groups:
        margin = 3
        # Find bar top
        bar_top = None
        for y in range(30, x_axis_y):
            strip_sat = hsv[y, x1 + margin:x2 - margin, 1]
            if np.sum(strip_sat > 50) >= 3:
                bar_top = y
                break
        if bar_top is None:
            continue

        # Bottom-up scan: find boundaries between hue segments
        boundaries = [x_axis_y]  # start at baseline
        prev_hue_idx = None
        for y in range(x_axis_y, 30, -1):
            strip_sat = hsv[y, x1 + margin:x2 - margin, 1]
            colored = strip_sat > 50
            if np.sum(colored) < 3:
                continue
            strip_hue = hsv[y, x1 + margin:x2 - margin, 0]
            med_hue = float(np.median(strip_hue[colored]))
            # Find nearest reference hue
            dists = [abs(med_hue - rh) for rh in ref_hues]
            hue_idx = int(np.argmin(dists))
            if (
                prev_hue_idx is not None
                and hue_idx != prev_hue_idx
                and len(boundaries) <= len(ref_hues)
            ):
                boundaries.append(y)
            prev_hue_idx = hue_idx

        # Add bar top
        boundaries.append(bar_top)

        # Calculate segment values from boundaries
        for j in range(len(boundaries) - 1):
            bot_y = boundaries[j]
            top_y = boundaries[j + 1]
            seg_val = y_to_val(top_y) - y_to_val(bot_y)
            values.append(float(abs(seg_val)))

    return values


# ---------------------------------------------------------------------------
# Box plots
# ---------------------------------------------------------------------------

def extract_box(img):
    """Extract values from a box plot."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, w = img.shape[:2]

    x_axis_y = _find_x_axis(gray, h, w)

    # Find gridlines at gap columns
    # First find box locations via orange median lines
    orange_mask = ((hsv[:, :, 2] > 200) & (hsv[:, :, 1] > 80) &
                   (hsv[:, :, 1] < 220) & (hsv[:, :, 0] < 30))
    orange_cols = np.where(np.sum(orange_mask, axis=0) > 0)[0]
    box_groups = _cluster_runs(orange_cols, gap=20)

    if not box_groups:
        return []

    # Calibrate from gridlines
    gl_candidates = []
    gap_xs = []
    for i in range(len(box_groups) - 1):
        gap_xs.append((box_groups[i][1] + box_groups[i + 1][0]) // 2)
    if box_groups:
        gap_xs.append(max(10, box_groups[0][0] - 30))

    for gx in gap_xs:
        if gx >= w:
            continue
        col = gray[:, gx]
        for y in range(30, x_axis_y):
            if 140 < col[y] < 230:
                gl_candidates.append(y)

    gridlines = _cluster_1d(sorted(gl_candidates), gap=8)

    if len(gridlines) < 2:
        # Try label-based calibration
        y_axis_x = _find_y_axis(gray, h, w)
        label_centers = _find_label_centers(gray, 10, y_axis_x, 20, x_axis_y)
        if len(label_centers) >= 3:
            spacings = np.diff(label_centers)
            med_sp = np.median(spacings)
            filtered = [label_centers[0]]
            for lc in label_centers[1:]:
                if abs(lc - filtered[-1] - med_sp) < med_sp * 0.4:
                    filtered.append(lc)
            gridlines = filtered

    if len(gridlines) < 2:
        return []

    # Assign unit-step values (reversed: top=high, bottom=low)
    n = len(gridlines)
    cal_ys = [*reversed(gridlines), x_axis_y]
    cal_vs = [*range(n, 0, -1), 0]
    a, b = np.polyfit(cal_ys, cal_vs, 1)

    def y_to_val(y):
        return a * y + b

    values = []
    for ox1, ox2 in box_groups:
        search_x1 = max(0, ox1 - 20)
        search_x2 = min(w, ox2 + 20)
        box_width = ox2 - ox1

        # Find horizontal dark lines in/near this box
        # Classify by width: box edges span full box width,
        # whisker caps span ~half, gridlines span the whole plot
        dark_features = []
        for y in range(30, x_axis_y):
            row = gray[y, ox1:ox2]
            dark_count = np.sum(row < 80)
            if dark_count > box_width * 0.25:
                # Check span: how wide is the dark feature?
                dark_positions = np.where(gray[y, search_x1:search_x2] < 80)[0]
                if len(dark_positions) < 2:
                    continue
                span = dark_positions[-1] - dark_positions[0]
                # Gridlines span the full plot width; box features are localized
                plot_width = w - _find_y_axis(gray, h, w)
                if span < plot_width * 0.6:  # reject plot-wide gridlines
                    dark_features.append(y)

        # Cluster into distinct features (Q1, Q3, whisker caps)
        feature_clusters = _cluster_1d(dark_features, gap=3)

        for fc in feature_clusters:
            values.append(float(y_to_val(fc)))

        # Median: orange line
        orange_rows = np.where(np.any(orange_mask[30:x_axis_y, ox1:ox2],
                                      axis=1))[0] + 30
        if len(orange_rows) > 0:
            values.append(float(y_to_val(np.median(orange_rows))))

        # Outliers: red dots
        red_mask = ((img[30:x_axis_y, search_x1:search_x2, 2] > 200) &
                    (img[30:x_axis_y, search_x1:search_x2, 1] < 100) &
                    (img[30:x_axis_y, search_x1:search_x2, 0] < 100))
        red_rows = np.where(np.any(red_mask, axis=1))[0] + 30
        if len(red_rows) > 0:
            red_groups = _cluster_runs(red_rows, gap=10)
            for ry1, ry2 in red_groups:
                values.append(float(y_to_val((ry1 + ry2) / 2)))

    # Deduplicate nearby values
    values.sort()
    deduped = []
    for v in values:
        if not deduped or abs(v - deduped[-1]) > 0.02:
            deduped.append(v)
    return deduped


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

CHART_TYPE_MAP = {
    "histogram": "vbar",       # most ChartX histograms are vertical bars
    "histogram_hbar": "hbar",  # explicit horizontal override
    "bar_chart": "vbar",
    "bar_chart_num": "vbar",
    "box": "box",
}


def extract_values(image_path, chart_type=None, stacked_hint=False):
    """Extract numeric values from a chart image.

    Args:
        image_path: Path to PNG image.
        chart_type: One of "histogram", "bar_chart", "bar_chart_num", "box",
                    or the internal names "hbar", "vbar", "box".
        stacked_hint: If True, treat vertical bars as stacked.

    Returns:
        List of extracted float values (in pixel-calibrated units).
        Without OCR, values are in gridline-step units, not absolute.
        Use find_best_scale() to calibrate against ground truth.
    """
    img = cv2.imread(str(image_path))
    if img is None:
        return []

    ctype = (
        None if chart_type is None else CHART_TYPE_MAP.get(chart_type, chart_type)
    )

    if ctype == "hbar":
        return extract_hbar(img)
    elif ctype == "vbar":
        return extract_vbar(img, stacked=stacked_hint)
    elif ctype == "box":
        return extract_box(img)
    else:
        return []


def find_best_scale(extracted, ground_truth, tol=0.05):
    """Find the scale factor k that maximizes recall: ex*k ~ gt.

    Tries every candidate k = gt_val / ex_val for large values,
    picks the one maximizing recall.
    """
    if not extracted or not ground_truth:
        return 1.0

    ex = [v for v in extracted if abs(v) > 1e-6]
    gt = list(ground_truth)
    if not ex:
        return 1.0

    # Generate candidate scale factors
    candidates = set()
    for g in gt:
        for e in ex:
            candidates.add(g / e)

    best_k = 1.0
    best_recall = -1

    for k in candidates:
        scaled = {e * k for e in ex}
        matched = 0
        for g in gt:
            for s in scaled:
                if g == 0:
                    if abs(s) < 0.01:
                        matched += 1
                        break
                elif abs(s - g) / abs(g) <= tol:
                    matched += 1
                    break
        recall = matched / len(gt)
        if recall > best_recall:
            best_recall = recall
            best_k = k

    return best_k


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <image_path> [--type TYPE]")
        sys.exit(1)

    img_path = sys.argv[1]
    ctype = None
    stacked = False
    for i, arg in enumerate(sys.argv[2:], 2):
        if arg == "--type" and i + 1 < len(sys.argv):
            ctype = sys.argv[i + 1]
        if arg == "--stacked":
            stacked = True

    vals = extract_values(img_path, chart_type=ctype, stacked_hint=stacked)
    print(f"Extracted {len(vals)} values: {vals}")
