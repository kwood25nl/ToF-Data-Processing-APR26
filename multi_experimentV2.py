import os
import re
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors

from scipy import stats as sp_stats

# ── Colours (match Visualize.py) ─────────────────────────────────────────────
ALWAYS_VALID_COLOR = "#52A83C"
MOST_VALID_COLOR   = "#F6C6A7"
LEAST_VALID_COLOR  = "#E97122"
NO_DATA_COLOR      = "#D9D9D9"
PALETTE = ["#E8708E", "#EDD020", "#E97122", "#52A83C",
           "#18A5AA", "#55A9D4", "#2E61A8"]

# Validity plot colors (match Visualize.py)
VALID_COLOR     = "#52A83C"
INVALID_COLOR   = "#E97122"
ALWAYS_VALID_BG = "#BADCB1"

# Box colors (match Visualize.py)
BOX_WITHIN = "#52A83C"
BOX_OVER   = "#2E61A8"
BOX_LOW    = "#E8708E"


# ── Data loading (same pattern as multi_experiment.py) ────────────────────────
from Calculations import trim_tof_data
from ImportData import load_tof_data


def setup_output_folder(output_path: str, label: str) -> str:
    """Create an output folder (non-throwing if it already exists)."""
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    full_path = os.path.join(output_path, f"plots-{label}-{timestamp}")
    os.makedirs(full_path, exist_ok=True)
    return full_path


def load_multi_experiment(experiment_folders: dict, origin: int) -> dict:
    """
    Loads and trims multiple experiments.

    Args:
        experiment_folders: dict of {experiment_name: folder_path}
        origin:             shared origin distance for all experiments

    Returns:
        dict of {experiment_name: (trimmed_data, n_frames, shortest_key)}
    """
    experiments = {}
    for name, folder in experiment_folders.items():
        print(f"Loading experiment: {name}")
        raw = load_tof_data(folder, origin)
        trimmed, n_frames, shortest = trim_tof_data(raw)
        print(f"  {name}: {n_frames} frames, shortest dataset: {shortest}")
        experiments[name] = (trimmed, n_frames, shortest)
    return experiments


# ── Shared helpers (ported from Visualize.py conventions) ─────────────────────
def extract_true_distance(key: str) -> float:
    """Extracts the numeric true distance from a distance key (matches Visualize.py)."""
    digits = re.findall(r"-?\d+", key)
    return float("".join(digits))


def zone_to_grid(z: int):
    """
    For subplot arrays (axes[row][col]) where row 0 is the TOP:
    flip physical row so Z0 bottom-left and Z63 top-right.
    """
    physical_row = z // 8
    col = z % 8
    row = 7 - physical_row
    return row, col


def _safe_concat(arr_list):
    arr_list = [a for a in arr_list if a is not None and len(a) > 0]
    return np.concatenate(arr_list) if arr_list else None


def _interp_rgb(rgb_a, rgb_b, t: float):
    return tuple(rgb_a[i] * (1 - t) + rgb_b[i] * t for i in range(3))

def _lighten_rgb(rgb, amount: float = 0.5):
    """
    Mix color with white.
    amount=0   -> original
    amount=1.0 -> white
    """
    r, g, b = rgb
    return (r + (1 - r) * amount, g + (1 - g) * amount, b + (1 - b) * amount)


# ── (1) Overall validity heatmap (all distances + experiments concatenated) ───
def compute_zone_validity_percent_concat(experiments: dict) -> dict:
    """
    Computes percent-valid per zone by concatenating is_valid_range across:
      - all experiments
      - all distances
      - all frames

    Returns:
        zone_pct_valid: dict of zone -> percent valid (0..100) or None if no data
    """
    ones_count = {z: 0 for z in range(64)}
    total_count = {z: 0 for z in range(64)}

    for _exp_name, (trimmed, _n_frames, _shortest) in experiments.items():
        for _dist_key, zone_data in trimmed.items():
            for z in range(64):
                arr = zone_data.get("is_valid_range", {}).get(z)
                if arr is None:
                    continue

                a = np.asarray(arr)
                mask = (a == 0) | (a == 1)
                if not np.any(mask):
                    continue

                a01 = a[mask]
                total_count[z] += int(a01.size)
                ones_count[z] += int(np.sum(a01 == 1))

    zone_pct_valid = {}
    for z in range(64):
        zone_pct_valid[z] = (100.0 * ones_count[z] / total_count[z]) if total_count[z] > 0 else None

    return zone_pct_valid


def plot_multiexp_validity_heatmap(experiments: dict, output_path: str):
    """
    Single 8x8 heatmap showing validity across ALL experiments and distances
    (concatenated is_valid_range), per zone.

    Orientation: patch-based, no flip => Z0 bottom-left, Z63 top-right.
    """
    zone_pct_valid = compute_zone_validity_percent_concat(experiments)

    invalid_vals = [v for v in zone_pct_valid.values() if v is not None and v < 100.0]
    min_valid = min(invalid_vals) if invalid_vals else 0.0
    max_valid = 100.0

    fig, ax = plt.subplots(figsize=(8, 8))
    fig.suptitle("Overall Zone Validity \n(All Experiments Concatenated)", fontsize=12)

    for z in range(64):
        r = z // 8
        c = z % 8
        plot_r = r
        pct = zone_pct_valid.get(z)

        if pct is None:
            color = (*mcolors.to_rgb(NO_DATA_COLOR), 1.0)
            text = "No\nData"
            txt_color = "black"
        elif pct >= 100.0:
            color = (*mcolors.to_rgb(ALWAYS_VALID_COLOR), 1.0)
            text = "Always\nValid"
            txt_color = "white"
        else:
            t = (pct - min_valid) / (max_valid - min_valid + 1e-9)
            color_a = mcolors.to_rgb(LEAST_VALID_COLOR)
            color_b = mcolors.to_rgb(MOST_VALID_COLOR)
            rgb = _interp_rgb(color_a, color_b, t)
            color = (*rgb, 1.0)
            text = f"{pct:.1f}%"
            txt_color = "black"

        rect = mpatches.FancyBboxPatch(
            (c, plot_r), 1, 1,
            boxstyle="square,pad=0",
            facecolor=color, edgecolor="white", linewidth=0.5
        )
        ax.add_patch(rect)
        ax.text(c + 0.5, plot_r + 0.5, text,
                ha="center", va="center", fontsize=7,
                color=txt_color, fontweight="bold")

    ax.set_xlim(0, 8)
    ax.set_ylim(0, 8)
    ax.set_yticks([])
    ax.set_aspect("equal")

    plt.tight_layout()
    save_path = os.path.join(output_path, "MultiExperiment_Validity_Heatmap_MultiExperiment.png")
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close(fig)


# ── (2) Error & validity plot (multi-experiment) ──────────────────────────────
def plot_multiexp_error_and_validity(experiments: dict, output_path: str):
    """
    Multi-experiment version of Visualize.plot_error_and_validity.

    Per distance:
      - Boxplot data: concatenated error samples across ALL experiments,
        but ONLY from zones that are always valid for that distance within that
        experiment (is_valid_range == 1 for all frames in that zone array).
      - Validity line: for that distance, concatenate is_valid_range samples
        across ALL experiments and zones, compute percent valid.

    Saves: MultiExperiment_Error_Validity_MultiExperiment.png
    """
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D

    all_dist_keys = set()
    for _name, (trimmed, _n_frames, _shortest) in experiments.items():
        all_dist_keys.update(trimmed.keys())

    sorted_dist_keys = sorted(all_dist_keys, key=extract_true_distance)
    if not sorted_dist_keys:
        return

    dist_labels = []
    error_arrays = []
    mean_errors = []
    validity_pcts = []
    pct_errors = []
    q1_list, q3_list = [], []

    for dist_key in sorted_dist_keys:
        true_distance = extract_true_distance(dist_key)
        dist_labels.append(dist_key)

        per_dist_error_chunks = []
        valid_ones = 0
        valid_total = 0

        for _exp_name, (trimmed, _n_frames, _shortest) in experiments.items():
            if dist_key not in trimmed:
                continue
            zone_data = trimmed[dist_key]

            # validity pool
            for z in range(64):
                v = zone_data.get("is_valid_range", {}).get(z)
                if v is None:
                    continue
                a = np.asarray(v)
                mask = (a == 0) | (a == 1)
                if not np.any(mask):
                    continue
                a01 = a[mask]
                valid_total += int(a01.size)
                valid_ones += int(np.sum(a01 == 1))

            # error pool (always-valid zones only)
            for z in range(64):
                arr = zone_data.get("distance_mm", {}).get(z)
                valid_arr = zone_data.get("is_valid_range", {}).get(z)
                if arr is None or valid_arr is None:
                    continue

                v = np.asarray(valid_arr)
                mask01 = (v == 0) | (v == 1)
                if not np.any(mask01):
                    continue
                v01 = v[mask01]

                if np.any(v01 == 0):
                    continue  # not always-valid

                a = np.asarray(arr)
                a = a[:len(v)]
                per_dist_error_chunks.append(a - true_distance)

        error = _safe_concat(per_dist_error_chunks)
        if error is None:
            error = np.array([0.0])

        error_arrays.append(error)

        mean_err = float(np.mean(error))
        mean_errors.append(mean_err)

        pct_errors.append(100.0 * abs(mean_err) / true_distance if true_distance != 0 else 0.0)
        validity_pcts.append(100.0 * valid_ones / valid_total if valid_total > 0 else 0.0)

        q1_list.append(float(np.percentile(error, 25)))
        q3_list.append(float(np.percentile(error, 75)))

    fig, ax1 = plt.subplots(figsize=(12, 7))
    ax2 = ax1.twinx()

    ax1.yaxis.set_minor_locator(plt.MultipleLocator(0.2))
    ax1.grid(True, which="major", color="grey", linewidth=0.5, alpha=0.3, zorder=0)
    ax1.grid(True, which="minor", color="grey", linewidth=0.3, alpha=0.2, zorder=0)

    ax1.set_zorder(ax2.get_zorder() + 1)
    ax1.patch.set_visible(False)

    x_positions = np.arange(len(dist_labels))

    ax2.fill_between(x_positions, 0, validity_pcts, color="#A3DBDD", alpha=0.5)
    ax2.plot(x_positions, validity_pcts, color="#18A5AA", linewidth=2.0)
    ax2.plot(x_positions, pct_errors, color="#E97122", linewidth=2.0)

    ax1.axhline(0, color="black", linewidth=1.0, linestyle=":", alpha=0.7)

    for i, (err_arr, mean_err) in enumerate(zip(error_arrays, mean_errors)):
        if abs(mean_err) <= 2:
            color = BOX_WITHIN
        elif mean_err < -2:
            color = BOX_LOW
        else:
            color = BOX_OVER

        ax1.boxplot(err_arr,
                    positions=[x_positions[i]],
                    widths=0.5,
                    patch_artist=True,
                    manage_ticks=False,
                    medianprops=dict(color="white", linewidth=1.5),
                    whiskerprops=dict(color=color, linewidth=1.2),
                    capprops=dict(color=color, linewidth=1.2),
                    flierprops=dict(marker=".", markersize=3,
                                    markerfacecolor=color, alpha=0.25),
                    boxprops=dict(facecolor=color, color=color,
                                  linewidth=1.2))

    if q1_list and q3_list:
        global_q1 = float(np.min(q1_list))
        global_q3 = float(np.max(q3_list))
        global_iqr = max(global_q3 - global_q1, 1e-6)
        pad = 1.5 * global_iqr
        y_min = min(global_q1 - pad, -0.5)
        y_max = max(global_q3 + pad, 0.5)
        ax1.set_ylim(y_min, y_max)

    ax1.set_ylabel("Error [mm]", fontsize=10)
    ax1.set_xticks(x_positions)
    ax1.set_xticklabels(dist_labels, fontsize=9)
    ax1.set_xlabel("Distance [mm]", fontsize=10)
    ax1.set_title("Multi-Experiment Error and Zone Validity", fontsize=12)

    ax2.set_ylabel("% Valid (concatenated)  /  % Error", fontsize=10)

    legend_elements = [
        Line2D([0], [0], color="#18A5AA", linewidth=2.0,
               label="% valid (all experiments/zones concatenated)"),
        Line2D([0], [0], color="#E97122", linewidth=2.0,
               label="% Error"),
        Patch(facecolor=BOX_WITHIN, label="Mean |error| ≤ 2mm"),
        Patch(facecolor=BOX_LOW, label="Mean error < −2mm"),
        Patch(facecolor=BOX_OVER, label="Mean error > +2mm"),
    ]

    fig.subplots_adjust(right=0.78)
    ax1.legend(handles=legend_elements, fontsize=8,
               loc="upper left", bbox_to_anchor=(1.01, 1.0),
               borderaxespad=0.0, framealpha=0.95)

    plt.tight_layout()
    save_path = os.path.join(output_path, "MultiExperiment_Error_Validity_MultiExperiment.png")
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close(fig)


# ── (3) Overall error heatmaps (all distances + experiments pooled; valid only) ─
def compute_multiexp_zone_errors_concat(experiments: dict) -> tuple[dict, dict]:
    """
    Pools valid-only signed error samples per zone across ALL experiments and ALL distances.

    Returns:
        abs_errors:  zone -> mean absolute error
        bias_errors: zone -> mean bias (signed) error
    """
    zone_signed = {z: [] for z in range(64)}

    for _exp_name, (trimmed, _n_frames, _shortest) in experiments.items():
        for dist_key, zone_data in trimmed.items():
            true_distance = extract_true_distance(dist_key)

            for z in range(64):
                arr = zone_data.get("distance_mm", {}).get(z)
                v = zone_data.get("is_valid_range", {}).get(z)
                if arr is None or v is None:
                    continue

                arr = np.asarray(arr)
                v = np.asarray(v)

                mask01 = (v == 0) | (v == 1)
                if not np.any(mask01):
                    continue

                v01 = v[mask01]
                arr01 = arr[:len(v)][mask01]

                mask_valid = (v01 == 1)
                if not np.any(mask_valid):
                    continue

                signed_err = arr01[mask_valid] - true_distance
                zone_signed[z].append(signed_err)

    abs_errors, bias_errors = {}, {}
    for z in range(64):
        pooled = _safe_concat(zone_signed[z])
        if pooled is None or len(pooled) == 0:
            abs_errors[z] = None
            bias_errors[z] = None
        else:
            bias_errors[z] = float(np.mean(pooled))
            abs_errors[z] = float(np.mean(np.abs(pooled)))

    return abs_errors, bias_errors


def _bias_colour(bias: float, min_bias: float, max_bias: float) -> tuple:
    extreme = max(abs(min_bias), abs(max_bias))
    threshold = extreme * 0.2

    if abs(bias) <= threshold:
        base = mcolors.to_rgb("#52A83C")
        alpha = 1.0 - 0.5 * (abs(bias) / (threshold + 1e-9))
    elif bias > threshold:
        base = mcolors.to_rgb("#2E61A8")
        alpha = 0.5 + 0.5 * (bias / (max_bias + 1e-9))
    else:
        base = mcolors.to_rgb("#E8708E")
        alpha = 0.5 + 0.5 * (abs(bias) / (abs(min_bias) + 1e-9))

    return (*base, float(np.clip(alpha, 0.5, 1.0)))


def _draw_error_heatmap(ax, grid_shape: tuple, labels: list, cell_abs: list, cell_bias: list):
    rows, cols = grid_shape
    valid_bias = [v for v in cell_bias if v is not None]
    min_bias = min(valid_bias) if valid_bias else 0.0
    max_bias = max(valid_bias) if valid_bias else 0.0

    for i, (label, abs_val, bias_val) in enumerate(zip(labels, cell_abs, cell_bias)):
        r = i // cols
        c = i % cols

        color = _bias_colour(bias_val, min_bias, max_bias) if bias_val is not None else (0.85, 0.85, 0.85, 1.0)

        rect = mpatches.FancyBboxPatch(
            (c, r), 1, 1,
            boxstyle="square,pad=0",
            facecolor=color, edgecolor="white", linewidth=0.5,
            transform=ax.transData
        )
        ax.add_patch(rect)

        abs_text = f"{abs_val:.1f}" if abs_val is not None else "N/A"
        bias_text = f"b:{bias_val:+.1f}" if bias_val is not None else ""
        ax.text(c + 0.5, r + 0.5, f"{label}\n{abs_text}\n{bias_text}",
                ha="center", va="center", fontsize=5.5,
                color="white", fontweight="bold")

    ax.set_xlim(0, cols)
    ax.set_ylim(0, rows)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")


def plot_multiexp_error_heatmaps(experiments: dict, output_path: str):
    from matplotlib.patches import Patch

    def get_ring(z):
        r = z // 8
        c = z % 8
        return min(r, 7 - r, c, 7 - c)

    abs_errors, bias_errors = compute_multiexp_zone_errors_concat(experiments)

    fig, axes = plt.subplots(2, 2, figsize=(16, 16))
    fig.suptitle("Multi-Experiment Error Heatmaps\n(text = mean absolute error, colour = bias)",
                 fontsize=14, y=1.002)

    # 8x8
    ax = axes[0][0]
    ax.set_title("8×8 Zone Errors", fontsize=10)
    labels = [f"Z{z}" for z in range(64)]
    cell_abs = [abs_errors.get(z) for z in range(64)]
    cell_bias = [bias_errors.get(z) for z in range(64)]
    _draw_error_heatmap(ax, (8, 8), labels, cell_abs, cell_bias)

    # 4x4
    ax = axes[0][1]
    ax.set_title("4×4 Group Errors", fontsize=10)
    lb, ca, cb = [], [], []
    for sg in range(16):
        sr, sc = sg // 4, sg % 4
        abs_vals, bias_vals = [], []
        for dr in range(2):
            for dc in range(2):
                z = (sr * 2 + dr) * 8 + (sc * 2 + dc)
                if abs_errors.get(z) is not None:
                    abs_vals.append(abs_errors[z])
                if bias_errors.get(z) is not None:
                    bias_vals.append(bias_errors[z])
        lb.append(f"G{sg}")
        ca.append(float(np.mean(abs_vals)) if abs_vals else None)
        cb.append(float(np.mean(bias_vals)) if bias_vals else None)
    _draw_error_heatmap(ax, (4, 4), lb, ca, cb)

    # 2x2
    ax = axes[1][0]
    ax.set_title("2×2 Group Errors", fontsize=10)
    lb, ca, cb = [], [], []
    for sg in range(4):
        sr, sc = sg // 2, sg % 2
        abs_vals, bias_vals = [], []
        for dr in range(4):
            for dc in range(4):
                z = (sr * 4 + dr) * 8 + (sc * 4 + dc)
                if abs_errors.get(z) is not None:
                    abs_vals.append(abs_errors[z])
                if bias_errors.get(z) is not None:
                    bias_vals.append(bias_errors[z])
        lb.append(f"G{sg}")
        ca.append(float(np.mean(abs_vals)) if abs_vals else None)
        cb.append(float(np.mean(bias_vals)) if bias_vals else None)
    _draw_error_heatmap(ax, (2, 2), lb, ca, cb)

    # Rings
    ax = axes[1][1]
    ax.set_title("Ring Group Errors", fontsize=10)

    ring_abs = {0: [], 1: [], 2: [], 3: []}
    ring_bias = {0: [], 1: [], 2: [], 3: []}
    for z in range(64):
        ring = get_ring(z)
        if abs_errors.get(z) is not None:
            ring_abs[ring].append(abs_errors[z])
        if bias_errors.get(z) is not None:
            ring_bias[ring].append(bias_errors[z])

    ring_abs_means = {r: (float(np.mean(v)) if v else None) for r, v in ring_abs.items()}
    ring_bias_means = {r: (float(np.mean(v)) if v else None) for r, v in ring_bias.items()}

    vb = [v for v in ring_bias_means.values() if v is not None]
    min_bias = min(vb) if vb else 0.0
    max_bias = max(vb) if vb else 0.0

    for z in range(64):
        r = z // 8
        c = z % 8
        ring = get_ring(z)
        bias_val = ring_bias_means.get(ring)
        color = _bias_colour(bias_val, min_bias, max_bias) if bias_val is not None else (0.85, 0.85, 0.85, 1.0)

        rect = mpatches.FancyBboxPatch((c, r), 1, 1, boxstyle="square,pad=0",
                                       facecolor=color, edgecolor="white", linewidth=0.5)
        ax.add_patch(rect)

        # label once per ring at top-left of ring k: (c=k, r=7-k)
        label_once = (c == ring) and (r == (7 - ring))
        if label_once:
            abs_val = ring_abs_means.get(ring)
            abs_text = f"{abs_val:.1f}" if abs_val is not None else "N/A"
            bias_text = f"b:{bias_val:+.1f}" if bias_val is not None else ""
            ax.text(c + 0.5, r + 0.5, f"{abs_text}\n{bias_text}",
                    ha="center", va="center", fontsize=7,
                    color="white", fontweight="bold")

    for ring in range(1, 4):
        o = ring
        ax.plot([o, 8 - o, 8 - o, o, o],
                [o, o, 8 - o, 8 - o, o],
                color="white", linewidth=1.5, linestyle="--")

    legend_elements = [
        Patch(facecolor=BOX_WITHIN, label="Mean |error| ≤ 2mm"),
        Patch(facecolor=BOX_LOW, label="Mean error < −2mm"),
        Patch(facecolor=BOX_OVER, label="Mean error > +2mm"),
    ]
    fig.legend(handles=legend_elements, loc="upper center", ncol=3,
               framealpha=0.98, fontsize=9, bbox_to_anchor=(0.5, 0.97))

    ax.set_xlim(0, 8)
    ax.set_ylim(0, 8)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")

    plt.tight_layout(rect=[0, 0, 1, 0.98])
    save_path = os.path.join(output_path, "MultiExperiment_Error_Heatmaps_MultiExperiment.png")
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close(fig)


# ── (4) Per-distance heatmaps across all experiments (valid-only pooled) ──────
def colour_for_value(value: float, true_distance: float, min_val: float, max_val: float):
    """
    Ported from multi_experiment.py: green within ±2mm, pink below, blue above, with alpha scaling.
    """
    diff = value - true_distance
    if abs(diff) <= 2:
        base = mcolors.to_rgb("#52A83C")
        extent = max(abs(min_val - true_distance), abs(max_val - true_distance), 2)
        alpha = 1.0 - 0.5 * (abs(diff) / extent)
    elif diff < -2:
        base = mcolors.to_rgb("#E8708E")
        most_below = min(min_val - true_distance, -2)
        alpha = 1.0 - 0.5 * ((diff - most_below) / (-2 - most_below + 1e-9))
        alpha = np.clip(alpha, 0.5, 1.0)
    else:
        base = mcolors.to_rgb("#2E61A8")
        most_above = max(max_val - true_distance, 2)
        alpha = 1.0 - 0.5 * ((most_above - diff) / (most_above - 2 + 1e-9))
        alpha = np.clip(alpha, 0.5, 1.0)
    return (*base, float(alpha))


def draw_heatmap(ax, means: dict, true_distance: float, grid_shape, zone_labels, cell_means):
    """
    Patch-based heatmap like multi_experiment.py draw_heatmap.
    Orientation: NO flip => Z0 bottom-left, Z63 top-right.
    """
    rows, cols = grid_shape
    all_vals = [v for v in means.values() if v is not None]
    min_val = min(all_vals) if all_vals else true_distance
    max_val = max(all_vals) if all_vals else true_distance

    for i, (label, mean_val) in enumerate(zip(zone_labels, cell_means)):
        r = i // cols
        c = i % cols
        color = colour_for_value(mean_val, true_distance, min_val, max_val) if mean_val is not None else (0.85, 0.85, 0.85, 1.0)

        rect = mpatches.FancyBboxPatch(
            (c, r), 1, 1,
            boxstyle="square,pad=0",
            facecolor=color, edgecolor="white", linewidth=0.5,
            transform=ax.transData
        )
        ax.add_patch(rect)

        text_val = f"{mean_val:+.1f}" if mean_val is not None else "N/A"
        ax.text(c + 0.5, r + 0.5, f"{label}\n{text_val}",
                ha="center", va="center", fontsize=6,
                color="white", fontweight="bold")

    ax.set_xlim(0, cols)
    ax.set_ylim(0, rows)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")


def _compute_multiexp_zone_mean_errors_for_distance(experiments: dict, dist_key: str) -> dict:
    """
    For a single dist_key:
      - pool valid-only measured distances across experiments per zone
      - compute mean measured distance per zone
      - return mean signed error per zone (mean_measured - true_distance)
    """
    true_distance = extract_true_distance(dist_key)
    per_zone_valid = {z: [] for z in range(64)}

    for _exp_name, (trimmed, _n_frames, _shortest) in experiments.items():
        if dist_key not in trimmed:
            continue
        zone_data = trimmed[dist_key]

        for z in range(64):
            arr = zone_data.get("distance_mm", {}).get(z)
            v = zone_data.get("is_valid_range", {}).get(z)
            if arr is None or v is None:
                continue

            arr = np.asarray(arr)
            v = np.asarray(v)

            mask01 = (v == 0) | (v == 1)
            if not np.any(mask01):
                continue

            v01 = v[mask01]
            arr01 = arr[:len(v)][mask01]
            mask_valid = (v01 == 1)
            if not np.any(mask_valid):
                continue

            per_zone_valid[z].append(arr01[mask_valid])

    zone_mean_error = {}
    for z in range(64):
        pooled = _safe_concat(per_zone_valid[z])
        if pooled is None:
            zone_mean_error[z] = None
        else:
            zone_mean_error[z] = float(np.mean(pooled) - true_distance)

    return zone_mean_error


def plot_multiexp_per_distance_heatmaps(experiments: dict, output_path: str):
    """
    For each distance key, generate a 4-panel heatmap figure (8x8, 4x4, 2x2, rings)
    using pooled valid-only data across experiments.

    Output filename:
      {dist_key}_Heatmaps_MultiExperiment.png
    """
    def get_ring(z):
        r = z // 8
        c = z % 8
        return min(r, 7 - r, c, 7 - c)

    all_dist_keys = set()
    for _exp_name, (trimmed, _n_frames, _shortest) in experiments.items():
        all_dist_keys.update(trimmed.keys())

    for dist_key in sorted(all_dist_keys, key=extract_true_distance):
        true_distance = extract_true_distance(dist_key)
        zone_err = _compute_multiexp_zone_mean_errors_for_distance(experiments, dist_key)

        fig, axes = plt.subplots(2, 2, figsize=(16, 16))
        fig.suptitle(f"{dist_key} — Heatmaps (Multi-Experiment, Valid-only)", fontsize=14, y=1.002)

        # 8x8
        ax = axes[0][0]
        ax.set_title("8×8", fontsize=10)
        draw_heatmap(
            ax,
            zone_err,
            0.0,
            (8, 8),
            [f"Z{z}" for z in range(64)],
            [zone_err.get(z) for z in range(64)]
        )

        # 4x4 (average errors)
        ax = axes[0][1]
        ax.set_title("4×4", fontsize=10)
        gm, lb, cm = {}, [], []
        for sg in range(16):
            sr, sc = sg // 4, sg % 4
            vals = [
                zone_err[z]
                for dr in range(2) for dc in range(2)
                if (z := (sr * 2 + dr) * 8 + (sc * 2 + dc)) in zone_err
                and zone_err[z] is not None
            ]
            mv = float(np.mean(vals)) if vals else None
            gm[sg] = mv
            lb.append(f"G{sg}")
            cm.append(mv)
        draw_heatmap(ax, gm, 0.0, (4, 4), lb, cm)

        # 2x2
        ax = axes[1][0]
        ax.set_title("2×2", fontsize=10)
        gm, lb, cm = {}, [], []
        for sg in range(4):
            sr, sc = sg // 2, sg % 2
            vals = [
                zone_err[z]
                for dr in range(4) for dc in range(4)
                if (z := (sr * 4 + dr) * 8 + (sc * 4 + dc)) in zone_err
                and zone_err[z] is not None
            ]
            mv = float(np.mean(vals)) if vals else None
            gm[sg] = mv
            lb.append(f"G{sg}")
            cm.append(mv)
        draw_heatmap(ax, gm, 0.0, (2, 2), lb, cm)

        # Rings
        ax = axes[1][1]
        ax.set_title("Rings", fontsize=10)

        rz = {0: [], 1: [], 2: [], 3: []}
        for z, mv in zone_err.items():
            if mv is None:
                continue
            rz[get_ring(z)].append(mv)
        rm = {r: (float(np.mean(v)) if v else None) for r, v in rz.items()}

        # draw 8x8 ring-mapped heatmap (color by ring mean error)
        means = {z: rm.get(get_ring(z)) for z in range(64)}
        draw_heatmap(
            ax,
            means,
            0.0,
            (8, 8),
            ["" for _ in range(64)],
            [means.get(z) for z in range(64)]
        )
        for ring in range(1, 4):
            o = ring
            ax.plot([o, 8-o, 8-o, o, o], [o, o, 8-o, 8-o, o],
                    color="white", linewidth=1.5, linestyle="--")

        plt.tight_layout(rect=[0, 0, 1, 0.98])
        save_path = os.path.join(output_path, f"{dist_key}_Heatmaps_MultiExperiment.png")
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        plt.close(fig)


# ── (5) Per-distance tiled boxplots pooled across experiments (include invalid) ─
def _pooled_zone_arrays_for_distance(experiments: dict, dist_key: str) -> tuple[dict, dict]:
    """
    For a single distance:
      - pooled_dist[z] = concat of distance_mm[z] across experiments (ALL samples, valid+invalid)
      - pooled_valid[z] = concat of is_valid_range[z] across experiments (0/1 validity stream)

    Returns:
        pooled_dist, pooled_valid
    """
    pooled_dist = {z: [] for z in range(64)}
    pooled_valid = {z: [] for z in range(64)}

    for _exp_name, (trimmed, _n_frames, _shortest) in experiments.items():
        if dist_key not in trimmed:
            continue
        zone_data = trimmed[dist_key]

        for z in range(64):
            arr = zone_data.get("distance_mm", {}).get(z)
            v = zone_data.get("is_valid_range", {}).get(z)

            if arr is not None:
                pooled_dist[z].append(np.asarray(arr))
            if v is not None:
                a = np.asarray(v)
                mask01 = (a == 0) | (a == 1)
                if np.any(mask01):
                    pooled_valid[z].append(a[mask01])

    pooled_dist = {z: _safe_concat(pooled_dist[z]) for z in range(64)}
    pooled_valid = {z: _safe_concat(pooled_valid[z]) for z in range(64)}
    return pooled_dist, pooled_valid


def plot_multiexp_tiled_boxplots(experiments: dict, output_path: str):
    """
    One 8x8 tiled boxplot figure per distance, pooled across experiments per zone.
    Includes invalid measurements in the boxplot data.
    Shades background (#F6C6A7) if zone is ever invalid in pooled validity stream.
    """
    all_dist_keys = set()
    for _exp_name, (trimmed, _n_frames, _shortest) in experiments.items():
        all_dist_keys.update(trimmed.keys())

    for dist_key in sorted(all_dist_keys, key=extract_true_distance):
        true_distance = extract_true_distance(dist_key)

        pooled_dist, pooled_valid = _pooled_zone_arrays_for_distance(experiments, dist_key)

        fig, axes = plt.subplots(8, 8, figsize=(24, 24))
        fig.suptitle(f"Distance Boxplots (Multi-Experiment) — {dist_key}", fontsize=14, y=1.002)

        for z in range(64):
            row, col = zone_to_grid(z)
            ax = axes[row][col]

            arr = pooled_dist.get(z)
            v = pooled_valid.get(z)

            # shade if any invalid appears
            if v is not None and np.any(v == 0):
                ax.set_facecolor("#F6C6A7")

            if arr is not None and len(arr) > 0:
                mean = float(np.mean(arr))
                diff = mean - true_distance
                if abs(diff) <= 2:
                    box_color = BOX_WITHIN
                elif diff < -2:
                    box_color = BOX_LOW
                else:
                    box_color = BOX_OVER

                sns.boxplot(y=arr, ax=ax, color=box_color,
                            linewidth=0.8, fliersize=1.5,
                            flierprops=dict(marker=".", markersize=1.5))

                y_lo, y_hi = ax.get_ylim()
                half_range = max(abs(y_hi - true_distance),
                                 abs(true_distance - y_lo))
                ax.set_ylim(true_distance - half_range,
                            true_distance + half_range)

                q1z = np.percentile(arr, 25)
                medz = np.median(arr)
                q3z = np.percentile(arr, 75)
                annotation = f"Q1:{q1z:.1f} M:{medz:.1f} Q3:{q3z:.1f} μ:{mean:.1f}"

                ax.tick_params(axis="x", bottom=False)
                ax.annotate(annotation,
                            xy=(0.5, 0), xycoords="axes fraction",
                            xytext=(0, -8), textcoords="offset points",
                            ha="center", va="top", fontsize=4.5,
                            annotation_clip=False)
            else:
                ax.text(0.5, 0.5, "No data", ha="center", va="center",
                        fontsize=7, transform=ax.transAxes)

            ax.set_title(f"Z{z}", fontsize=7, pad=2)
            ax.set_ylabel("")
            ax.set_xlabel("")
            ax.tick_params(labelsize=4.5)

        plt.tight_layout(rect=[0, 0, 1, 1])
        save_path = os.path.join(output_path, f"{dist_key}_Tiled_Boxplots_MultiExperiment.png")
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        plt.close(fig)


# ── (6) Per-distance tiled validity pooled across experiments ─────────────────
def plot_multiexp_tiled_validity(experiments: dict, output_path: str):
    """
    One 8x8 tiled validity scatter figure per distance.
    Validity arrays are concatenated across experiments per zone.
    Always-valid = strictly no zeros in concatenated stream.
    """
    all_dist_keys = set()
    for _exp_name, (trimmed, _n_frames, _shortest) in experiments.items():
        all_dist_keys.update(trimmed.keys())

    for dist_key in sorted(all_dist_keys, key=extract_true_distance):
        _, pooled_valid = _pooled_zone_arrays_for_distance(experiments, dist_key)

        fig, axes = plt.subplots(8, 8, figsize=(22, 14))
        fig.suptitle(f"Zone Validity (Multi-Experiment) — {dist_key}", fontsize=14, y=1.002)

        for z in range(64):
            row, col = zone_to_grid(z)
            ax = axes[row][col]
            arr = pooled_valid.get(z)

            if arr is not None and len(arr) > 0 and np.any(arr == 0):
                frames = np.arange(len(arr))
                colors = [VALID_COLOR if v == 1 else INVALID_COLOR for v in arr]
                ax.scatter(frames, arr, c=colors, s=8, alpha=0.8, linewidths=0)
                ax.set_yticks([0, 1])
                ax.set_yticklabels(["Inv", "Val"], fontsize=4)
                ax.set_xticks([])
                pct = 100.0 * float(np.mean(arr == 1))
                ax.annotate(f"\n{pct:.1f}% valid",
                            xy=(0.5, 0), xycoords="axes fraction",
                            xytext=(0, -8), textcoords="offset points",
                            ha="center", va="top", fontsize=5.5,
                            annotation_clip=False)
            else:
                ax.set_facecolor(ALWAYS_VALID_BG)
                ax.text(0.5, 0.5, "Always\nValid", ha="center", va="center",
                        fontsize=7, color="#2E7D32", transform=ax.transAxes)
                ax.set_xticks([])
                ax.set_yticks([])

            ax.set_title(f"Z{z}", fontsize=7, pad=2)

        plt.tight_layout(rect=[0, 0, 1, 1])
        save_path = os.path.join(output_path, f"{dist_key}_Tiled_Validity_MultiExperiment.png")
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        plt.close(fig)

def plot_multiexp_drift_envelope(experiments: dict, output_path: str):
    """
    Drift plot across experiments per distance with envelope shading.

    For each (distance dist_key):
      1) For each experiment:
         - For each frame t, compute mean(distance_mm over zones that are valid at t),
           i.e. validity mask applied per-frame per-zone.
         - Convert to error vs true distance: mean_measured[t] - true_distance
      2) Across experiments (at each t):
         - min line, max line
         - mean-of-means line
      3) Fit a linear regression to mean-of-means line (error vs frame)
         and plot it as a thicker line.

    Color scheme:
      - One base color per distance (same palette ordering as Visualize comparative drift style).
      - min/max/mean lines: lighter shade of the distance color
      - regression line: full distance color, thicker
      - shading between min/max: regression color with alpha=0.60 (lower to lighten)
      - If a palette color repeats, the *repeated* distance gets a dashed regression line.
    """
    # Requires in module scope:
    #   import numpy as np
    #   import matplotlib.pyplot as plt
    #   import matplotlib.colors as mcolors
    #   from scipy import stats as sp_stats
    # And helpers:
    #   extract_true_distance()
    #   _lighten_rgb()

    PALETTE = ["#E8708E", "#EDD020", "#E97122", "#52A83C", "#18A5AA", "#55A9D4", "#2E61A8"]

    # collect all distances
    all_dist_keys = set()
    for _exp_name, (trimmed, _n_frames, _shortest) in experiments.items():
        all_dist_keys.update(trimmed.keys())
    dist_keys = sorted(all_dist_keys, key=extract_true_distance)
    if not dist_keys:
        return

    fig = plt.figure(figsize=(14, 9))
    gs = plt.GridSpec(2, 1, height_ratios=[10, 2], hspace=0.05)
    ax = fig.add_subplot(gs[0])
    ax_text = fig.add_subplot(gs[1])
    ax_text.axis("off")
    ax.tick_params(axis="x", bottom=False, labelbottom=False)

    # Store fit slopes for annotation
    slopes = {}

    # track palette reuse to dash repeated distances (like Visualize.py)
    used_base_hex = {}

    for j, dist_key in enumerate(dist_keys):
        true_distance = extract_true_distance(dist_key)

        # Per-experiment time series of mean(valid zones per frame)
        exp_series = []

        for _exp_name, (trimmed, _n_frames, _shortest) in experiments.items():
            if dist_key not in trimmed:
                continue
            zone_data = trimmed[dist_key]

            # find a common length for this experiment at this distance
            zone_lengths = []
            for z in range(64):
                arr = zone_data.get("distance_mm", {}).get(z)
                v = zone_data.get("is_valid_range", {}).get(z)
                if arr is None or v is None:
                    continue
                zone_lengths.append(min(len(arr), len(v)))

            if not zone_lengths:
                continue

            T = min(zone_lengths)

            # build arrays: dist_stack (n_zones, T), valid_stack (n_zones, T)
            dist_stack = []
            valid_stack = []
            for z in range(64):
                arr = zone_data.get("distance_mm", {}).get(z)
                v = zone_data.get("is_valid_range", {}).get(z)
                if arr is None or v is None:
                    continue
                a = np.asarray(arr)[:T]
                vv = np.asarray(v)[:T]

                # keep only entries where validity is 0/1; if there is noise, treat non-0/1 as invalid
                mask01 = (vv == 0) | (vv == 1)
                vv01 = np.where(mask01, vv, 0)

                dist_stack.append(a)
                valid_stack.append(vv01)

            if not dist_stack:
                continue

            dist_stack = np.array(dist_stack)     # (Z, T)
            valid_stack = np.array(valid_stack)   # (Z, T)

            # For each t: mean over zones that are valid at t
            mean_per_frame = np.empty(T, dtype=float)
            mean_per_frame[:] = np.nan

            for t in range(T):
                mask = (valid_stack[:, t] == 1)
                if np.any(mask):
                    mean_per_frame[t] = float(np.mean(dist_stack[mask, t]))

            # Convert to error vs true distance
            err_per_frame = mean_per_frame - true_distance
            exp_series.append(err_per_frame)

        if not exp_series:
            continue

        # Align all experiments for this distance by shortest length (so min/max/mean line up)
        min_T = min(len(s) for s in exp_series)
        S = np.array([s[:min_T] for s in exp_series], dtype=float)  # (E, T)

        # Per-frame envelopes ignoring NaNs
        min_line = np.nanmin(S, axis=0)
        max_line = np.nanmax(S, axis=0)
        mean_line = np.nanmean(S, axis=0)

        frames = np.arange(min_T)

        base_hex = PALETTE[j % len(PALETTE)]
        base_rgb = mcolors.to_rgb(base_hex)
        light_rgb = _lighten_rgb(base_rgb, amount=0.55)

        used_base_hex[base_hex] = used_base_hex.get(base_hex, 0) + 1
        fit_linestyle = "--" if used_base_hex[base_hex] > 1 else "-"

        # Shading between min/max: regression color at 60% alpha (lower to lighten)
        ax.fill_between(frames, min_line, max_line,
                        color=(*base_rgb, 0.20), linewidth=0)

        ax.plot(frames, mean_line, color=light_rgb, linewidth=2.0, linestyle="-",
                label=f"{dist_key} mean")

        # Regression on mean line (ignore NaNs)
        ok = np.isfinite(mean_line)
        if np.sum(ok) >= 2:
            slope, intercept, _, _, _ = sp_stats.linregress(frames[ok], mean_line[ok])
            slopes[dist_key] = float(slope)
            ax.plot(frames, intercept + slope * frames,
                    color=base_rgb, linewidth=3.0, linestyle=fit_linestyle,
                    label=f"{dist_key} :({slope:+.4f} mm/frame)")
        else:
            slopes[dist_key] = None

    ax.axhline(0, color="black", linewidth=1.0, linestyle="-", alpha=0.4)
    ax.set_ylabel("Error [mm]", fontsize=10)
    ax.set_xlabel("Time →")
    ax.set_title("Drift Across Experiments\n", pad=60)

    # Legend above plot but below title: keep ONLY "fit" entries so it stays readable
    handles, labels = ax.get_legend_handles_labels()
    keep_h, keep_l = [], []
    for h, l in zip(handles, labels):
        if "mm/frame" in l:
            keep_h.append(h)
            keep_l.append(l)

    if keep_h:
        ax.legend(keep_h, keep_l,
                  fontsize=8,
                  loc="upper center",
                  bbox_to_anchor=(0.5, 1.15),  # above axes, below title
                  ncol=min(4, len(keep_h)),
                  framealpha=0.95)

    save_path = os.path.join(output_path, "MultiExperiment_Drift_Envelope.png")
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close(fig)

def comparative_valid_raw_boxplots(experiments: dict, output_path: str):
    """
    Grouped boxplots of RAW VALID measured distances per distance.
    Each experiment is its own color (PALETTE).
    Data: pool all zones' distance_mm values where is_valid_range == 1.
    """
    all_dist_keys = set()
    for name, (trimmed, _, _) in experiments.items():
        all_dist_keys.update(trimmed.keys())

    exp_names = list(experiments.keys())
    n_exps = len(exp_names)

    dist_keys = sorted(all_dist_keys, key=lambda k: extract_true_distance(k))
    if not dist_keys or n_exps == 0:
        return

    x_positions = np.arange(len(dist_keys))
    width = 0.8 / max(n_exps, 1)

    fig, ax = plt.subplots(figsize=(max(12, 2.0 * len(dist_keys)), 7))
    ax.set_title("Raw Valid Distance Readings per Experiment (Grouped Boxplots)", fontsize=12)

    for i, name in enumerate(exp_names):
        trimmed, _, _ = experiments[name]
        color = PALETTE[i % len(PALETTE)]

        for j, dist_key in enumerate(dist_keys):
            if dist_key not in trimmed:
                continue

            zone_data = trimmed[dist_key]

            # pool valid-only measurements across zones (no analyse_tof_data needed)
            chunks = []
            for z in range(64):
                arr = zone_data.get("distance_mm", {}).get(z)
                v = zone_data.get("is_valid_range", {}).get(z)
                if arr is None or v is None:
                    continue

                arr = np.asarray(arr)
                v = np.asarray(v)

                mask01 = (v == 0) | (v == 1)
                if not np.any(mask01):
                    continue
                v01 = v[mask01]
                arr01 = arr[:len(v)][mask01]

                mask_valid = (v01 == 1)
                if np.any(mask_valid):
                    chunks.append(arr01[mask_valid])

            all_valid = _safe_concat(chunks)
            if all_valid is None or len(all_valid) == 0:
                continue

            pos = x_positions[j] + (i - n_exps / 2 + 0.5) * width

            ax.boxplot(all_valid,
                       positions=[pos],
                       widths=width * 0.9,
                       patch_artist=True,
                       manage_ticks=False,
                       medianprops=dict(color="white", linewidth=1.3),
                       whiskerprops=dict(color=color, linewidth=1.0),
                       capprops=dict(color=color, linewidth=1.0),
                       flierprops=dict(marker=".", markersize=2,
                                       markerfacecolor=color, alpha=0.35),
                       boxprops=dict(facecolor=color, color=color,
                                     linewidth=1.0))

    ax.set_xticks(x_positions)
    ax.set_xticklabels(dist_keys, fontsize=9)
    ax.set_xlabel("Distance", fontsize=10)
    ax.set_ylabel("Measured distance [mm] (valid only)", fontsize=10)

    ax.yaxis.set_minor_locator(plt.MultipleLocator(0.2))
    ax.grid(True, which="major", color="grey", linewidth=0.5, alpha=0.25, zorder=0)
    ax.grid(True, which="minor", color="grey", linewidth=0.3, alpha=0.15, zorder=0)

    legend_elements = [
        mpatches.Patch(facecolor=PALETTE[i % len(PALETTE)], label=name)
        for i, name in enumerate(exp_names)
    ]
    ax.legend(handles=legend_elements, fontsize=8, loc="upper left", framealpha=0.95)

    plt.tight_layout()
    save_path = os.path.join(output_path, "Comparative_Valid_Raw_Boxplots_MultiExperiment.png")
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close(fig)

def ridgeplot_valid_raw_all_experiments(experiments: dict, output_path: str):
    """
    Ridge/KDE plot like Visualize.plot_ridge_by_distance, but:
      - Pools ALL VALID raw distance readings across ALL experiments per distance.
      - Uses measured distance values (mm), not error.
      - Draws dashed vertical lines at each true distance (target).

    Output:
      MultiExperiment_Ridgeplot_ValidRaw_MultiExperiment.png
    """
    import pandas as pd
    import seaborn as sns

    sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})

    # Collect all distances
    all_dist_keys = set()
    for _name, (trimmed, _n_frames, _shortest) in experiments.items():
        all_dist_keys.update(trimmed.keys())

    dist_keys = sorted(all_dist_keys, key=lambda k: extract_true_distance(k))
    if not dist_keys:
        return

    per_dist = []
    for dist_key in dist_keys:
        true_distance = extract_true_distance(dist_key)

        pooled_chunks = []

        for _exp_name, (trimmed, _n_frames, _shortest) in experiments.items():
            if dist_key not in trimmed:
                continue

            zone_data = trimmed[dist_key]

            # pool valid-only measurements across zones for this experiment
            for z in range(64):
                arr = zone_data.get("distance_mm", {}).get(z)
                v = zone_data.get("is_valid_range", {}).get(z)
                if arr is None or v is None:
                    continue

                arr = np.asarray(arr)
                v = np.asarray(v)

                # robustness: only 0/1 validity entries are considered; others treated as invalid
                mask01 = (v == 0) | (v == 1)
                if not np.any(mask01):
                    continue

                v01 = v[mask01]
                arr01 = arr[:len(v)][mask01]

                mask_valid = (v01 == 1)
                if np.any(mask_valid):
                    pooled_chunks.append(arr01[mask_valid])

        meas = _safe_concat(pooled_chunks)
        if meas is None or len(meas) == 0:
            continue

        per_dist.append((dist_key, true_distance, meas))

    if not per_dist:
        return

    # Build long dataframe
    rows = []
    for dist_key, true_distance, meas in per_dist:
        for x in meas:
            rows.append({"dist_key": dist_key, "true_distance": true_distance, "x": float(x)})

    df = pd.DataFrame(rows)

    order = [k for (k, _, _) in per_dist]
    df["dist_key"] = pd.Categorical(df["dist_key"], categories=order, ordered=True)

    palette = sns.cubehelix_palette(len(order), rot=-.25, light=.7)

    g = sns.FacetGrid(
        df,
        row="dist_key",
        hue="dist_key",
        aspect=22,
        height=0.42,
        palette=palette
    )

    g.map(
        sns.kdeplot,
        "x",
        bw_adjust=0.8,
        clip_on=False,
        fill=False,
        alpha=1,
        linewidth=1.3
    )
    g.map(
        sns.kdeplot,
        "x",
        bw_adjust=0.8,
        clip_on=False,
        fill=True,
        alpha=.7,
        linewidth=0
    )
    g.map(plt.axhline, y=0, lw=0.8, clip_on=False)

    true_dists = [td for _, td, _ in per_dist]
    xticks = sorted(true_dists)

    axes = g.axes.flatten()
    for idx, ax in enumerate(axes):
        for td in true_dists:
            ax.axvline(td, color="grey", linestyle="--",
                       linewidth=0.8, alpha=0.7, zorder=0)

        ax.set_xticks(xticks)
        ax.grid(True, axis="x", color="grey", linewidth=0.4, alpha=0.2)

        # Only bottom facet shows x tick labels
        if idx != len(axes) - 1:
            ax.tick_params(axis="x", labelbottom=False)
        else:
            ax.tick_params(axis="x", labelrotation=45)

    def label(x, color, label):
        ax = plt.gca()
        ax.text(
            0.01, 0.2, label,
            fontweight="bold", color=color,
            ha="left", va="center",
            transform=ax.transAxes,
            fontsize=7.5
        )

    g.map(label, "x")

    g.fig.subplots_adjust(
        hspace=-0.55,
        left=0.06,
        right=0.995,
        top=0.92,
        bottom=0.13
    )

    g.set_titles("")
    g.set(yticks=[], ylabel="")
    g.despine(bottom=True, left=True)
    g.set_xlabels("Distance [mm] (valid only, all experiments pooled)")

    g.fig.suptitle(
        "Distance Ridge Plot (valid-only, all experiments pooled)\n",
        fontsize=12
    )

    save_path = os.path.join(output_path, "MultiExperiment_Ridgeplot_ValidRaw_MultiExperiment.png")
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close(g.fig)

# ── Master entrypoint ─────────────────────────────────────────────────────────
def generate_multi_experiment_plots_v2(experiment_folders: dict, origin: int, output_path: str) -> str:
    experiments = load_multi_experiment(experiment_folders, origin)
    plots_folder = setup_output_folder(output_path, "multiV2")

    print("Generating concatenated overall validity heatmap...")
    plot_multiexp_validity_heatmap(experiments, plots_folder)

    print("Generating error and validity plot...")
    plot_multiexp_error_and_validity(experiments, plots_folder)

    print("Generating multi-experiment error heatmaps...")
    plot_multiexp_error_heatmaps(experiments, plots_folder)

    print("Generating per-distance heatmaps (multi-experiment, valid-only)...")
    plot_multiexp_per_distance_heatmaps(experiments, plots_folder)

    print("Generating per-distance tiled boxplots (multi-experiment, includes invalid measurements)...")
    plot_multiexp_tiled_boxplots(experiments, plots_folder)

    print("Generating per-distance tiled validity plots (multi-experiment)...")
    plot_multiexp_tiled_validity(experiments, plots_folder)

    print("Generating drift envelope plot...")
    plot_multiexp_drift_envelope(experiments, plots_folder)

    print("Generating grouped raw-valid boxplots...")
    comparative_valid_raw_boxplots(experiments, plots_folder)

    print("Generating pooled valid-only ridge plot...")
    ridgeplot_valid_raw_all_experiments(experiments, plots_folder)

    print("Done!")
    return plots_folder
