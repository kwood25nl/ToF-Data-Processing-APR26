import os
import re
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from datetime import datetime


# ── Colours ────────────────────────────────────────────────────────────────────
BOX_WITHIN       = "#52A83C"
BOX_OVER         = "#2E61A8"
BOX_LOW          = "#E8708E"
VALID_COLOR      = "#52A83C"
INVALID_COLOR    = "#E97122"
ALWAYS_VALID_BG  = "#BADCB1"


# ── Folder helpers ─────────────────────────────────────────────────────────────

def setup_output_folder(output_path: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    full_path = os.path.join(output_path, f"plots-{timestamp}")
    os.makedirs(full_path, exist_ok=False)
    return full_path


def zone_to_grid(z: int):
    """
    Zone 0 = bottom-left, zone 63 = top-right.
    Flips row so that zone 0 appears at the bottom of the matplotlib grid.
    """
    physical_row = z // 8
    col          = z % 8
    row          = 7 - physical_row
    return row, col


def extract_true_distance(key: str) -> float:
    digits = re.findall(r"-?\d+", key)
    return float("".join(digits))


# ── Data helpers ───────────────────────────────────────────────────────────────

def get_valid_distances(zone_data: dict, stats_data_dist: dict) -> dict:
    """
    Returns a dict of zone -> array of valid distance readings only.
    Zones with no valid readings are excluded.
    """
    valid = {}
    for z in range(64):
        arr       = zone_data["distance_mm"][z]
        valid_arr = zone_data["is_valid_range"][z]
        if arr is not None and valid_arr is not None:
            mask = valid_arr == 1
            if mask.any():
                valid[z] = arr[mask]
    return valid


def compute_zone_means(valid_distances: dict) -> dict:
    """Returns zone -> mean of valid distance readings."""
    return {z: float(np.mean(arr)) for z, arr in valid_distances.items()}


def colour_for_value(value: float, true_distance: float,
                     min_val: float, max_val: float) -> tuple:
    """
    Returns an RGBA colour based on how far value is from true_distance.
    Green  (#52A83C): within +/-2mm, darkest = closest to target
    Pink   (#E8708E): below target-2mm, darkest = furthest below
    Blue   (#2E61A8): above target+2mm, darkest = furthest above
    Alpha ranges from 0.5 (lightest) to 1.0 (darkest).
    """
    diff = value - true_distance

    if abs(diff) <= 2:
        base   = mcolors.to_rgb("#52A83C")
        extent = max(abs(min_val - true_distance),
                     abs(max_val - true_distance), 2)
        alpha  = 1.0 - 0.5 * (abs(diff) / extent)
    elif diff < -2:
        base      = mcolors.to_rgb("#E8708E")
        most_below = min(min_val - true_distance, -2)
        alpha     = 1.0 - 0.5 * ((diff - most_below) / (-2 - most_below + 1e-9))
        alpha     = np.clip(alpha, 0.5, 1.0)
    else:
        base       = mcolors.to_rgb("#2E61A8")
        most_above = max(max_val - true_distance, 2)
        alpha      = 1.0 - 0.5 * ((most_above - diff) / (most_above - 2 + 1e-9))
        alpha      = np.clip(alpha, 0.5, 1.0)

    return (*base, alpha)


def draw_heatmap(ax, means: dict, true_distance: float,
                 grid_shape: tuple, zone_labels: list,
                 cell_means: list):
    """
    Generic heatmap drawer.

    Args:
        ax:            Matplotlib axis to draw on.
        means:         Dict of group_index -> mean value.
        true_distance: Target distance for colour coding.
        grid_shape:    (rows, cols) of the grid.
        zone_labels:   List of label strings per cell (row-major, top-to-bottom).
        cell_means:    List of mean values per cell (row-major, top-to-bottom).
    """
    rows, cols = grid_shape
    all_vals   = [v for v in means.values() if v is not None]
    min_val    = min(all_vals) if all_vals else true_distance
    max_val    = max(all_vals) if all_vals else true_distance

    for i, (label, mean_val) in enumerate(zip(zone_labels, cell_means)):
        r = i // cols
        c = i % cols
        r = (rows - 1) - r  # flip for matplotlib

        if mean_val is not None:
            color = colour_for_value(mean_val, true_distance, min_val, max_val)
        else:
            color = (0.85, 0.85, 0.85, 1.0)

        rect = mpatches.FancyBboxPatch(
            (c, r), 1, 1,
            boxstyle="square,pad=0",
            facecolor=color, edgecolor="white", linewidth=0.5,
            transform=ax.transData
        )
        ax.add_patch(rect)

        text_val = f"{mean_val:.1f}" if mean_val is not None else "N/A"
        ax.text(c + 0.5, r + 0.5, f"{label}\n{text_val}",
                ha="center", va="center", fontsize=6,
                color="white", fontweight="bold")

    ax.set_xlim(0, cols)
    ax.set_ylim(0, rows)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")


# ── Tiled boxplots ─────────────────────────────────────────────────────────────

def plot_tiled_boxplots(data: dict, stats_data: dict, output_path: str):
    """
    One 8x8 tiled boxplot figure per distance.
    - Y-axis autoscales per zone but is centred on the true distance.
    - Box colour indicates whether mean is within +/-2mm of true distance.
    - Background is #F6C6A7 if zone is ever invalid.
    - Stats annotation sits just below x-axis with minimal gap.
    """
    ZONES = range(64)

    for dist_key, zone_data in data.items():
        true_distance = extract_true_distance(dist_key)

        fig, axes = plt.subplots(8, 8, figsize=(24, 24))
        fig.suptitle(f"Distance Boxplots — {dist_key}", fontsize=14, y=1.002)

        for z in ZONES:
            row, col = zone_to_grid(z)
            ax       = axes[row][col]
            arr      = zone_data["distance_mm"][z]
            mean     = stats_data[dist_key]["per_zone"]["distance_mm"][z]["mean"]
            pct      = stats_data[dist_key]["per_zone"]["percent_valid"][z]

            if pct < 100.0:
                ax.set_facecolor("#F6C6A7")

            if arr is not None:
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

                q1z  = np.percentile(arr, 25)
                medz = np.median(arr)
                q3z  = np.percentile(arr, 75)
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
        save_path = os.path.join(output_path, f"{dist_key}_Tiled_Boxplots.png")
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        plt.close(fig)


# ── Tiled validity ─────────────────────────────────────────────────────────────

def plot_tiled_validity(data: dict, stats_data: dict, output_path: str):
    """
    One 8x8 tiled validity scatter figure per distance.
    - Always valid zones: green background, no scatter.
    - Sometimes invalid zones: white background with scatter plot.
    """
    ZONES = range(64)

    for dist_key, zone_data in data.items():

        fig, axes = plt.subplots(8, 8, figsize=(22, 14))
        fig.suptitle(f"Zone Validity — {dist_key}", fontsize=14, y=1.002)

        for z in ZONES:
            row, col = zone_to_grid(z)
            ax       = axes[row][col]
            arr      = zone_data["is_valid_range"][z]
            pct      = stats_data[dist_key]["per_zone"]["percent_valid"][z]

            if arr is not None and pct < 100.0:
                frames = np.arange(len(arr))
                colors = [VALID_COLOR if v == 1 else INVALID_COLOR for v in arr]
                ax.scatter(frames, arr, c=colors, s=8, alpha=0.8, linewidths=0)
                ax.set_yticks([0, 1])
                ax.set_yticklabels(["Inv", "Val"], fontsize=4)
                ax.set_xticks([])
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
        save_path = os.path.join(output_path, f"{dist_key}_Tiled_Validity.png")
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        plt.close(fig)


# ── Total boxplot ──────────────────────────────────────────────────────────────

def plot_total_boxplots(data: dict, stats_data: dict, output_path: str):
    """
    One boxplot per distance using all valid readings across all zones.
    """
    for dist_key, zone_data in data.items():
        true_distance   = extract_true_distance(dist_key)
        valid_distances = get_valid_distances(zone_data, stats_data[dist_key])

        if not valid_distances:
            continue

        all_valid = np.concatenate(list(valid_distances.values()))
        mean      = float(np.mean(all_valid))
        std       = float(np.std(all_valid))
        q1        = float(np.percentile(all_valid, 25))
        medv      = float(np.median(all_valid))
        q3        = float(np.percentile(all_valid, 75))

        diff = mean - true_distance
        if abs(diff) <= 2:
            box_color = BOX_WITHIN
        elif diff < -2:
            box_color = BOX_LOW
        else:
            box_color = BOX_OVER

        fig = plt.figure(figsize=(6, 7))
        gs  = plt.GridSpec(2, 1, height_ratios=[10, 1], hspace=0.05)
        ax  = fig.add_subplot(gs[0])
        ax_text = fig.add_subplot(gs[1])
        ax_text.axis("off")

        sns.boxplot(y=all_valid, ax=ax, color=box_color,
                    linewidth=0.8, fliersize=2)

        y_lo, y_hi = ax.get_ylim()
        half_range = max(abs(y_hi - true_distance),
                         abs(true_distance - y_lo))
        ax.set_ylim(true_distance - half_range, true_distance + half_range)
        ax.axhline(true_distance, color="grey", linewidth=0.8,
                   linestyle="--", alpha=0.7, label="Target")

        ax.set_ylabel("Measured Distance [mm]")
        ax.set_title(f"{dist_key} Total Boxplot")
        ax.tick_params(axis="x", bottom=False)

        stats_str = (f"Q1: {q1:.2f}  |  Median: {medv:.2f}  |  "
                     f"Q3: {q3:.2f}  |  Mean: {mean:.2f}  |  Std: {std:.2f}")
        ax_text.text(0.5, 0.8, stats_str, ha="center", va="top",
                     fontsize=8, transform=ax_text.transAxes)

        save_path = os.path.join(output_path, f"{dist_key}_Total_Boxplot.png")
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        plt.close(fig)


# ── Tiled heatmaps ─────────────────────────────────────────────────────────────

def plot_tiled_heatmaps(data: dict, stats_data: dict, output_path: str):
    """
    One 2x2 tiled figure per distance containing:
    Top-left:     8x8 zone heatmap
    Top-right:    4x4 group heatmap
    Bottom-left:  2x2 group heatmap
    Bottom-right: Ring group heatmap
    """
    def get_ring(z):
        r = z // 8
        c = z % 8
        return min(r, 7 - r, c, 7 - c)

    for dist_key, zone_data in data.items():
        true_distance   = extract_true_distance(dist_key)
        valid_distances = get_valid_distances(zone_data, stats_data[dist_key])
        zone_means      = compute_zone_means(valid_distances)

        fig, axes = plt.subplots(2, 2, figsize=(16, 16))
        fig.suptitle(f"{dist_key} — Distance Heatmaps", fontsize=14, y=1.002)

        # ── Top-left: 8x8 ─────────────────────────────────────────────────────
        ax = axes[0][0]
        ax.set_title("8×8 Zone Means", fontsize=10)
        labels     = [f"Z{z}" for z in range(64)]
        cell_means = [zone_means.get(z, None) for z in range(64)]
        draw_heatmap(ax, zone_means, true_distance,
                     (8, 8), labels, cell_means)

        # ── Top-right: 4x4 ────────────────────────────────────────────────────
        ax = axes[0][1]
        ax.set_title("4×4 Group Means", fontsize=10)
        group_means_4x4 = {}
        labels_4x4      = []
        cell_means_4x4  = []

        for sg in range(16):
            sr = sg // 4
            sc = sg % 4
            zones_in_group = []
            for dr in range(2):
                for dc in range(2):
                    z = (sr * 2 + dr) * 8 + (sc * 2 + dc)
                    if z in zone_means:
                        zones_in_group.append(zone_means[z])
            mean_val = float(np.mean(zones_in_group)) if zones_in_group else None
            group_means_4x4[sg] = mean_val
            labels_4x4.append(f"G{sg}")
            cell_means_4x4.append(mean_val)

        draw_heatmap(ax, group_means_4x4, true_distance,
                     (4, 4), labels_4x4, cell_means_4x4)

        # ── Bottom-left: 2x2 ──────────────────────────────────────────────────
        ax = axes[1][0]
        ax.set_title("2×2 Group Means", fontsize=10)
        group_means_2x2 = {}
        labels_2x2      = []
        cell_means_2x2  = []

        for sg in range(4):
            sr = sg // 2
            sc = sg % 2
            zones_in_group = []
            for dr in range(4):
                for dc in range(4):
                    z = (sr * 4 + dr) * 8 + (sc * 4 + dc)
                    if z in zone_means:
                        zones_in_group.append(zone_means[z])
            mean_val = float(np.mean(zones_in_group)) if zones_in_group else None
            group_means_2x2[sg] = mean_val
            labels_2x2.append(f"G{sg}")
            cell_means_2x2.append(mean_val)

        draw_heatmap(ax, group_means_2x2, true_distance,
                     (2, 2), labels_2x2, cell_means_2x2)

        # ── Bottom-right: Rings ────────────────────────────────────────────────
        ax = axes[1][1]
        ax.set_title("Ring Group Means", fontsize=10)

        ring_zones = {0: [], 1: [], 2: [], 3: []}
        for z, mean_val in zone_means.items():
            ring_zones[get_ring(z)].append(mean_val)

        ring_means = {
            ring: float(np.mean(vals)) if vals else None
            for ring, vals in ring_zones.items()
        }

        all_vals = [v for v in ring_means.values() if v is not None]
        min_val  = min(all_vals) if all_vals else true_distance
        max_val  = max(all_vals) if all_vals else true_distance

        for z in range(64):
            r        = z // 8
            c        = z % 8
            ring     = get_ring(z)
            mean_val = ring_means.get(ring)
            plot_r   = 7 - r

            color = colour_for_value(mean_val, true_distance, min_val, max_val) \
                    if mean_val is not None else (0.85, 0.85, 0.85, 1.0)

            rect = mpatches.FancyBboxPatch(
                (c, plot_r), 1, 1,
                boxstyle="square,pad=0",
                facecolor=color, edgecolor="white", linewidth=0.5
            )
            ax.add_patch(rect)

            text_val = f"{mean_val:.1f}" if mean_val is not None else "N/A"
            ax.text(c + 0.5, plot_r + 0.5, text_val,
                    ha="center", va="center", fontsize=6,
                    color="white", fontweight="bold")

        # Ring boundary lines
        for ring in range(1, 4):
            o = ring
            ax.plot([o, 8-o, 8-o, o, o],
                    [o, o, 8-o, 8-o, o],
                    color="white", linewidth=1.5, linestyle="--")

        # Ring legend
        for ring, mean_val in ring_means.items():
            label = f"Ring {ring}: {mean_val:.1f} mm" \
                    if mean_val is not None else f"Ring {ring}: N/A"
            ax.text(8.2, 7.5 - ring * 1.8, label,
                    ha="left", va="center", fontsize=8)

        ax.set_xlim(0, 8)
        ax.set_ylim(0, 8)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal")

        plt.tight_layout()
        save_path = os.path.join(output_path, f"{dist_key}_Heatmaps.png")
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        plt.close(fig)

# ── Experiment-wide error heatmaps ────────────────────────────────────────────

def compute_zone_errors(data: dict, stats_data: dict) -> tuple[dict, dict]:
    """
    Computes per-zone absolute average error and bias error across all distances.

    Returns:
        abs_errors:  zone -> mean absolute error across all distances
        bias_errors: zone -> mean bias error across all distances
    """
    zone_abs  = {z: [] for z in range(64)}
    zone_bias = {z: [] for z in range(64)}

    for dist_key, zone_data in data.items():
        true_distance   = extract_true_distance(dist_key)
        valid_distances = get_valid_distances(zone_data, stats_data[dist_key])
        zone_means      = compute_zone_means(valid_distances)

        for z in range(64):
            if z in zone_means:
                error = zone_means[z] - true_distance
                zone_abs[z].append(abs(error))
                zone_bias[z].append(error)

    abs_errors  = {z: float(np.mean(v)) if v else None
                   for z, v in zone_abs.items()}
    bias_errors = {z: float(np.mean(v)) if v else None
                   for z, v in zone_bias.items()}

    return abs_errors, bias_errors


def bias_colour(bias: float, min_bias: float, max_bias: float) -> tuple:
    """
    Colour based on signed bias:
    Positive (overestimate): blue  #2E61A8, darkest = most positive
    Negative (underestimate): pink #E8708E, darkest = most negative
    Near zero (mixed):        green #52A83C, darkest = closest to zero

    Boundary between regions is determined by min/max bias in the dataset.
    Alpha ranges from 0.5 (lightest) to 1.0 (darkest).
    """
    # Define "near zero" as within 20% of the larger extreme
    extreme   = max(abs(min_bias), abs(max_bias))
    threshold = extreme * 0.2

    if abs(bias) <= threshold:
        # Green — darkest = closest to zero
        base  = mcolors.to_rgb("#52A83C")
        alpha = 1.0 - 0.5 * (abs(bias) / (threshold + 1e-9))
    elif bias > threshold:
        # Blue — darkest = most positive
        base  = mcolors.to_rgb("#2E61A8")
        alpha = 0.5 + 0.5 * (bias / (max_bias + 1e-9))
    else:
        # Pink — darkest = most negative
        base  = mcolors.to_rgb("#E8708E")
        alpha = 0.5 + 0.5 * (abs(bias) / (abs(min_bias) + 1e-9))

    return (*base, float(np.clip(alpha, 0.5, 1.0)))


def draw_error_heatmap(ax, abs_errors: dict, bias_errors: dict,
                       grid_shape: tuple, zone_labels: list,
                       cell_abs: list, cell_bias: list):
    """
    Draws a heatmap where:
    - Cell colour represents bias error
    - Cell text shows absolute average error
    """
    rows, cols = grid_shape

    valid_bias = [v for v in cell_bias if v is not None]
    min_bias   = min(valid_bias) if valid_bias else 0.0
    max_bias   = max(valid_bias) if valid_bias else 0.0

    for i, (label, abs_val, bias_val) in enumerate(
            zip(zone_labels, cell_abs, cell_bias)):
        r = i // cols
        c = i % cols
        r = (rows - 1) - r  # flip for matplotlib

        if bias_val is not None:
            color = bias_colour(bias_val, min_bias, max_bias)
        else:
            color = (0.85, 0.85, 0.85, 1.0)

        rect = mpatches.FancyBboxPatch(
            (c, r), 1, 1,
            boxstyle="square,pad=0",
            facecolor=color, edgecolor="white", linewidth=0.5,
            transform=ax.transData
        )
        ax.add_patch(rect)

        abs_text  = f"{abs_val:.1f}" if abs_val is not None else "N/A"
        bias_text = f"b:{bias_val:+.1f}" if bias_val is not None else ""
        ax.text(c + 0.5, r + 0.5, f"{label}\n{abs_text}\n{bias_text}",
                ha="center", va="center", fontsize=5.5,
                color="white", fontweight="bold")

    ax.set_xlim(0, cols)
    ax.set_ylim(0, rows)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")


def plot_error_heatmaps(data: dict, stats_data: dict, output_path: str):
    """
    Single tiled 2x2 figure with 8x8, 4x4, 2x2 and ring error heatmaps
    across the entire experiment set.
    Colour = bias, text = absolute average error.
    """
    def get_ring(z):
        r = z // 8
        c = z % 8
        return min(r, 7 - r, c, 7 - c)

    abs_errors, bias_errors = compute_zone_errors(data, stats_data)

    fig, axes = plt.subplots(2, 2, figsize=(16, 16))
    fig.suptitle("Experiment Error Heatmaps\n(text = mean absolute error, colour = bias)",
                 fontsize=14, y=1.002)

    # ── Top-left: 8x8 ─────────────────────────────────────────────────────────
    ax = axes[0][0]
    ax.set_title("8×8 Zone Errors", fontsize=10)
    labels    = [f"Z{z}" for z in range(64)]
    cell_abs  = [abs_errors.get(z) for z in range(64)]
    cell_bias = [bias_errors.get(z) for z in range(64)]
    draw_error_heatmap(ax, abs_errors, bias_errors,
                       (8, 8), labels, cell_abs, cell_bias)

    # ── Top-right: 4x4 ────────────────────────────────────────────────────────
    ax = axes[0][1]
    ax.set_title("4×4 Group Errors", fontsize=10)
    group_abs_4x4  = {}
    group_bias_4x4 = {}
    labels_4x4     = []
    cell_abs_4x4   = []
    cell_bias_4x4  = []

    for sg in range(16):
        sr = sg // 4
        sc = sg % 4
        abs_vals  = []
        bias_vals = []
        for dr in range(2):
            for dc in range(2):
                z = (sr * 2 + dr) * 8 + (sc * 2 + dc)
                if abs_errors.get(z) is not None:
                    abs_vals.append(abs_errors[z])
                if bias_errors.get(z) is not None:
                    bias_vals.append(bias_errors[z])
        group_abs_4x4[sg]  = float(np.mean(abs_vals))  if abs_vals  else None
        group_bias_4x4[sg] = float(np.mean(bias_vals)) if bias_vals else None
        labels_4x4.append(f"G{sg}")
        cell_abs_4x4.append(group_abs_4x4[sg])
        cell_bias_4x4.append(group_bias_4x4[sg])

    draw_error_heatmap(ax, group_abs_4x4, group_bias_4x4,
                       (4, 4), labels_4x4, cell_abs_4x4, cell_bias_4x4)

    # ── Bottom-left: 2x2 ──────────────────────────────────────────────────────
    ax = axes[1][0]
    ax.set_title("2×2 Group Errors", fontsize=10)
    group_abs_2x2  = {}
    group_bias_2x2 = {}
    labels_2x2     = []
    cell_abs_2x2   = []
    cell_bias_2x2  = []

    for sg in range(4):
        sr = sg // 2
        sc = sg % 2
        abs_vals  = []
        bias_vals = []
        for dr in range(4):
            for dc in range(4):
                z = (sr * 4 + dr) * 8 + (sc * 4 + dc)
                if abs_errors.get(z) is not None:
                    abs_vals.append(abs_errors[z])
                if bias_errors.get(z) is not None:
                    bias_vals.append(bias_errors[z])
        group_abs_2x2[sg]  = float(np.mean(abs_vals))  if abs_vals  else None
        group_bias_2x2[sg] = float(np.mean(bias_vals)) if bias_vals else None
        labels_2x2.append(f"G{sg}")
        cell_abs_2x2.append(group_abs_2x2[sg])
        cell_bias_2x2.append(group_bias_2x2[sg])

    draw_error_heatmap(ax, group_abs_2x2, group_bias_2x2,
                       (2, 2), labels_2x2, cell_abs_2x2, cell_bias_2x2)

    # ── Bottom-right: Rings ────────────────────────────────────────────────────
    ax = axes[1][1]
    ax.set_title("Ring Group Errors", fontsize=10)

    ring_abs  = {0: [], 1: [], 2: [], 3: []}
    ring_bias = {0: [], 1: [], 2: [], 3: []}
    for z in range(64):
        ring = get_ring(z)
        if abs_errors.get(z) is not None:
            ring_abs[ring].append(abs_errors[z])
        if bias_errors.get(z) is not None:
            ring_bias[ring].append(bias_errors[z])

    ring_abs_means  = {r: float(np.mean(v)) if v else None
                       for r, v in ring_abs.items()}
    ring_bias_means = {r: float(np.mean(v)) if v else None
                       for r, v in ring_bias.items()}

    valid_bias = [v for v in ring_bias_means.values() if v is not None]
    min_bias   = min(valid_bias) if valid_bias else 0.0
    max_bias   = max(valid_bias) if valid_bias else 0.0

    for z in range(64):
        r        = z // 8
        c        = z % 8
        ring     = get_ring(z)
        bias_val = ring_bias_means.get(ring)
        plot_r   = 7 - r

        color = bias_colour(bias_val, min_bias, max_bias) \
                if bias_val is not None else (0.85, 0.85, 0.85, 1.0)

        rect = mpatches.FancyBboxPatch(
            (c, plot_r), 1, 1,
            boxstyle="square,pad=0",
            facecolor=color, edgecolor="white", linewidth=0.5
        )
        ax.add_patch(rect)

        abs_val  = ring_abs_means.get(ring)
        abs_text = f"{abs_val:.1f}" if abs_val is not None else "N/A"
        bias_text = f"b:{bias_val:+.1f}" if bias_val is not None else ""
        ax.text(c + 0.5, plot_r + 0.5, f"{abs_text}\n{bias_text}",
                ha="center", va="center", fontsize=6,
                color="white", fontweight="bold")

    # Ring boundary lines
    for ring in range(1, 4):
        o = ring
        ax.plot([o, 8-o, 8-o, o, o],
                [o, o, 8-o, 8-o, o],
                color="white", linewidth=1.5, linestyle="--")

    # Ring legend
    for ring in range(4):
        abs_val  = ring_abs_means.get(ring)
        bias_val = ring_bias_means.get(ring)
        label    = (f"Ring {ring}: |err|={abs_val:.1f} b={bias_val:+.1f}"
                    if abs_val is not None else f"Ring {ring}: N/A")
        ax.text(8.2, 7.5 - ring * 1.8, label,
                ha="left", va="center", fontsize=8)

    ax.set_xlim(0, 8)
    ax.set_ylim(0, 8)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")

    plt.tight_layout()
    save_path = os.path.join(output_path, "Experiment_Error_Heatmaps.png")
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close(fig)


# ── Overall validity heatmap ───────────────────────────────────────────────────

def plot_overall_validity_heatmap(data: dict, stats_data: dict, output_path: str):
    """
    Single 8x8 heatmap showing validity across all distances.
    Always valid zones: #52A83C
    Sometimes invalid: shades from #F6C6A7 (most valid) to #E97122 (least valid)
    """
    # Compute percent validity per zone across all distances
    zone_validity = {z: [] for z in range(64)}
    for dist_key, zone_data in data.items():
        for z in range(64):
            pct = stats_data[dist_key]["per_zone"]["percent_valid"][z]
            if pct is not None:
                zone_validity[z].append(pct)

    zone_mean_validity = {
        z: float(np.mean(v)) if v else None
        for z, v in zone_validity.items()
    }

    # Find min validity among non-100% zones for colour scaling
    invalid_vals = [v for v in zone_mean_validity.values()
                    if v is not None and v < 100.0]
    min_valid = min(invalid_vals) if invalid_vals else 0.0
    max_valid = 100.0

    fig, ax = plt.subplots(figsize=(8, 8))
    fig.suptitle("Overall Zone Validity", fontsize=12)

    for z in range(64):
        r      = z // 8
        c      = z % 8
        plot_r = 7 - r
        pct    = zone_mean_validity.get(z)

        if pct is None or pct >= 100.0:
            color    = (*mcolors.to_rgb("#52A83C"), 1.0)
            text     = "Always\nValid"
            txt_color = "white"
        else:
            # Interpolate between #F6C6A7 (most valid) and #E97122 (least valid)
            t     = (pct - min_valid) / (max_valid - min_valid + 1e-9)
            color_a = mcolors.to_rgb("#E97122")  # least valid
            color_b = mcolors.to_rgb("#F6C6A7")  # most valid among invalids
            rgb   = tuple(color_a[i] * (1 - t) + color_b[i] * t for i in range(3))
            color = (*rgb, 1.0)
            text  = f"{pct:.1f}%"
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
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")

    plt.tight_layout()
    save_path = os.path.join(output_path, "Experiment_Validity_Heatmap.png")
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close(fig)


# ── Distance Drift ─────────────────────────────────────────────
from scipy import stats as sp_stats
def plot_drift_analysis(data: dict, stats_data: dict, output_path: str):
    """
    Plots raw error (distance_mm - true_distance) over time per distance,
    with a linear fit overlaid. Annotates drift rate in mm per frame
    below the plot.
    """
    PALETTE = ["#E8708E", "#EDD020", "#E97122", "#52A83C",
               "#18A5AA", "#55A9D4", "#2E61A8"]

    fig = plt.figure(figsize=(14, 9))
    gs  = plt.GridSpec(2, 1, height_ratios=[10, 2], hspace=0.05)
    ax      = fig.add_subplot(gs[0])
    ax_text = fig.add_subplot(gs[1])
    ax_text.axis("off")

    drift_rates = {}

    for i, (dist_key, zone_data) in enumerate(sorted(data.items())):
        true_distance = extract_true_distance(dist_key)
        color         = PALETTE[i % len(PALETTE)]
        is_repeat     = i >= len(PALETTE)
        fit_linestyle = "--" if is_repeat else "-"

        zone_arrays = [zone_data["distance_mm"][z] for z in range(64)
                       if zone_data["distance_mm"][z] is not None]
        if not zone_arrays:
            continue

        min_len         = min(len(a) for a in zone_arrays)
        trimmed         = np.array([a[:min_len] for a in zone_arrays])
        mean_per_frame  = np.mean(trimmed, axis=0)
        error_per_frame = mean_per_frame - true_distance
        frames          = np.arange(len(error_per_frame))

        # Raw error — always solid, slight fade
        ax.plot(frames, error_per_frame,
                color=color, linewidth=0.8, alpha=0.6,
                linestyle="-")

        # Linear regression
        slope, intercept, _, _, _ = sp_stats.linregress(frames, error_per_frame)
        drift_rates[dist_key] = float(slope)

        # Linear fit — label without "(dashed)"
        ax.plot(frames, intercept + slope * frames,
                color=color, linewidth=2.0,
                linestyle=fit_linestyle, label=dist_key)

    ax.axhline(0, color="black", linewidth=1.0, linestyle="-", alpha=0.4)

    ax.set_ylabel("Error [mm]  (measured − true)")
    ax.set_xticks([])
    ax.set_xlabel("Time →")
    ax.set_title("Distance Error Over Time per Distance\n"
                 "(faint = raw data, solid = linear fit)",
                 pad=40)

    ax.legend(fontsize=7, loc="lower center",
              bbox_to_anchor=(0.5, 1.02),
              ncol=7, borderaxespad=0)

    # Drift rate annotation
    steepest_key   = max(drift_rates, key=lambda k: abs(drift_rates[k]))
    steepest_slope = drift_rates[steepest_key]
    avg_slope      = float(np.mean(list(drift_rates.values())))

    sorted_rates   = sorted(drift_rates.items())
    items_per_line = max(1, round(len(sorted_rates) / 2))
    rate_lines     = []
    for j in range(0, len(sorted_rates), items_per_line):
        chunk = sorted_rates[j:j + items_per_line]
        rate_lines.append("    ".join(f"{k}: {v:+.4f}" for k, v in chunk))

    summary_line = (f"Steepest: {steepest_key} ({steepest_slope:+.4f} mm/frame)"
                    f"     Avg drift: {avg_slope:+.4f} mm/frame")

    all_lines = rate_lines + [summary_line]

    # Pin annotation just below xlabel using figure transform
    # Start just below the xlabel and use consistent line spacing
    line_spacing = 0.045
    start_y      = -0.08
    for j, line in enumerate(all_lines):
        ax.annotate(line,
                    xy=(0.5, start_y - j * line_spacing),
                    xycoords="axes fraction",
                    ha="center", va="top", fontsize=7.5,
                    annotation_clip=False)

    save_path = os.path.join(output_path, "Experiment_Drift_Analysis.png")
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close(fig)

def plot_error_and_validity(data: dict, stats_data: dict, output_path: str):
    """
    Boxplot of distance error per distance with:
    - Overall zone validity per distance as shaded line
    - % error per distance as a second line
    - Box colour: #2E61A8 if mean error positive, #E8708E if negative
    - Dotted horizontal line at 0mm error
    """
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D

    sorted_keys   = sorted(data.items(), key=lambda x: extract_true_distance(x[0]))
    dist_labels   = []
    true_dists    = []
    error_arrays  = []
    mean_errors   = []
    validity_pcts = []
    pct_errors    = []

    for dist_key, zone_data in sorted_keys:
        true_distance = extract_true_distance(dist_key)
        dist_labels.append(dist_key)
        true_dists.append(true_distance)

        zone_arrays = [
            zone_data["distance_mm"][z]
            for z in range(64)
            if zone_data["distance_mm"][z] is not None
               and stats_data[dist_key]["per_zone"]["percent_valid"][z] is not None
               and stats_data[dist_key]["per_zone"]["percent_valid"][z] >= 100.0
        ]

        if zone_arrays:
            min_len        = min(len(a) for a in zone_arrays)
            trimmed        = np.array([a[:min_len] for a in zone_arrays])
            mean_per_frame = np.mean(trimmed, axis=0)
            error          = mean_per_frame - true_distance
        else:
            error = np.array([0.0])

        error_arrays.append(error)
        mean_err = float(np.mean(error))
        mean_errors.append(mean_err)

        pct_errors.append(100.0 * abs(mean_err) / true_distance
                          if true_distance != 0 else 0.0)

        n_always_valid = sum(
            1 for z in range(64)
            if stats_data[dist_key]["per_zone"]["percent_valid"][z] is not None
            and stats_data[dist_key]["per_zone"]["percent_valid"][z] >= 100.0
        )
        validity_pcts.append(100.0 * n_always_valid / 64)

    fig, ax1 = plt.subplots(figsize=(12, 7))
    ax2 = ax1.twinx()
    ax1.yaxis.set_minor_locator(plt.MultipleLocator(0.2))
    ax1.grid(True, which="major", color="grey", linewidth=0.5, alpha=0.3, zorder=0)
    ax1.grid(True, which="minor", color="grey", linewidth=0.3, alpha=0.2, zorder=0)

    # Force ax2 behind ax1
    ax1.set_zorder(ax2.get_zorder() + 1)
    ax1.patch.set_visible(False)  # keep ax1 background transparent so ax2 shows through

    x_positions = np.arange(len(dist_labels))

    # ── Validity shading and line on ax2 ──────────────────────────────────────
    ax2.fill_between(x_positions, 0, validity_pcts,
                     color="#A3DBDD", alpha=0.5)
    ax2.plot(x_positions, validity_pcts,
             color="#18A5AA", linewidth=2.0)

    # ── % error line on ax2 ───────────────────────────────────────────────────
    ax2.plot(x_positions, pct_errors,
             color="#E97122", linewidth=2.0)

    # ── Zero error reference line on ax1 ──────────────────────────────────────
    ax1.axhline(0, color="black", linewidth=1.0,
                linestyle=":", alpha=0.7)

    # ── Boxplots on ax1 ───────────────────────────────────────────────────────
    for i, (error, mean_err) in enumerate(zip(error_arrays, mean_errors)):
        color = "#2E61A8" if mean_err >= 0 else "#E8708E"
        ax1.boxplot(error,
                    positions=[x_positions[i]],
                    widths=0.5,
                    patch_artist=True,
                    manage_ticks=False,
                    medianprops=dict(color="white", linewidth=1.5),
                    whiskerprops=dict(color=color, linewidth=1.2),
                    capprops=dict(color=color, linewidth=1.2),
                    flierprops=dict(marker=".", markersize=3,
                                    markerfacecolor=color, alpha=0.4),
                    boxprops=dict(facecolor=color, color=color,
                                  linewidth=1.2))

    # ── Axis formatting ───────────────────────────────────────────────────────
    ax1.set_ylabel("Error [mm]  (measured − true)", fontsize=10)
    ax1.set_xticks(x_positions)
    ax1.set_xticklabels(dist_labels, fontsize=9)
    ax1.set_xlabel("Distance", fontsize=10)
    ax1.set_title("Distance Error and Zone Validity per Distance", fontsize=12)

    ax2.set_ylabel("% Zones Always Valid  /  % Error", fontsize=10)
    ax2.tick_params(axis="y", colors="black")

    # ── Legend ────────────────────────────────────────────────────────────────
    legend_elements = [
        Patch(facecolor="#2E61A8", label="Positive mean error"),
        Patch(facecolor="#E8708E", label="Negative mean error"),
        Line2D([0], [0], color="#18A5AA", linewidth=2.0,
               label="% Zones always valid"),
        Line2D([0], [0], color="#E97122", linewidth=2.0,
               label="% Error"),
    ]
    ax1.legend(handles=legend_elements, fontsize=8,
               loc="upper left", framealpha=0.9)

    plt.tight_layout()
    save_path = os.path.join(output_path, "Experiment_Error_Validity.png")
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close(fig)

# ── Master function ────────────────────────────────────────────────────

def generate_all_zone_plots(data: dict, stats_data: dict, output_path: str) -> str:
    """
    Runs all plot functions and saves to a timestamped folder.
    """
    plots_folder = setup_output_folder(output_path)
    print(f"Saving plots to: {plots_folder}")

    print("Generating tiled boxplots...")
    plot_tiled_boxplots(data, stats_data, plots_folder)

    print("Generating tiled validity plots...")
    plot_tiled_validity(data, stats_data, plots_folder)

    print("Generating total boxplots...")
    plot_total_boxplots(data, stats_data, plots_folder)

    print("Generating tiled heatmaps...")
    plot_tiled_heatmaps(data, stats_data, plots_folder)

    print("Generating experiment error heatmaps...")
    plot_error_heatmaps(data, stats_data, plots_folder)

    print("Generating overall validity heatmap...")
    plot_overall_validity_heatmap(data, stats_data, plots_folder)

    print("Generating drift analysis...")
    plot_drift_analysis(data, stats_data, plots_folder)

    print("Generating error and validity plot...")
    plot_error_and_validity(data, stats_data, plots_folder)

    print("Done!")
    return plots_folder
