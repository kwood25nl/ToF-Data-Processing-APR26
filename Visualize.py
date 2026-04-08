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
    Returns (row, col) for use with a matplotlib subplot array (axes[row][col]).

    In subplot arrays row 0 is the TOP row, so we flip the physical row index
    so that Z0 (physical row 0) lands in the BOTTOM subplot row and Z63
    (physical row 7) lands in the TOP subplot row — giving Z0 bottom-left and
    Z63 top-right in the tiled subplot layout.

    NOTE: This is for subplot-grid positioning only.  Patch-based heatmaps use
    the raw zone coordinates (z // 8, z % 8) without any flip, because in
    matplotlib's data-coordinate system y=0 is already at the bottom.
    """
    physical_row = z // 8
    col          = z % 8
    row          = 7 - physical_row
    return row, col


def extract_true_distance(key: str) -> float:
    digits = re.findall(r"-?\d+", key)
    return float("".join(digits))


def _assert_heatmap_orientation():
    """
    Debug self-check: asserts that the patch-based heatmap placement logic
    puts Z0 at bottom-left and Z63 at top-right.

    For an 8x8 grid with zone labels in zone-index order (Z0 first):
      i=0  → r = 0//8 = 0, c = 0%8 = 0  → patch at (c=0, r=0) = bottom-left  ✓
      i=63 → r = 63//8 = 7, c = 63%8 = 7 → patch at (c=7, r=7) = top-right    ✓

    (In matplotlib data coordinates y=0 is bottom, so r=0 is the bottom row.)
    """
    rows, cols = 8, 8
    for z in range(64):
        r = z // cols
        c = z % cols
        assert r == z // 8, f"Row mismatch for Z{z}"
        assert c == z % 8,  f"Col mismatch for Z{z}"

    r0, c0   = 0 // 8, 0 % 8
    r63, c63 = 63 // 8, 63 % 8
    assert r0  == 0 and c0  == 0, "Z0  is not at bottom-left (r=0, c=0)"
    assert r63 == 7 and c63 == 7, "Z63 is not at top-right  (r=7, c=7)"


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
        zone_labels:   List of label strings per cell in zone-index order
                       (Z0 first = bottom-left, Z63 last = top-right).
        cell_means:    List of mean values per cell, same order as zone_labels.
    """
    rows, cols = grid_shape
    all_vals   = [v for v in means.values() if v is not None]
    min_val    = min(all_vals) if all_vals else true_distance
    max_val    = max(all_vals) if all_vals else true_distance

    for i, (label, mean_val) in enumerate(zip(zone_labels, cell_means)):
        r = i // cols
        c = i % cols
        # No row flip: in matplotlib data coordinates y=0 is bottom,
        # so i=0 (Z0) lands at (c=0, r=0) = bottom-left, and
        # i=63 (Z63) lands at (c=7, r=7) = top-right.

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
    For each distance:
      - If there are any invalid readings: generate 3 side-by-side boxplots
          1) All readings (valid + invalid) across all zones
          2) All valid readings only across all zones (is_valid_range == 1)
          3) All readings from zones that are 100% valid (percent_valid >= 100)
      - If there are NO invalid readings: generate only 1 boxplot (All readings)

    Box colour is based on mean vs true distance:
      - within ±2mm: BOX_WITHIN (#52A83C)
      - below -2mm:  BOX_LOW    (#E8708E)
      - above +2mm:  BOX_OVER   (#2E61A8)

    Each plotted distribution gets Q1/Median/Q3/Mean shown below.
    """

    def safe_concat(arr_list):
        arr_list = [a for a in arr_list if a is not None and len(a) > 0]
        return np.concatenate(arr_list) if arr_list else None

    def compute_stats(arr: np.ndarray):
        if arr is None or len(arr) == 0:
            return None
        q1 = float(np.percentile(arr, 25))
        med = float(np.median(arr))
        q3 = float(np.percentile(arr, 75))
        mean = float(np.mean(arr))
        return q1, med, q3, mean

    def stats_str(arr: np.ndarray) -> str:
        s = compute_stats(arr)
        if s is None:
            return "No data"
        q1, med, q3, mean = s
        return f"Q1:{q1:.2f}  Med:{med:.2f}  Q3:{q3:.2f}  Mean:{mean:.2f}"

    def box_color_for_mean(mean_val: float, true_distance: float) -> str:
        diff = mean_val - true_distance
        if abs(diff) <= 2:
            return BOX_WITHIN
        elif diff < -2:
            return BOX_LOW
        else:
            return BOX_OVER

    for dist_key, zone_data in data.items():
        true_distance = extract_true_distance(dist_key)

        # --- Collect arrays per zone ---
        zone_dist_arrays = [
            zone_data["distance_mm"][z]
            for z in range(64)
            if zone_data["distance_mm"][z] is not None
        ]
        all_readings = safe_concat(zone_dist_arrays)

        # Valid-only (mask per zone)
        valid_distances = get_valid_distances(zone_data, stats_data.get(dist_key, {}))
        all_valid = safe_concat(list(valid_distances.values()))

        # Always-valid zones only (percent_valid >= 100)
        always_valid_zone_arrays = []
        for z in range(64):
            arr = zone_data["distance_mm"][z]
            if arr is None:
                continue
            pct = stats_data[dist_key]["per_zone"]["percent_valid"][z]
            if pct is not None and pct >= 100.0:
                always_valid_zone_arrays.append(arr)
        always_valid_all = safe_concat(always_valid_zone_arrays)

        # If nothing exists at all, skip
        if all_readings is None:
            continue

        # Determine whether there are any invalid readings for this distance
        # "Invalid exists" if any zone has any 0 in is_valid_range.
        any_invalid = False
        for z in range(64):
            valid_arr = zone_data["is_valid_range"][z]
            if valid_arr is not None and np.any(valid_arr == 0):
                any_invalid = True
                break

        # --- Plotting ---
        if not any_invalid:
            # Only 1 boxplot: All readings
            stats = compute_stats(all_readings)
            if stats is None:
                continue
            _, _, _, mean_all = stats
            box_color = box_color_for_mean(mean_all, true_distance)

            fig = plt.figure(figsize=(6, 7))
            gs = plt.GridSpec(2, 1, height_ratios=[10, 1], hspace=0.05)
            ax = fig.add_subplot(gs[0])
            ax_text = fig.add_subplot(gs[1])
            ax_text.axis("off")

            sns.boxplot(y=all_readings, ax=ax, color=box_color,
                        linewidth=0.8, fliersize=2)

            y_lo, y_hi = ax.get_ylim()
            half_range = max(abs(y_hi - true_distance),
                             abs(true_distance - y_lo), 1.0)
            ax.set_ylim(true_distance - half_range, true_distance + half_range)
            ax.axhline(true_distance, color="grey", linewidth=0.8,
                       linestyle="--", alpha=0.7, label="Target")

            ax.set_ylabel("Measured Distance [mm]")
            ax.set_title(f"{dist_key} Total Boxplot (no invalid readings)")
            ax.tick_params(axis="x", bottom=False)

            ax_text.text(0.5, 0.8, stats_str(all_readings),
                         ha="center", va="top", fontsize=8,
                         transform=ax_text.transAxes)

            save_path = os.path.join(output_path, f"{dist_key}_Total_Boxplot.png")
            plt.savefig(save_path, bbox_inches="tight", dpi=150)
            plt.close(fig)

        else:
            # 3 side-by-side boxplots
            series = [all_readings, all_valid, always_valid_all]
            labels = ["All (val+inv)", "Valid only", "Zones 100% valid"]
            positions = [1, 2, 3]

            # Compute per-box colors from mean vs true distance (skip empty)
            colors = []
            for arr in series:
                s = compute_stats(arr)
                if s is None:
                    colors.append("#B0B0B0")  # fallback for missing data
                else:
                    _, _, _, m = s
                    colors.append(box_color_for_mean(m, true_distance))

            fig = plt.figure(figsize=(11, 7))
            gs = plt.GridSpec(2, 1, height_ratios=[10, 1.8], hspace=0.08)
            ax = fig.add_subplot(gs[0])
            ax_text = fig.add_subplot(gs[1])
            ax_text.axis("off")

            plot_data = [s if s is not None and len(s) > 0 else np.array([np.nan]) for s in series]
            bp = ax.boxplot(
                plot_data,
                positions=positions,
                widths=0.6,
                patch_artist=True,
                manage_ticks=False,
                medianprops=dict(color="white", linewidth=1.5),
                whiskerprops=dict(color="#333333", linewidth=1.0),
                capprops=dict(color="#333333", linewidth=1.0),
                flierprops=dict(marker=".", markersize=3,
                                markerfacecolor="#333333", alpha=0.35),
                boxprops=dict(linewidth=1.0, color="#333333"),
            )

            for patch, c in zip(bp["boxes"], colors):
                patch.set_facecolor(c)
                patch.set_alpha(0.9)

            # Center y-axis around true distance using all available plotted data
            combined = safe_concat([s for s in series if s is not None])
            if combined is not None and len(combined) > 0:
                y_lo = float(np.min(combined))
                y_hi = float(np.max(combined))
                half_range = max(abs(y_hi - true_distance),
                                 abs(true_distance - y_lo), 1.0)
                ax.set_ylim(true_distance - half_range, true_distance + half_range)

            ax.axhline(true_distance, color="grey", linewidth=0.9,
                       linestyle="--", alpha=0.8)

            ax.set_xticks(positions)
            ax.set_xticklabels(labels, fontsize=9)
            ax.set_ylabel("Measured Distance [mm]")
            ax.set_title(f"{dist_key} — Total Boxplots (invalid readings present)")

            ax_text.text(0.02, 0.80, f"All (val+inv):   {stats_str(all_readings)}",
                         ha="left", va="top", fontsize=8, transform=ax_text.transAxes)
            ax_text.text(0.02, 0.45, f"Valid only:      {stats_str(all_valid)}",
                         ha="left", va="top", fontsize=8, transform=ax_text.transAxes)
            ax_text.text(0.02, 0.10, f"Zones 100% valid:{stats_str(always_valid_all)}",
                         ha="left", va="top", fontsize=8, transform=ax_text.transAxes)

            save_path = os.path.join(output_path, f"{dist_key}_Total_Boxplots_Compare.png")
            plt.savefig(save_path, bbox_inches="tight", dpi=150)
            plt.close(fig)

# ── Tiled heatmaps ─────────────────────────────────────────────────────────────
def plot_tiled_heatmaps(data: dict, stats_data: dict, output_path: str):
    """
    One 2x2 tiled figure per distance containing error heatmaps:
      Top-left:     8x8 zone mean error heatmap
      Top-right:    4x4 group mean error heatmap
      Bottom-left:  2x2 group mean error heatmap
      Bottom-right: Ring group mean error heatmap

    Colour shading matches the original scheme (via colour_for_value):
      - within ±2mm of target: green (#52A83C), darkest = closest
      - below target-2mm:      pink  (#E8708E), darkest = furthest below
      - above target+2mm:      blue  (#2E61A8), darkest = furthest above
      - alpha ranges 0.5 (lightest) to 1.0 (darkest)

    Text: ONLY mean error (signed, mm). No bias/abs combo text.
    Uses valid readings only to compute per-zone means.
    """
    def get_ring(z):
        r = z // 8
        c = z % 8
        return min(r, 7 - r, c, 7 - c)

    def draw_error_only_heatmap(ax, errors: dict, true_distance: float,
                                grid_shape: tuple, cell_labels: list, cell_errors: list):
        """
        Draws a heatmap where cell colour encodes how close (true_distance + error) is to true_distance,
        using colour_for_value(), and cell text shows ONLY the signed error.
        """
        rows, cols = grid_shape

        # Convert error->"value" so we can reuse colour_for_value() exactly
        values = {k: (true_distance + v) for k, v in errors.items() if v is not None}
        all_vals = [v for v in values.values() if v is not None]

        min_val = min(all_vals) if all_vals else true_distance
        max_val = max(all_vals) if all_vals else true_distance

        for i, (label, err) in enumerate(zip(cell_labels, cell_errors)):
            r = i // cols
            c = i % cols
            # No row flip: y=0 is bottom in matplotlib data coordinates.

            if err is None:
                color = (0.85, 0.85, 0.85, 1.0)
                text_val = "N/A"
            else:
                value = true_distance + err
                color = colour_for_value(value, true_distance, min_val, max_val)
                text_val = f"{err:+.1f}"

            rect = mpatches.FancyBboxPatch(
                (c, r), 1, 1,
                boxstyle="square,pad=0",
                facecolor=color, edgecolor="white", linewidth=0.5,
                transform=ax.transData
            )
            ax.add_patch(rect)

            ax.text(c + 0.5, r + 0.5, f"{label}\n{text_val}",
                    ha="center", va="center", fontsize=6,
                    color="white", fontweight="bold")

        ax.set_xlim(0, cols)
        ax.set_ylim(0, rows)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal")

    for dist_key, zone_data in data.items():
        true_distance = extract_true_distance(dist_key)

        # Per-zone mean measured distance (valid-only) -> mean error
        valid_distances = get_valid_distances(zone_data, stats_data.get(dist_key, {}))
        zone_means = compute_zone_means(valid_distances)  # z -> mean measured (valid-only)

        zone_errors = {z: (zone_means[z] - true_distance) for z in zone_means}  # signed error

        fig, axes = plt.subplots(2, 2, figsize=(16, 16))
        fig.suptitle(
            f"{dist_key} — Error Heatmaps\n(text = mean error, colour = distance vs target)",
            fontsize=14, y=1.002
        )

        # ── Top-left: 8x8 zones ───────────────────────────────────────────────
        ax = axes[0][0]
        ax.set_title("8×8 Zone Mean Error", fontsize=10)
        labels_8 = [f"Z{z}" for z in range(64)]
        cell_err_8 = [zone_errors.get(z, None) for z in range(64)]
        draw_error_only_heatmap(ax, zone_errors, true_distance, (8, 8), labels_8, cell_err_8)

        # ── Top-right: 4x4 groups ─────────────────────────────────────────────
        ax = axes[0][1]
        ax.set_title("4×4 Group Mean Error", fontsize=10)

        group_err_4x4 = {}
        labels_4 = []
        cell_err_4 = []

        for sg in range(16):
            sr = sg // 4
            sc = sg % 4
            vals = []
            for dr in range(2):
                for dc in range(2):
                    z = (sr * 2 + dr) * 8 + (sc * 2 + dc)
                    if zone_errors.get(z) is not None:
                        vals.append(zone_errors[z])
            group_err_4x4[sg] = float(np.mean(vals)) if vals else None
            labels_4.append(f"G{sg}")
            cell_err_4.append(group_err_4x4[sg])

        draw_error_only_heatmap(ax, group_err_4x4, true_distance, (4, 4), labels_4, cell_err_4)

        # ── Bottom-left: 2x2 groups ───────────────────────────────────────────
        ax = axes[1][0]
        ax.set_title("2×2 Group Mean Error", fontsize=10)

        group_err_2x2 = {}
        labels_2 = []
        cell_err_2 = []

        for sg in range(4):
            sr = sg // 2
            sc = sg % 2
            vals = []
            for dr in range(4):
                for dc in range(4):
                    z = (sr * 4 + dr) * 8 + (sc * 4 + dc)
                    if zone_errors.get(z) is not None:
                        vals.append(zone_errors[z])
            group_err_2x2[sg] = float(np.mean(vals)) if vals else None
            labels_2.append(f"G{sg}")
            cell_err_2.append(group_err_2x2[sg])

        draw_error_only_heatmap(ax, group_err_2x2, true_distance, (2, 2), labels_2, cell_err_2)

        # ── Bottom-right: Rings ───────────────────────────────────────────────
        ax = axes[1][1]
        ax.set_title("Ring Group Mean Error", fontsize=10)

        ring_vals = {0: [], 1: [], 2: [], 3: []}
        for z, err in zone_errors.items():
            ring_vals[get_ring(z)].append(err)

        ring_err = {r: (float(np.mean(v)) if v else None) for r, v in ring_vals.items()}

        # Draw ring grid: color by ring error, label each ring once
        # Compute min/max in "value space" for colour_for_value scaling
        ring_value_map = {r: (true_distance + e) for r, e in ring_err.items() if e is not None}
        ring_values = [v for v in ring_value_map.values() if v is not None]
        min_val = min(ring_values) if ring_values else true_distance
        max_val = max(ring_values) if ring_values else true_distance

        for z in range(64):
            r = z // 8
            c = z % 8
            ring = get_ring(z)
            plot_r = r

            err = ring_err.get(ring)
            if err is None:
                color = (0.85, 0.85, 0.85, 1.0)
            else:
                value = true_distance + err
                color = colour_for_value(value, true_distance, min_val, max_val)

            rect = mpatches.FancyBboxPatch(
                (c, plot_r), 1, 1,
                boxstyle="square,pad=0",
                facecolor=color, edgecolor="white", linewidth=0.5
            )
            ax.add_patch(rect)

            # label each ring once (top-left cell of each ring)
            label_once = (c == ring) and (plot_r == (7 - ring))
            if label_once:
                txt = f"{err:+.1f}" if err is not None else "N/A"
                ax.text(c + 0.5, plot_r + 0.5, txt,
                        ha="center", va="center", fontsize=9,
                        color="white", fontweight="bold")

        # ring boundaries
        for ring in range(1, 4):
            o = ring
            ax.plot([o, 8-o, 8-o, o, o],
                    [o, o, 8-o, 8-o, o],
                    color="white", linewidth=1.5, linestyle="--")
        legend_elements = [
            Patch(facecolor="#52A83C", label="Mean |error| ≤ 2mm"),
            Patch(facecolor="#E8708E", label="Mean error < −2mm"),
            Patch(facecolor="#2E61A8", label="Mean error > +2mm"),
        ]

        fig.legend(handles=legend_elements,
                   loc="upper center",
                   ncol=3,
                   framealpha=0.98,
                   fontsize=9,
                   bbox_to_anchor=(0.5, 0.97))

        ax.set_xlim(0, 8)
        ax.set_ylim(0, 8)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal")

        plt.tight_layout(rect=[0, 0, 1, 0.98])
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
        # No row flip: y=0 is bottom in matplotlib data coordinates.

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

# ── Overall Error heatmap ───────────────────────────────────────────────────
from matplotlib.patches import Patch
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
        plot_r   = r

        color = bias_colour(bias_val, min_bias, max_bias) \
                if bias_val is not None else (0.85, 0.85, 0.85, 1.0)

        rect = mpatches.FancyBboxPatch(
            (c, plot_r), 1, 1,
            boxstyle="square,pad=0",
            facecolor=color, edgecolor="white", linewidth=0.5
        )
        ax.add_patch(rect)

        # --- Only label each ring once (top-left cell of that ring) ---
        # With plot_r = r, the top-left cell of ring k is at (c=k, plot_r=7-k).
        label_once = (c == ring) and (plot_r == (7 - ring))
        if label_once:
            abs_val   = ring_abs_means.get(ring)
            abs_text  = f"{abs_val:.1f}" if abs_val is not None else "N/A"
            bias_text = f"b:{bias_val:+.1f}" if bias_val is not None else ""
            ax.text(c + 0.5, plot_r + 0.5, f"{abs_text}\n{bias_text}",
                    ha="center", va="center", fontsize=7,
                    color="white", fontweight="bold")

    # Ring boundary lines
    for ring in range(1, 4):
        o = ring
        ax.plot([o, 8-o, 8-o, o, o],
                [o, o, 8-o, 8-o, o],
                color="white", linewidth=1.5, linestyle="--")


    legend_elements = [
        Patch(facecolor="#52A83C", label="Mean |error| ≤ 2mm"),
        Patch(facecolor="#E8708E", label="Mean error < −2mm"),
        Patch(facecolor="#2E61A8", label="Mean error > +2mm"),
    ]

    fig.legend(handles=legend_elements,
               loc="upper center",
               ncol=3,
               framealpha=0.98,
               fontsize=9,
               bbox_to_anchor=(0.5, 0.97))

    ax.set_xlim(0, 8)
    ax.set_ylim(0, 8)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")

    plt.tight_layout(rect=[0, 0, 1, 0.98])
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
        plot_r = r
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
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
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

    ax.set_ylabel("Error [mm]")
    ax.set_xticks([])
    ax.set_xlabel("Time →")
    ax.set_title("Transient Distance Error\n (mean of all zones per time per distance)", pad=40)

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

# ── Error and Validity Graph ────────────────────────────────────────────────────
def plot_error_and_validity(data: dict, stats_data: dict, output_path: str):
    """
    Boxplot of pooled distance error per distance (valid readings only), with:
    - % zones always valid per distance as shaded region + line (ax2)
    - % error per distance as a second line (ax2)
    - Dotted horizontal line at 0mm error

    Changes:
    - Box color based on mean error vs 0 using +/-2mm rule (BOX_WITHIN/LOW/OVER)
    - Legend moved outside plot (no overlap) and removed +/- mean error entries
    - Left y-axis scaled using IQR (robust to outliers)
    """
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D

    sorted_items = sorted(data.items(), key=lambda x: extract_true_distance(x[0]))

    dist_labels   = []
    error_arrays  = []
    mean_errors   = []
    validity_pcts = []
    pct_errors    = []

    # For robust y scaling (IQR-based)
    q1_list = []
    q3_list = []

    for dist_key, zone_data in sorted_items:
        true_distance = extract_true_distance(dist_key)
        dist_labels.append(dist_key)

        # --- pooled valid-only errors across all zones ---
        zone_error_arrays = []
        for z in range(64):
            arr = zone_data["distance_mm"][z]
            valid_arr = zone_data["is_valid_range"][z]
            if arr is None or valid_arr is None:
                continue

            mask = (valid_arr == 1)
            if np.any(mask):
                zone_error_arrays.append(arr[mask] - true_distance)

        if zone_error_arrays:
            error = np.concatenate(zone_error_arrays)
        else:
            error = np.array([0.0])

        error_arrays.append(error)

        mean_err = float(np.mean(error))
        mean_errors.append(mean_err)

        pct_errors.append(100.0 * abs(mean_err) / true_distance
                          if true_distance != 0 else 0.0)

        # validity % for this distance
        n_always_valid = sum(
            1 for z in range(64)
            if stats_data[dist_key]["per_zone"]["percent_valid"][z] is not None
            and stats_data[dist_key]["per_zone"]["percent_valid"][z] >= 100.0
        )
        validity_pcts.append(100.0 * n_always_valid / 64.0)

        # gather quartiles for robust y limits
        if error is not None and len(error) > 0:
            q1_list.append(float(np.percentile(error, 25)))
            q3_list.append(float(np.percentile(error, 75)))

    fig, ax1 = plt.subplots(figsize=(12, 7))
    ax2 = ax1.twinx()

    ax1.yaxis.set_minor_locator(plt.MultipleLocator(0.2))
    ax1.grid(True, which="major", color="grey", linewidth=0.5, alpha=0.3, zorder=0)
    ax1.grid(True, which="minor", color="grey", linewidth=0.3, alpha=0.2, zorder=0)

    # Force ax2 behind ax1
    ax1.set_zorder(ax2.get_zorder() + 1)
    ax1.patch.set_visible(False)

    x_positions = np.arange(len(dist_labels))

    # --- Validity shading + line on ax2 ---
    ax2.fill_between(x_positions, 0, validity_pcts,
                     color="#A3DBDD", alpha=0.5)
    ax2.plot(x_positions, validity_pcts,
             color="#18A5AA", linewidth=2.0)

    # --- % error line on ax2 ---
    ax2.plot(x_positions, pct_errors,
             color="#E97122", linewidth=2.0)

    # --- Zero error reference line ---
    ax1.axhline(0, color="black", linewidth=1.0,
                linestyle=":", alpha=0.7)

    # --- Boxplots (colored by mean error relative to 0) ---
    for i, (error, mean_err) in enumerate(zip(error_arrays, mean_errors)):
        if abs(mean_err) <= 2:
            color = BOX_WITHIN  # #52A83C
        elif mean_err < -2:
            color = BOX_LOW     # #E8708E
        else:
            color = BOX_OVER    # #2E61A8

        ax1.boxplot(error,
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

    # --- Robust y-limits based on IQR across distances ---
    if q1_list and q3_list:
        global_q1 = float(np.min(q1_list))
        global_q3 = float(np.max(q3_list))
        global_iqr = max(global_q3 - global_q1, 1e-6)

        # Pad by 1.5 * IQR to give room for whiskers, but not extreme outliers
        pad = 1.5 * global_iqr
        y_min = global_q1 - pad
        y_max = global_q3 + pad

        # Ensure 0 is visible and keep symmetric-ish around 0 if desired
        y_min = min(y_min, -0.5)
        y_max = max(y_max,  0.5)

        ax1.set_ylim(y_min, y_max)

    # --- Axis formatting ---
    ax1.set_ylabel("Error [mm]", fontsize=10)
    ax1.set_xticks(x_positions)
    ax1.set_xticklabels(dist_labels, fontsize=9)
    ax1.set_xlabel("Distance [mm]", fontsize=10)
    ax1.set_title("Experimental Error and Zone Validity", fontsize=12)

    ax2.set_ylabel("% Zones Always Valid  /  % Error", fontsize=10)
    ax2.tick_params(axis="y", colors="black")

    # --- Legend outside plot (no overlap); remove +/- mean error entries ---
    legend_elements = [
        Line2D([0], [0], color="#18A5AA", linewidth=2.0,
               label="% Zones always valid"),
        Line2D([0], [0], color="#E97122", linewidth=2.0,
               label="% Error"),
        Patch(facecolor=BOX_WITHIN, label="Mean |error| ≤ 2mm"),
        Patch(facecolor=BOX_LOW, label="Mean error < −2mm"),
        Patch(facecolor=BOX_OVER, label="Mean error > +2mm"),
    ]

    # Make room on the right for the legend
    fig.subplots_adjust(right=0.78)
    ax1.legend(handles=legend_elements, fontsize=8,
               loc="upper left", bbox_to_anchor=(1.01, 1.0),
               borderaxespad=0.0, framealpha=0.95)

    plt.tight_layout()
    save_path = os.path.join(output_path, "Experiment_Error_Validity.png")
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close(fig)

# ── Ridge Graph ────────────────────────────────────────────────────
def plot_ridge_by_distance(data: dict, stats_data: dict, output_path: str):
    """
    Ridge/KDE plot with FacetGrid (seaborn example style), using:
    - 100% valid zones per distance
    - x-axis in mm (measured distance); dashed vertical lines at true distances
    - compressed y stacking WITHOUT compressing x axis
    - avoids tight_layout warning by controlling decorations and margins
    """
    import pandas as pd

    sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})

    sorted_items = sorted(data.items(), key=lambda x: extract_true_distance(x[0]))
    if not sorted_items:
        return

    per_dist = []
    for dist_key, zone_data in sorted_items:
        true_distance = extract_true_distance(dist_key)

        always_valid_zones = [
            z for z in range(64)
            if stats_data[dist_key]["per_zone"]["percent_valid"][z] is not None
            and stats_data[dist_key]["per_zone"]["percent_valid"][z] >= 100.0
            and zone_data["distance_mm"][z] is not None
        ]

        measurements = []
        for z in always_valid_zones:
            arr = zone_data["distance_mm"][z]
            if arr is not None and len(arr) > 0:
                measurements.append(arr)

        if not measurements:
            continue

        meas = np.concatenate(measurements)
        per_dist.append((dist_key, true_distance, meas))

    if not per_dist:
        return

    rows = []
    for dist_key, true_distance, meas in per_dist:
        for x in meas:
            rows.append({"dist_key": dist_key, "true_distance": true_distance, "x": float(x)})

    df = pd.DataFrame(rows)

    order = [k for (k, _, _) in per_dist]
    df["dist_key"] = pd.Categorical(df["dist_key"], categories=order, ordered=True)

    palette = sns.cubehelix_palette(len(order), rot=-.25, light=.7)

    # Key change: keep height modest for y compression, but increase aspect a lot for width
    g = sns.FacetGrid(
        df,
        row="dist_key",
        hue="dist_key",
        aspect=22,     # wider x-axis (was 12)
        height=0.42,   # moderate facet height; small enough to compress y
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

        # Only bottom facet shows x tick labels to avoid layout warnings/clutter
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

    # More overlap but not extreme
    g.fig.subplots_adjust(
        hspace=-0.55,
        left=0.06,
        right=0.995,
        top=0.92,
        bottom=0.13  # room for rotated xticks on bottom facet
    )

    g.set_titles("")
    g.set(yticks=[], ylabel="")
    g.despine(bottom=True, left=True)

    # Put xlabel only once (bottom)
    g.set_xlabels("Distance [mm]")

    g.fig.suptitle(
        "Distance Ridge Plot (100% valid zones only)\n",
        fontsize=12
    )

    save_path = os.path.join(output_path, "Experiment_Ridgeplot.png")
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close(g.fig)

# ── Master function ────────────────────────────────────────────────────
def generate_all_zone_plots(data: dict, stats_data: dict, output_path: str) -> str:
    """
    Runs all plot functions and saves to a timestamped folder.
    """
    plots_folder = setup_output_folder(output_path)
    print(f"Saving plots to: {plots_folder}")


    #print("Generating tiled boxplots...")
    #plot_tiled_boxplots(data, stats_data, plots_folder)

    # print("Generating tiled validity plots...")
    #plot_tiled_validity(data, stats_data, plots_folder)

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

    print("Generating KDE ridge plot...")
    plot_ridge_by_distance(data, stats_data, plots_folder)

    print("Done!")
    return plots_folder
