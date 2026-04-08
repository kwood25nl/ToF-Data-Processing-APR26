import os
import re
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from scipy import stats as sp_stats

from Calculations import trim_tof_data, analyse_tof_data
from ImportData import load_tof_data


# ── Colours ────────────────────────────────────────────────────────────────────
PALETTE          = ["#E8708E", "#EDD020", "#E97122", "#52A83C",
                    "#18A5AA", "#55A9D4", "#2E61A8"]
BOX_WITHIN       = "#52A83C"
BOX_OVER         = "#2E61A8"
BOX_LOW          = "#E8708E"
VALID_COLOR      = "#52A83C"
INVALID_COLOR    = "#E97122"
ALWAYS_VALID_BG  = "#BADCB1"


# ── Helpers ────────────────────────────────────────────────────────────────────

def extract_true_distance(key: str) -> float:
    digits = re.findall(r"-?\d+", key)
    return float("".join(digits))


def zone_to_grid(z: int):
    physical_row = z // 8
    col          = z % 8
    row          = 7 - physical_row
    return row, col


def get_valid_distances(zone_data: dict, stats_data_dist: dict) -> dict:
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
    return {z: float(np.mean(arr)) for z, arr in valid_distances.items()}


def colour_for_value(value, true_distance, min_val, max_val):
    diff = value - true_distance
    if abs(diff) <= 2:
        base   = mcolors.to_rgb("#52A83C")
        extent = max(abs(min_val - true_distance),
                     abs(max_val - true_distance), 2)
        alpha  = 1.0 - 0.5 * (abs(diff) / extent)
    elif diff < -2:
        base       = mcolors.to_rgb("#E8708E")
        most_below = min(min_val - true_distance, -2)
        alpha      = 1.0 - 0.5 * ((diff - most_below) /
                                   (-2 - most_below + 1e-9))
        alpha      = np.clip(alpha, 0.5, 1.0)
    else:
        base       = mcolors.to_rgb("#2E61A8")
        most_above = max(max_val - true_distance, 2)
        alpha      = 1.0 - 0.5 * ((most_above - diff) /
                                   (most_above - 2 + 1e-9))
        alpha      = np.clip(alpha, 0.5, 1.0)
    return (*base, alpha)


def bias_colour(bias, min_bias, max_bias):
    extreme   = max(abs(min_bias), abs(max_bias))
    threshold = extreme * 0.2
    if abs(bias) <= threshold:
        base  = mcolors.to_rgb("#52A83C")
        alpha = 1.0 - 0.5 * (abs(bias) / (threshold + 1e-9))
    elif bias > threshold:
        base  = mcolors.to_rgb("#2E61A8")
        alpha = 0.5 + 0.5 * (bias / (max_bias + 1e-9))
    else:
        base  = mcolors.to_rgb("#E8708E")
        alpha = 0.5 + 0.5 * (abs(bias) / (abs(min_bias) + 1e-9))
    return (*base, float(np.clip(alpha, 0.5, 1.0)))


def draw_heatmap(ax, means, true_distance, grid_shape, zone_labels, cell_means):
    rows, cols = grid_shape
    all_vals   = [v for v in means.values() if v is not None]
    min_val    = min(all_vals) if all_vals else true_distance
    max_val    = max(all_vals) if all_vals else true_distance

    for i, (label, mean_val) in enumerate(zip(zone_labels, cell_means)):
        r = i // cols
        c = i % cols
        r = (rows - 1) - r
        color = colour_for_value(mean_val, true_distance, min_val, max_val) \
                if mean_val is not None else (0.85, 0.85, 0.85, 1.0)
        rect = mpatches.FancyBboxPatch(
            (c, r), 1, 1, boxstyle="square,pad=0",
            facecolor=color, edgecolor="white", linewidth=0.5,
            transform=ax.transData)
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


# ── Loading and merging ────────────────────────────────────────────────────────

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
        raw                     = load_tof_data(folder, origin)
        trimmed, n_frames, shortest = trim_tof_data(raw)
        print(f"  {name}: {n_frames} frames, shortest dataset: {shortest}")
        experiments[name] = (trimmed, n_frames, shortest)
    return experiments


def combine_experiments(experiments: dict,
                        mode: str = "concatenate") -> tuple[dict, int]:
    """
    Combines multiple experiments into one dataset.

    Args:
        experiments: Output from load_multi_experiment.
        mode:        "concatenate" — joins all experiment frames end to end
                     per zone, preserving full spread and zone-to-zone variance.
                     "average" — averages across experiments per frame per zone,
                     showing mean behaviour and inter-experiment repeatability.

    Returns:
        combined_data, n_frames
    """
    all_dist_keys = set()
    for name, (trimmed, _, _) in experiments.items():
        all_dist_keys.update(trimmed.keys())

    PARAMS = ["distance_mm", "signal_per_spad", "ambient_per_spad", "is_valid_range"]
    ZONES  = range(64)

    combined          = {}
    n_frames_per_dist = {}

    for dist_key in all_dist_keys:
        min_len = None
        for name, (trimmed, _, _) in experiments.items():
            if dist_key in trimmed:
                for param in PARAMS:
                    for z in ZONES:
                        arr = trimmed[dist_key][param][z]
                        if arr is not None:
                            if min_len is None or len(arr) < min_len:
                                min_len = len(arr)

        if min_len is None:
            continue

        n_frames_per_dist[dist_key] = min_len
        combined[dist_key] = {}

        for param in PARAMS:
            combined[dist_key][param] = {}
            for z in ZONES:
                arrays = []
                for name, (trimmed, _, _) in experiments.items():
                    if dist_key in trimmed:
                        arr = trimmed[dist_key][param][z]
                        if arr is not None:
                            arrays.append(arr[:min_len])

                if arrays:
                    if mode == "concatenate":
                        combined[dist_key][param][z] = np.concatenate(arrays)
                    else:
                        combined[dist_key][param][z] = np.mean(
                            np.array(arrays), axis=0)
                else:
                    combined[dist_key][param][z] = None

    overall_min = min(n_frames_per_dist.values()) if n_frames_per_dist else 0
    return combined, overall_min


# ── Setup ──────────────────────────────────────────────────────────────────────

def setup_output_folder(output_path: str, label: str) -> str:
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    full_path = os.path.join(output_path, f"plots-{label}-{timestamp}")
    os.makedirs(full_path, exist_ok=False)
    return full_path


# ── Comparative plots ──────────────────────────────────────────────────────────

def comparative_tiled_heatmaps(experiments: dict, output_path: str):
    """
    For each distance, one figure with experiments side by side (4 heatmap
    panels each). One column of 4-panel heatmaps per experiment.
    """
    def get_ring(z):
        r = z // 8
        c = z % 8
        return min(r, 7 - r, c, 7 - c)

    # Collect all distances
    all_dist_keys = set()
    for name, (trimmed, _, _) in experiments.items():
        all_dist_keys.update(trimmed.keys())

    exp_names = list(experiments.keys())
    n_exps    = len(exp_names)

    for dist_key in sorted(all_dist_keys):
        true_distance = extract_true_distance(dist_key)

        # 4 rows (8x8, 4x4, 2x2, rings) x n_exps columns
        fig, axes = plt.subplots(4, n_exps,
                                 figsize=(6 * n_exps, 24))
        if n_exps == 1:
            axes = axes.reshape(4, 1)

        fig.suptitle(f"{dist_key} — Distance Heatmaps Comparison",
                     fontsize=14, y=1.002)

        for col, name in enumerate(exp_names):
            trimmed, _, _ = experiments[name]
            if dist_key not in trimmed:
                for row in range(4):
                    axes[row][col].text(0.5, 0.5, "No data",
                                        ha="center", va="center")
                    axes[row][col].axis("off")
                continue

            zone_data       = trimmed[dist_key]
            stats_dist      = analyse_tof_data({dist_key: zone_data})[dist_key]
            valid_distances = get_valid_distances(zone_data, stats_dist)
            zone_means      = compute_zone_means(valid_distances)

            axes[0][col].set_title(name, fontsize=10)

            # Row 0: 8x8
            ax = axes[0][col]
            draw_heatmap(ax, zone_means, true_distance,
                         (8, 8),
                         [f"Z{z}" for z in range(64)],
                         [zone_means.get(z) for z in range(64)])
            ax.set_ylabel("8×8", fontsize=8) if col == 0 else None

            # Row 1: 4x4
            ax = axes[1][col]
            gm, lb, cm = {}, [], []
            for sg in range(16):
                sr, sc = sg // 4, sg % 4
                vals   = [zone_means[z] for dr in range(2) for dc in range(2)
                          if (z := (sr*2+dr)*8+(sc*2+dc)) in zone_means]
                mv = float(np.mean(vals)) if vals else None
                gm[sg] = mv; lb.append(f"G{sg}"); cm.append(mv)
            draw_heatmap(ax, gm, true_distance, (4, 4), lb, cm)
            ax.set_ylabel("4×4", fontsize=8) if col == 0 else None

            # Row 2: 2x2
            ax = axes[2][col]
            gm, lb, cm = {}, [], []
            for sg in range(4):
                sr, sc = sg // 2, sg % 2
                vals   = [zone_means[z] for dr in range(4) for dc in range(4)
                          if (z := (sr*4+dr)*8+(sc*4+dc)) in zone_means]
                mv = float(np.mean(vals)) if vals else None
                gm[sg] = mv; lb.append(f"G{sg}"); cm.append(mv)
            draw_heatmap(ax, gm, true_distance, (2, 2), lb, cm)
            ax.set_ylabel("2×2", fontsize=8) if col == 0 else None

            # Row 3: Rings
            ax       = axes[3][col]
            rz       = {0: [], 1: [], 2: [], 3: []}
            for z, mv in zone_means.items():
                rz[get_ring(z)].append(mv)
            rm = {r: float(np.mean(v)) if v else None for r, v in rz.items()}
            all_v  = [v for v in rm.values() if v is not None]
            min_v  = min(all_v) if all_v else true_distance
            max_v  = max(all_v) if all_v else true_distance

            for z in range(64):
                r, c   = z // 8, z % 8
                ring   = get_ring(z)
                plot_r = 7 - r
                mv     = rm.get(ring)
                color  = colour_for_value(mv, true_distance, min_v, max_v) \
                         if mv is not None else (0.85, 0.85, 0.85, 1.0)
                rect   = mpatches.FancyBboxPatch(
                    (c, plot_r), 1, 1, boxstyle="square,pad=0",
                    facecolor=color, edgecolor="white", linewidth=0.5)
                ax.add_patch(rect)
                ax.text(c+0.5, plot_r+0.5,
                        f"{mv:.1f}" if mv is not None else "N/A",
                        ha="center", va="center", fontsize=6,
                        color="white", fontweight="bold")
            for ring in range(1, 4):
                o = ring
                ax.plot([o, 8-o, 8-o, o, o], [o, o, 8-o, 8-o, o],
                        color="white", linewidth=1.5, linestyle="--")
            ax.set_xlim(0, 8); ax.set_ylim(0, 8)
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_aspect("equal")
            ax.set_ylabel("Rings", fontsize=8) if col == 0 else None

        plt.tight_layout()
        save_path = os.path.join(output_path,
                                 f"{dist_key}_Comparative_Heatmaps.png")
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        plt.close(fig)


def comparative_tiled_validity(experiments: dict, output_path: str):
    """
    For each distance, 8x8 validity grids side by side per experiment.
    """
    all_dist_keys = set()
    for name, (trimmed, _, _) in experiments.items():
        all_dist_keys.update(trimmed.keys())

    exp_names = list(experiments.keys())
    n_exps    = len(exp_names)

    for dist_key in sorted(all_dist_keys):
        fig, axes = plt.subplots(8, 8 * n_exps,
                                 figsize=(11 * n_exps, 14))
        fig.suptitle(f"{dist_key} — Zone Validity Comparison",
                     fontsize=14, y=1.002)

        for col_exp, name in enumerate(exp_names):
            trimmed, _, _ = experiments[name]

            # Column header
            fig.text((col_exp + 0.5) / n_exps, 1.0, name,
                     ha="center", va="bottom", fontsize=11,
                     fontweight="bold")

            if dist_key not in trimmed:
                continue

            zone_data  = trimmed[dist_key]
            stats_dist = analyse_tof_data({dist_key: zone_data})[dist_key]

            for z in range(64):
                row, col = zone_to_grid(z)
                ax       = axes[row][col + col_exp * 8]
                arr      = zone_data["is_valid_range"][z]
                pct      = stats_dist["per_zone"]["percent_valid"][z]

                if arr is not None and pct < 100.0:
                    frames = np.arange(len(arr))
                    colors = [VALID_COLOR if v == 1
                              else INVALID_COLOR for v in arr]
                    ax.scatter(frames, arr, c=colors, s=8,
                               alpha=0.8, linewidths=0)
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
                    ax.text(0.5, 0.5, "Always\nValid",
                            ha="center", va="center", fontsize=7,
                            color="#2E7D32", transform=ax.transAxes)
                    ax.set_xticks([]); ax.set_yticks([])

                ax.set_title(f"Z{z}", fontsize=7, pad=2)

        plt.tight_layout()
        save_path = os.path.join(output_path,
                                 f"{dist_key}_Comparative_Validity.png")
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        plt.close(fig)


def comparative_total_boxplots(experiments: dict, output_path: str):
    """
    All experiments combined on one boxplot figure per distance.
    Experiments shown as grouped boxplots side by side per distance.
    """
    all_dist_keys = set()
    for name, (trimmed, _, _) in experiments.items():
        all_dist_keys.update(trimmed.keys())

    exp_names   = list(experiments.keys())
    n_exps      = len(exp_names)
    dist_keys   = sorted(all_dist_keys)
    x_positions = np.arange(len(dist_keys))
    width       = 0.8 / n_exps

    fig, ax = plt.subplots(figsize=(max(10, 2 * len(dist_keys)), 7))
    ax.set_title("Total Distance Error per Experiment", fontsize=12)

    for i, name in enumerate(exp_names):
        trimmed, _, _ = experiments[name]
        color         = PALETTE[i % len(PALETTE)]

        for j, dist_key in enumerate(dist_keys):
            if dist_key not in trimmed:
                continue

            true_distance   = extract_true_distance(dist_key)
            zone_data       = trimmed[dist_key]
            stats_dist      = analyse_tof_data({dist_key: zone_data})[dist_key]
            valid_distances = get_valid_distances(zone_data, stats_dist)

            if not valid_distances:
                continue

            all_valid = np.concatenate(list(valid_distances.values()))
            error     = all_valid - true_distance
            pos       = x_positions[j] + (i - n_exps / 2 + 0.5) * width

            ax.boxplot(error,
                       positions=[pos],
                       widths=width * 0.9,
                       patch_artist=True,
                       manage_ticks=False,
                       medianprops=dict(color="white", linewidth=1.5),
                       whiskerprops=dict(color=color, linewidth=1.0),
                       capprops=dict(color=color, linewidth=1.0),
                       flierprops=dict(marker=".", markersize=2,
                                       markerfacecolor=color, alpha=0.4),
                       boxprops=dict(facecolor=color, color=color,
                                     linewidth=1.0))

    ax.axhline(0, color="black", linewidth=1.0, linestyle=":", alpha=0.7)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(dist_keys, fontsize=9)
    ax.set_xlabel("Distance", fontsize=10)
    ax.set_ylabel("Error [mm]  (measured − true)", fontsize=10)
    ax.yaxis.set_minor_locator(plt.MultipleLocator(0.2))
    ax.grid(True, which="major", color="grey", linewidth=0.5,
            alpha=0.3, zorder=0)
    ax.grid(True, which="minor", color="grey", linewidth=0.3,
            alpha=0.2, zorder=0)

    legend_elements = [Patch(facecolor=PALETTE[i % len(PALETTE)], label=name)
                       for i, name in enumerate(exp_names)]
    ax.legend(handles=legend_elements, fontsize=8, loc="upper left")

    plt.tight_layout()
    save_path = os.path.join(output_path, "Comparative_Total_Boxplots.png")
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close(fig)


def comparative_error_validity(experiments: dict, output_path: str):
    """
    Error and validity plot with all experiments overlaid.
    Each experiment gets its own colour for the error boxplots.
    Single averaged validity and % error line across all experiments.
    """
    all_dist_keys = set()
    for name, (trimmed, _, _) in experiments.items():
        all_dist_keys.update(trimmed.keys())

    exp_names = list(experiments.keys())
    n_exps    = len(exp_names)
    dist_keys = sorted(all_dist_keys, key=lambda k: extract_true_distance(k))
    x_pos     = np.arange(len(dist_keys))
    width     = 0.8 / n_exps

    fig, ax1  = plt.subplots(figsize=(max(10, 2 * len(dist_keys)), 7))
    ax2       = ax1.twinx()
    ax1.set_zorder(ax2.get_zorder() + 1)
    ax1.patch.set_visible(False)

    ax1.set_title("Distance Error and Zone Validity — Comparative", fontsize=12)

    # ── Boxplots per experiment ────────────────────────────────────────────────
    for i, name in enumerate(exp_names):
        trimmed, _, _ = experiments[name]
        color         = PALETTE[i % len(PALETTE)]

        for j, dist_key in enumerate(dist_keys):
            if dist_key not in trimmed:
                continue

            true_distance = extract_true_distance(dist_key)
            zone_data     = trimmed[dist_key]

            valid_distances = get_valid_distances(zone_data,
                                                  analyse_tof_data({dist_key: zone_data})[dist_key])
            if not valid_distances:
                continue
            all_valid = np.concatenate(list(valid_distances.values()))
            error = all_valid - true_distance
            pos            = x_pos[j] + (i - n_exps / 2 + 0.5) * width

            ax1.boxplot(error,
                        positions=[pos],
                        widths=width * 0.9,
                        patch_artist=True,
                        manage_ticks=False,
                        medianprops=dict(color="white", linewidth=1.5),
                        whiskerprops=dict(color=color, linewidth=1.0),
                        capprops=dict(color=color, linewidth=1.0),
                        flierprops=dict(marker=".", markersize=2,
                                        markerfacecolor=color, alpha=0.4),
                        boxprops=dict(facecolor=color, color=color,
                                      linewidth=1.0))

    # ── Averaged validity and % error lines ───────────────────────────────────
    avg_validity = []
    avg_pct_err  = []

    for j, dist_key in enumerate(dist_keys):
        v_vals = []
        p_vals = []

        for name, (trimmed, _, _) in experiments.items():
            if dist_key not in trimmed:
                continue

            true_distance = extract_true_distance(dist_key)
            zone_data     = trimmed[dist_key]
            stats_dist    = analyse_tof_data({dist_key: zone_data})[dist_key]

            n_always_valid = sum(
                1 for z in range(64)
                if stats_dist["per_zone"]["percent_valid"][z] is not None
                and stats_dist["per_zone"]["percent_valid"][z] >= 100.0)
            v_vals.append(100.0 * n_always_valid / 64)

            valid_distances = get_valid_distances(zone_data,
                                                  analyse_tof_data({dist_key: zone_data})[dist_key])
            if not valid_distances:
                continue
            all_valid = np.concatenate(list(valid_distances.values()))
            error = all_valid - true_distance
            mean_err       = float(np.mean(error))
            p_vals.append(100.0 * abs(mean_err) / true_distance
            if true_distance != 0 else 0.0)

        avg_validity.append(float(np.mean(v_vals)) if v_vals else None)
        avg_pct_err.append(float(np.mean(p_vals))  if p_vals else None)

    valid_x   = [x_pos[j] for j, v in enumerate(avg_validity) if v is not None]
    valid_y   = [v for v in avg_validity if v is not None]
    pct_err_x = [x_pos[j] for j, v in enumerate(avg_pct_err) if v is not None]
    pct_err_y = [v for v in avg_pct_err if v is not None]

    ax2.fill_between(valid_x, 0, valid_y, color="#A3DBDD", alpha=0.5)
    ax2.plot(valid_x, valid_y, color="#18A5AA", linewidth=2.0,
             label="Avg % zones always valid")
    ax2.plot(pct_err_x, pct_err_y, color="#E97122", linewidth=2.0,
             linestyle="--", label="Avg % error")

    # ── Axis formatting ───────────────────────────────────────────────────────
    ax1.axhline(0, color="black", linewidth=1.0, linestyle=":", alpha=0.7)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(dist_keys, fontsize=9)
    ax1.set_xlabel("Distance", fontsize=10)
    ax1.set_ylabel("Error [mm]  (measured − true)", fontsize=10)
    ax1.yaxis.set_minor_locator(plt.MultipleLocator(0.2))
    ax1.grid(True, which="major", color="grey", linewidth=0.5,
             alpha=0.3, zorder=0)
    ax1.grid(True, which="minor", color="grey", linewidth=0.3,
             alpha=0.2, zorder=0)

    ax2.set_ylabel("% Zones Always Valid  /  % Error", fontsize=10)
    ax2.set_ylim(0, 110)

    # ── Legend ────────────────────────────────────────────────────────────────
    legend_elements = [
        Patch(facecolor=PALETTE[i % len(PALETTE)], label=name)
        for i, name in enumerate(exp_names)
    ] + [
        Line2D([0], [0], color="#18A5AA", linewidth=2.0,
               label="Avg % zones always valid"),
        Line2D([0], [0], color="#E97122", linewidth=2.0,
               linestyle="--", label="Avg % error"),
    ]
    ax1.legend(handles=legend_elements, fontsize=7,
               loc="upper left", framealpha=0.9)

    plt.tight_layout()
    save_path = os.path.join(output_path, "Comparative_Error_Validity.png")
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close(fig)

def comparative_drift(experiments: dict, output_path: str):
    """
    Drift analysis with linear fits only (no raw data) per experiment,
    all overlaid on the same plot. One line style per distance,
    one colour per experiment.
    """
    all_dist_keys = set()
    for name, (trimmed, _, _) in experiments.items():
        all_dist_keys.update(trimmed.keys())

    exp_names = list(experiments.keys())
    dist_keys = sorted(all_dist_keys, key=lambda k: extract_true_distance(k))
    dist_styles = ["-", "--", "-.", ":"]

    fig = plt.figure(figsize=(14, 9))
    gs  = plt.GridSpec(2, 1, height_ratios=[10, 2], hspace=0.05)
    ax      = fig.add_subplot(gs[0])
    ax_text = fig.add_subplot(gs[1])
    ax_text.axis("off")

    drift_rates = {name: {} for name in exp_names}

    for i, name in enumerate(exp_names):
        trimmed, _, _ = experiments[name]
        color         = PALETTE[i % len(PALETTE)]

        for j, dist_key in enumerate(dist_keys):
            if dist_key not in trimmed:
                continue

            true_distance = extract_true_distance(dist_key)
            zone_data     = trimmed[dist_key]
            zone_arrays   = [zone_data["distance_mm"][z] for z in range(64)
                             if zone_data["distance_mm"][z] is not None]
            if not zone_arrays:
                continue

            min_len         = min(len(a) for a in zone_arrays)
            stacked         = np.array([a[:min_len] for a in zone_arrays])
            mean_per_frame  = np.mean(stacked, axis=0)
            error_per_frame = mean_per_frame - true_distance
            frames          = np.arange(len(error_per_frame))

            slope, intercept, _, _, _ = sp_stats.linregress(
                frames, error_per_frame)
            drift_rates[name][dist_key] = float(slope)

            linestyle = dist_styles[j % len(dist_styles)]
            label     = f"{name} — {dist_key}"
            ax.plot(frames, intercept + slope * frames,
                    color=color, linewidth=2.0,
                    linestyle=linestyle, label=label)

    ax.axhline(0, color="black", linewidth=1.0, linestyle="-", alpha=0.4)
    ax.set_ylabel("Error [mm]  (measured − true)", fontsize=10)
    ax.set_xticks([])
    ax.set_xlabel("Time →")
    ax.set_title("Drift Analysis — Linear Fits per Experiment and Distance\n"
                 "(colour = experiment, line style = distance)",
                 pad=40)
    ax.legend(fontsize=7, loc="lower center",
              bbox_to_anchor=(0.5, 1.02),
              ncol=min(7, len(exp_names) * len(dist_keys)),
              borderaxespad=0)

    # Annotation — steepest and average per experiment
    lines = []
    for name in exp_names:
        rates = drift_rates[name]
        if not rates:
            continue
        steepest_key   = max(rates, key=lambda k: abs(rates[k]))
        steepest_slope = rates[steepest_key]
        avg_slope      = float(np.mean(list(rates.values())))
        lines.append(f"{name} — Steepest: {steepest_key} "
                     f"({steepest_slope:+.4f} mm/frame)  "
                     f"Avg: {avg_slope:+.4f} mm/frame")

    line_spacing = 0.045
    start_y      = -0.08
    for j, line in enumerate(lines):
        ax.annotate(line,
                    xy=(0.5, start_y - j * line_spacing),
                    xycoords="axes fraction",
                    ha="center", va="top", fontsize=7.5,
                    annotation_clip=False)

    save_path = os.path.join(output_path, "Comparative_Drift_Analysis.png")
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close(fig)


def comparative_error_heatmaps(experiments: dict, output_path: str):
    """
    Error heatmaps side by side per experiment, plus a combined average.
    """
    def get_ring(z):
        r = z // 8
        c = z % 8
        return min(r, 7 - r, c, 7 - c)

    def draw_error_heatmap_inline(ax, abs_errors, bias_errors,
                                   grid_shape, labels, cell_abs, cell_bias):
        rows, cols  = grid_shape
        valid_bias  = [v for v in cell_bias if v is not None]
        min_bias    = min(valid_bias) if valid_bias else 0.0
        max_bias    = max(valid_bias) if valid_bias else 0.0

        for i, (label, abs_val, bias_val) in enumerate(
                zip(labels, cell_abs, cell_bias)):
            r = i // cols
            c = i % cols
            r = (rows - 1) - r
            color = bias_colour(bias_val, min_bias, max_bias) \
                    if bias_val is not None else (0.85, 0.85, 0.85, 1.0)
            rect = mpatches.FancyBboxPatch(
                (c, r), 1, 1, boxstyle="square,pad=0",
                facecolor=color, edgecolor="white", linewidth=0.5,
                transform=ax.transData)
            ax.add_patch(rect)
            abs_text  = f"{abs_val:.1f}"  if abs_val  is not None else "N/A"
            bias_text = f"b:{bias_val:+.1f}" if bias_val is not None else ""
            ax.text(c+0.5, r+0.5, f"{label}\n{abs_text}\n{bias_text}",
                    ha="center", va="center", fontsize=5.5,
                    color="white", fontweight="bold")
        ax.set_xlim(0, cols); ax.set_ylim(0, rows)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_aspect("equal")

    exp_names = list(experiments.keys())
    # Include a combined average column
    all_columns = exp_names + ["Combined"]
    n_cols      = len(all_columns)

    fig, axes = plt.subplots(4, n_cols, figsize=(6 * n_cols, 24))
    if n_cols == 1:
        axes = axes.reshape(4, 1)
    fig.suptitle("Experiment Error Heatmaps — Comparative",
                 fontsize=14, y=1.002)

    row_labels = ["8×8", "4×4", "2×2", "Rings"]
    for row in range(4):
        if axes[row][0].get_ylabel() == "":
            axes[row][0].set_ylabel(row_labels[row], fontsize=9)

    # Compute per-experiment and combined errors
    all_abs  = []
    all_bias = []

    for col, name in enumerate(all_columns):
        axes[0][col].set_title(name, fontsize=10)

        if name == "Combined":
            # Average abs and bias across all experiments
            if all_abs:
                abs_errors  = {z: float(np.mean([a[z] for a in all_abs
                                                  if a.get(z) is not None]))
                               for z in range(64)}
                bias_errors = {z: float(np.mean([b[z] for b in all_bias
                                                  if b.get(z) is not None]))
                               for z in range(64)}
            else:
                continue
        else:
            trimmed, _, _ = experiments[name]
            stats_data    = {dk: analyse_tof_data({dk: trimmed[dk]})[dk]
                             for dk in trimmed}

            zone_abs  = {z: [] for z in range(64)}
            zone_bias = {z: [] for z in range(64)}

            for dist_key, zone_data in trimmed.items():
                true_distance   = extract_true_distance(dist_key)
                valid_distances = get_valid_distances(
                    zone_data, stats_data[dist_key])
                zone_means      = compute_zone_means(valid_distances)

                for z in range(64):
                    if z in zone_means:
                        err = zone_means[z] - true_distance
                        zone_abs[z].append(abs(err))
                        zone_bias[z].append(err)

            abs_errors  = {z: float(np.mean(v)) if v else None
                           for z, v in zone_abs.items()}
            bias_errors = {z: float(np.mean(v)) if v else None
                           for z, v in zone_bias.items()}
            all_abs.append(abs_errors)
            all_bias.append(bias_errors)

        # Row 0: 8x8
        draw_error_heatmap_inline(
            axes[0][col], abs_errors, bias_errors,
            (8, 8),
            [f"Z{z}" for z in range(64)],
            [abs_errors.get(z) for z in range(64)],
            [bias_errors.get(z) for z in range(64)])

        # Row 1: 4x4
        ga, gb, lb = {}, {}, []
        ca, cb     = [], []
        for sg in range(16):
            sr, sc = sg // 4, sg % 4
            av = [abs_errors[z]  for dr in range(2) for dc in range(2)
                  if (z := (sr*2+dr)*8+(sc*2+dc)) in abs_errors
                  and abs_errors[z] is not None]
            bv = [bias_errors[z] for dr in range(2) for dc in range(2)
                  if (z := (sr*2+dr)*8+(sc*2+dc)) in bias_errors
                  and bias_errors[z] is not None]
            ga[sg] = float(np.mean(av)) if av else None
            gb[sg] = float(np.mean(bv)) if bv else None
            lb.append(f"G{sg}"); ca.append(ga[sg]); cb.append(gb[sg])
        draw_error_heatmap_inline(axes[1][col], ga, gb, (4, 4), lb, ca, cb)

        # Row 2: 2x2
        ga, gb, lb = {}, {}, []
        ca, cb     = [], []
        for sg in range(4):
            sr, sc = sg // 2, sg % 2
            av = [abs_errors[z]  for dr in range(4) for dc in range(4)
                  if (z := (sr*4+dr)*8+(sc*4+dc)) in abs_errors
                  and abs_errors[z] is not None]
            bv = [bias_errors[z] for dr in range(4) for dc in range(4)
                  if (z := (sr*4+dr)*8+(sc*4+dc)) in bias_errors
                  and bias_errors[z] is not None]
            ga[sg] = float(np.mean(av)) if av else None
            gb[sg] = float(np.mean(bv)) if bv else None
            lb.append(f"G{sg}"); ca.append(ga[sg]); cb.append(gb[sg])
        draw_error_heatmap_inline(axes[2][col], ga, gb, (2, 2), lb, ca, cb)

        # Row 3: Rings
        ax     = axes[3][col]
        ra, rb = {0: [], 1: [], 2: [], 3: []}, {0: [], 1: [], 2: [], 3: []}
        for z in range(64):
            ring = get_ring(z)
            if abs_errors.get(z)  is not None: ra[ring].append(abs_errors[z])
            if bias_errors.get(z) is not None: rb[ring].append(bias_errors[z])
        ram = {r: float(np.mean(v)) if v else None for r, v in ra.items()}
        rbm = {r: float(np.mean(v)) if v else None for r, v in rb.items()}
        vb  = [v for v in rbm.values() if v is not None]
        mn  = min(vb) if vb else 0.0
        mx  = max(vb) if vb else 0.0

        for z in range(64):
            r, c   = z // 8, z % 8
            ring   = get_ring(z)
            plot_r = 7 - r
            bv     = rbm.get(ring)
            av     = ram.get(ring)
            color  = bias_colour(bv, mn, mx) \
                     if bv is not None else (0.85, 0.85, 0.85, 1.0)
            rect   = mpatches.FancyBboxPatch(
                (c, plot_r), 1, 1, boxstyle="square,pad=0",
                facecolor=color, edgecolor="white", linewidth=0.5)
            ax.add_patch(rect)
            ax.text(c+0.5, plot_r+0.5,
                    f"{av:.1f}\nb:{bv:+.1f}" if av is not None else "N/A",
                    ha="center", va="center", fontsize=6,
                    color="white", fontweight="bold")
        for ring in range(1, 4):
            o = ring
            ax.plot([o, 8-o, 8-o, o, o], [o, o, 8-o, 8-o, o],
                    color="white", linewidth=1.5, linestyle="--")
        ax.set_xlim(0, 8); ax.set_ylim(0, 8)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_aspect("equal")

    plt.tight_layout()
    save_path = os.path.join(output_path, "Comparative_Error_Heatmaps.png")
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close(fig)


def comparative_overall_validity(experiments: dict, output_path: str):
    """
    Overall validity heatmaps side by side per experiment.
    """
    exp_names = list(experiments.keys())
    n_exps    = len(exp_names)

    fig, axes = plt.subplots(1, n_exps, figsize=(8 * n_exps, 8))
    if n_exps == 1:
        axes = [axes]
    fig.suptitle("Overall Zone Validity — Comparative", fontsize=14)

    for col, name in enumerate(exp_names):
        ax            = axes[col]
        trimmed, _, _ = experiments[name]
        stats_data    = {dk: analyse_tof_data({dk: trimmed[dk]})[dk]
                         for dk in trimmed}

        zone_validity = {z: [] for z in range(64)}
        for dist_key in trimmed:
            for z in range(64):
                pct = stats_data[dist_key]["per_zone"]["percent_valid"][z]
                if pct is not None:
                    zone_validity[z].append(pct)

        zone_mean_validity = {z: float(np.mean(v)) if v else None
                              for z, v in zone_validity.items()}

        invalid_vals = [v for v in zone_mean_validity.values()
                        if v is not None and v < 100.0]
        min_valid    = min(invalid_vals) if invalid_vals else 0.0

        ax.set_title(name, fontsize=10)

        for z in range(64):
            r      = z // 8
            c      = z % 8
            plot_r = 7 - r
            pct    = zone_mean_validity.get(z)

            if pct is None or pct >= 100.0:
                color     = (*mcolors.to_rgb("#52A83C"), 1.0)
                text      = "Always\nValid"
                txt_color = "white"
            else:
                t       = (pct - min_valid) / (100.0 - min_valid + 1e-9)
                color_a = mcolors.to_rgb("#E97122")
                color_b = mcolors.to_rgb("#F6C6A7")
                rgb     = tuple(color_a[i] * (1-t) + color_b[i] * t
                                for i in range(3))
                color     = (*rgb, 1.0)
                text      = f"{pct:.1f}%"
                txt_color = "black"

            rect = mpatches.FancyBboxPatch(
                (c, plot_r), 1, 1, boxstyle="square,pad=0",
                facecolor=color, edgecolor="white", linewidth=0.5)
            ax.add_patch(rect)
            ax.text(c+0.5, plot_r+0.5, text,
                    ha="center", va="center", fontsize=7,
                    color=txt_color, fontweight="bold")

        ax.set_xlim(0, 8); ax.set_ylim(0, 8)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_aspect("equal")

    plt.tight_layout()
    save_path = os.path.join(output_path,
                             "Comparative_Overall_Validity.png")
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close(fig)


# ── Master functions ───────────────────────────────────────────────────────────

def generate_combined_plots(experiments: dict, output_path: str) -> tuple[str, str]:
    """
    Generates two sets of combined plots:
    - Concatenated: full spread across all experiments and zones,
      equivalent to one long experiment. Shows total error distribution.
    - Averaged: mean behaviour across experiments per zone per frame,
      shows inter-experiment repeatability with compressed spread.

    Args:
        experiments: Output from load_multi_experiment.
        output_path: Path where the plots folders should be created.

    Returns:
        Tuple of (concatenated_plots_folder, averaged_plots_folder)
    """
    from Visualize import generate_all_zone_plots

    # ── Concatenated ──────────────────────────────────────────────────────────
    print("Combining experiments (concatenated)...")
    combined_concat, _ = combine_experiments(experiments, mode="concatenate")
    stats_concat       = analyse_tof_data(combined_concat)

    folder_concat = setup_output_folder(output_path, "combined-concatenated")
    print(f"Saving concatenated plots to: {folder_concat}")
    generate_all_zone_plots(combined_concat, stats_concat, folder_concat)
    print("Concatenated done!")

    # ── Averaged ──────────────────────────────────────────────────────────────
    print("Combining experiments (averaged)...")
    combined_avg, _ = combine_experiments(experiments, mode="average")
    stats_avg       = analyse_tof_data(combined_avg)

    folder_avg = setup_output_folder(output_path, "combined-averaged")
    print(f"Saving averaged plots to: {folder_avg}")
    generate_all_zone_plots(combined_avg, stats_avg, folder_avg)
    print("Averaged done!")

    return folder_concat, folder_avg