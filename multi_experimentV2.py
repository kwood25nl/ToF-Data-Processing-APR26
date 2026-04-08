import os
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors

from Calculations import trim_tof_data
from ImportData import load_tof_data


# ── Colours (match Visualize.py) ─────────────────────────────────────────────
ALWAYS_VALID_COLOR = "#52A83C"
MOST_VALID_COLOR   = "#F6C6A7"
LEAST_VALID_COLOR  = "#E97122"
NO_DATA_COLOR      = "#D9D9D9"

# Box colors (match Visualize.py)
BOX_WITHIN = "#52A83C"
BOX_OVER   = "#2E61A8"
BOX_LOW    = "#E8708E"

def extract_true_distance(key: str) -> float:
    """Extracts the numeric true distance from a distance key (matches Visualize.py)."""
    digits = re.findall(r"-?\d+", key)
    return float("".join(digits))

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

def compute_zone_validity_percent_concat(experiments: dict) -> dict:
    """
    Computes percent-valid per zone by concatenating is_valid_range across:
      - all experiments
      - all distances
      - all frames

    Missing distances/zones are simply ignored (Option A).

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

                # Expect 0/1 ints, but be robust: count only 0/1 values
                a = np.asarray(arr)
                mask = (a == 0) | (a == 1)
                if not np.any(mask):
                    continue

                a01 = a[mask]
                total_count[z] += int(a01.size)
                ones_count[z] += int(np.sum(a01 == 1))

    zone_pct_valid = {}
    for z in range(64):
        if total_count[z] == 0:
            zone_pct_valid[z] = None
        else:
            zone_pct_valid[z] = 100.0 * ones_count[z] / total_count[z]

    return zone_pct_valid

def _interp_rgb(rgb_a, rgb_b, t: float):
    return tuple(rgb_a[i] * (1 - t) + rgb_b[i] * t for i in range(3))

def plot_multiexp_validity_heatmap(experiments: dict, output_path: str):
    """
    Single 8x8 heatmap showing validity across ALL experiments and distances
    (concatenated is_valid_range), per zone.

    - 100% valid: ALWAYS_VALID_COLOR (#52A83C) with label "Always\nValid"
    - otherwise: interpolate MOST_VALID_COLOR -> LEAST_VALID_COLOR and label "{pct:.1f}%"
    - no data: NO_DATA_COLOR and label "No\nData"

    Orientation matches Visualize.py: plot_r = r
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
            # Interpolate between MOST_VALID_COLOR (high pct) and LEAST_VALID_COLOR (low pct)
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
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")

    plt.tight_layout()
    save_path = os.path.join(output_path, "MultiExperiment_Validity_Heatmap.png")
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close(fig)

def _safe_concat(arr_list: list[np.ndarray]) -> np.ndarray | None:
    arr_list = [a for a in arr_list if a is not None and len(a) > 0]
    return np.concatenate(arr_list) if arr_list else None

def plot_multiexp_error_and_validity(experiments: dict, output_path: str):
    """
    Multi-experiment version of Visualize.plot_error_and_validity.

    Per distance (dist_key):
      - Boxplot data: concatenated error samples across ALL experiments,
        but ONLY from zones that are always valid for that distance within that
        experiment (is_valid_range == 1 for all frames in that zone array).
        Error sample = distance_mm - true_distance.

      - Validity line: for that distance, concatenate is_valid_range samples
        across ALL experiments and zones, and compute percent valid.

    Output:
      - Saves MultiExperiment_Error_Validity.png
    """
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D

    # Collect all distance keys across experiments
    all_dist_keys = set()
    for _name, (trimmed, _n_frames, _shortest) in experiments.items():
        all_dist_keys.update(trimmed.keys())

    sorted_dist_keys = sorted(all_dist_keys, key=extract_true_distance)
    if not sorted_dist_keys:
        return

    dist_labels: list[str] = []
    error_arrays: list[np.ndarray] = []
    mean_errors: list[float] = []
    validity_pcts: list[float] = []
    pct_errors: list[float] = []

    # For robust y scaling (IQR-based)
    q1_list: list[float] = []
    q3_list: list[float] = []

    for dist_key in sorted_dist_keys:
        true_distance = extract_true_distance(dist_key)
        dist_labels.append(dist_key)

        # --- Boxplot data: errors from always-valid zones only ---
        per_dist_error_chunks: list[np.ndarray] = []

        # --- Validity line: concat all is_valid_range samples ---
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

            # error pool (always-valid zones)
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

                # Always-valid zone means every 0/1 entry is 1
                if np.any(v01 == 0):
                    continue

                a = np.asarray(arr)
                # Keep alignment with validity mask if validity had non-0/1 noise
                # (usually unnecessary, but consistent with validity robustness)
                a = a[:len(v)]
                per_dist_error_chunks.append(a - true_distance)

        error = _safe_concat(per_dist_error_chunks)
        if error is None:
            error = np.array([0.0])

        error_arrays.append(error)

        mean_err = float(np.mean(error))
        mean_errors.append(mean_err)

        pct_errors.append(100.0 * abs(mean_err) / true_distance
                          if true_distance != 0 else 0.0)

        validity_pcts.append(100.0 * valid_ones / valid_total if valid_total > 0 else 0.0)

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
    ax2.fill_between(x_positions, 0, validity_pcts, color="#A3DBDD", alpha=0.5)
    ax2.plot(x_positions, validity_pcts, color="#18A5AA", linewidth=2.0)

    # --- % error line on ax2 ---
    ax2.plot(x_positions, pct_errors, color="#E97122", linewidth=2.0)

    # --- Zero error reference line ---
    ax1.axhline(0, color="black", linewidth=1.0, linestyle=":", alpha=0.7)

    # --- Boxplots (colored by mean error relative to 0) ---
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

    # --- Robust y-limits based on IQR across distances ---
    if q1_list and q3_list:
        global_q1 = float(np.min(q1_list))
        global_q3 = float(np.max(q3_list))
        global_iqr = max(global_q3 - global_q1, 1e-6)

        pad = 1.5 * global_iqr
        y_min = global_q1 - pad
        y_max = global_q3 + pad

        y_min = min(y_min, -0.5)
        y_max = max(y_max, 0.5)
        ax1.set_ylim(y_min, y_max)

    ax1.set_ylabel("Error [mm]", fontsize=10)
    ax1.set_xticks(x_positions)
    ax1.set_xticklabels(dist_labels, fontsize=9)
    ax1.set_xlabel("Distance [mm]", fontsize=10)
    ax1.set_title("Multi-Experiment Error and Zone Validity", fontsize=12)

    ax2.set_ylabel("% Valid (concatenated)  /  % Error", fontsize=10)
    ax2.tick_params(axis="y", colors="black")

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
    save_path = os.path.join(output_path, "MultiExperiment_Error_Validity.png")
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close(fig)


def generate_multi_experiment_plots_v2(experiment_folders: dict, origin: int, output_path: str) -> str:
    """Convenience entrypoint: load experiments, then produce the v2 plot set."""
    experiments = load_multi_experiment(experiment_folders, origin)
    plots_folder = setup_output_folder(output_path, "multiV2")

    print("Generating concatenated overall validity heatmap...")
    plot_multiexp_validity_heatmap(experiments, plots_folder)

    print("Generating error and validity plot...")
    plot_multiexp_error_and_validity(experiments, plots_folder)

    print("Done!")
    return plots_folder