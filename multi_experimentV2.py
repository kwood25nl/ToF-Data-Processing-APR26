import os
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

def plot_overall_validity_heatmap_multi_concat(experiments: dict, output_path: str):
    """
    Single 8x8 heatmap showing validity across ALL experiments and distances
    (concatenated is_valid_range), per zone.

    - 100% valid: ALWAYS_VALID_COLOR (#52A83C) with label "Always\nValid"
    - otherwise: interpolate MOST_VALID_COLOR -> LEAST_VALID_COLOR and label "{pct:.1f}%"
    - no data: NO_DATA_COLOR and label "No\nData"

    Orientation matches Visualize.py: plot_r = 7 - r
    """
    zone_pct_valid = compute_zone_validity_percent_concat(experiments)

    invalid_vals = [v for v in zone_pct_valid.values() if v is not None and v < 100.0]
    min_valid = min(invalid_vals) if invalid_vals else 0.0
    max_valid = 100.0

    fig, ax = plt.subplots(figsize=(8, 8))
    fig.suptitle("Overall Zone Validity (All Experiments Concatenated)", fontsize=12)

    for z in range(64):
        r = z // 8
        c = z % 8
        plot_r = 7 - r
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
    save_path = os.path.join(output_path, "MultiExperiment_Validity_Heatmap_Concatenated.png")
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close(fig)

def generate_multi_experiment_plots_v2(experiment_folders: dict, origin: int, output_path: str) -> str:
    """Convenience entrypoint: load experiments, then produce the v2 plot set."""
    experiments = load_multi_experiment(experiment_folders, origin)
    plots_folder = setup_output_folder(output_path, "multiV2")

    print("Generating concatenated overall validity heatmap...")
    plot_overall_validity_heatmap_multi_concat(experiments, plots_folder)

    print("Done!")
    return plots_folder
