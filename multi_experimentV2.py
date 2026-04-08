import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors


def setup_output_folder(output_path: str, label: str):
    timestamped_folder = os.path.join(output_path, f'plots-multiV2-{label}')
    os.makedirs(timestamped_folder, exist_ok=True)
    return timestamped_folder


def plot_overall_validity_heatmap_multi_concat(experiments: dict, output_path: str):
    # Assuming experiments is a dict of {experiment_name: trimmed_data}
    zones_count = 64
    zones_validity = {z: [] for z in range(zones_count)}

    # Concatenate validity data across all experiments
    for experiment_name, trimmed_data in experiments.items():
        for dist_key in trimmed_data:
            for z in range(zones_count):
                zones_validity[z].extend(trimmed_data[dist_key]['is_valid_range'][z])

    # Calculate pct_valid for each zone
    pct_valids = []
    for z in range(zones_count):
        if len(zones_validity[z]) == 0:
            pct_valids.append(None)  # No validity data
        else:
            count_1s = sum(zones_validity[z])
            total_count = len(zones_validity[z])
            pct_valid = 100.0 * count_1s / total_count
            pct_valids.append(pct_valid)

    # Prepare heatmap data
    heatmap_data = np.array(pct_valids).reshape(8, 8)
    heatmap_data = np.flipud(heatmap_data)  # Flip to get correct orientation for plotting

    plt.figure(figsize=(8, 8))
    cmap = mcolors.ListedColormap(['#FFFFFF', '#F6C6A7', '#E97122', '#52A83C'])
    bounds = [-1, 99.9, 100, 100.1]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    im = plt.imshow(heatmap_data, cmap=cmap, norm=norm)
    plt.colorbar(im, ticks=[0, 25, 50, 75, 100])

    for (i, j), val in np.ndenumerate(heatmap_data):
        if val is None:
            plt.text(j, i, 'No\nData', ha='center', va='center', color='black')
        elif val >= 100:
            plt.text(j, i, 'Always\nValid', ha='center', va='center', color='black')
        else:
            plt.text(j, i, f'{val:.1f}%', ha='center', va='center')

    plt.title('Overall Validity Heatmap')
    plt.savefig(os.path.join(output_path, 'MultiExperiment_Validity_Heatmap_Concatenated.png'), dpi=300)
    plt.close()


def generate_multi_experiment_plots_v2(experiment_folders: dict, origin: int, output_path: str):
    experiments = load_multi_experiment(experiment_folders)  # Assuming load_multi_experiment is correctly imported
    folder = setup_output_folder(output_path, origin)
    plot_overall_validity_heatmap_multi_concat(experiments, folder)
