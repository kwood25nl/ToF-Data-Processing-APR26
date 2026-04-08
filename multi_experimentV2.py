def generate_plots(experiment_folders: dict, origin: int, output_path: str, do_overall_validity_concat: bool = True) -> str:
    import os
    # Create output folder if it doesn't exist
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Call plot_overall_validity_heatmap_multi_concat conditionally
    if do_overall_validity_concat:
        plot_overall_validity_heatmap_multi_concat(experiment_folders, origin, output_path)

    return output_path


# Existing function for backward compatibility

def generate_multi_experiment_plots_v2(...args):
    # Call the new master function instead of the previous implementation
    return generate_combined_plots(...args)


def generate_combined_plots(...args):
    # Logic to generate combined plots
    pass
