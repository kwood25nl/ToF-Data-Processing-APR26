# This is a sample Python script.
from ImportData import load_tof_data
from Calculations import trim_tof_data
from Calculations import analyse_tof_data
from Visualize import generate_all_zone_plots


# from multi_experiment import load_multi_experiment
# from multi_experiment import generate_comparative_plots, generate_combined_plots



# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    flatField01 = load_tof_data(r"Z:\Kaitie\ToF and Mono Camera Experiments APR26\flatField Data 2APR2026\flatField05-2APR26", 330)

    trimmed, n_frames, shortestKey = trim_tof_data(flatField01)
    stats = analyse_tof_data(trimmed)
    print(f"Trimmed to {n_frames} frames. Shortest dataset: {shortestKey}")



    plots_folder = generate_all_zone_plots(
        data=trimmed,
        stats_data=stats,
        output_path=r"Z:\Kaitie\ToF and Mono Camera Experiments APR26\flatField Data 2APR2026\flatField05-2APR26"
    )

    # experiments = load_multi_experiment(
    #     experiment_folders={
    #         "Flat Field 01": r"Z:\Kaitie\ToF and Mono Camera Experiments APR26\flatField Data 2APR2026\flatField01-2APR26",
    #         "Flat Field 02": r"Z:\Kaitie\ToF and Mono Camera Experiments APR26\flatField Data 2APR2026\flatField02-2APR26",
    #         "Flat Field 03": r"Z:\Kaitie\ToF and Mono Camera Experiments APR26\flatField Data 2APR2026\flatField03-2APR26",
    #         "Flat Field 04": r"Z:\Kaitie\ToF and Mono Camera Experiments APR26\flatField Data 2APR2026\flatField04-2APR26",
    #         "Flat Field 05": r"Z:\Kaitie\ToF and Mono Camera Experiments APR26\flatField Data 2APR2026\flatField05-2APR26",
    #     },
    #     origin=330
    # )
    #
    # # Comparative — all experiments side by side
    # generate_comparative_plots(experiments, r"Z:\Kaitie\ToF and Mono Camera Experiments APR26\flatField Data 2APR2026\testOutput")
    #
    # # Combined — averaged as one experiment
    # generate_combined_plots(experiments, r"Z:\Kaitie\ToF and Mono Camera Experiments APR26\flatField Data 2APR2026\testOutput")

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
