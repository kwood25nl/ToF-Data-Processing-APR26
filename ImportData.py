import os
import re
import pandas as pd

def load_tof_data(folder_path: str, origin: int) -> dict:
    """
    Load ToF sensor data from a folder of subfolders.

    Args:
        folder_path: Path to the parent folder containing distance subfolders.
        origin:      Reference distance. Keys will be (origin - folder_distance).

    Returns:
        A flat dict keyed by f"d{origin - folder_distance}", each holding arrays
        for distance_mm, signal_per_spad, ambient_per_spad, is_valid_range per zone (z0-z63).
    """

    ZONES = range(64)
    PARAMETERS = ["distance_mm", "signal_per_spad", "ambient_per_spad", "is_valid_range"]

    def col_name(param, zone):
        if param == "is_valid_range":
            return f".is_valid_range_z{zone}"
        return f"{param}_z{zone}"

    result = {}

    for entry in os.scandir(folder_path):
        if not entry.is_dir():
            continue

        digits = re.findall(r"\d+", entry.name)
        if not digits:
            continue
        folder_distance = int("".join(digits))
        key = f"d{origin - folder_distance}"

        data_file = None
        for f in os.scandir(entry.path):
            if f.name.startswith("data") and f.name.endswith(".csv"):
                data_file = f.path
                break

        if data_file is None:
            print(f"Warning: No data CSV found in {entry.path}, skipping.")
            continue

        df = pd.read_csv(data_file)

        zone_data = {}
        for param in PARAMETERS:
            zone_data[param] = {}
            for z in ZONES:
                col = col_name(param, z)
                if col in df.columns:
                    zone_data[param][z] = df[col].to_numpy()
                else:
                    print(f"Warning: Column '{col}' not found in {data_file}")
                    zone_data[param][z] = None

        result[key] = zone_data

    return result