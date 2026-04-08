import re
import numpy as np

def trim_tof_data(data: dict) -> tuple[dict, int, str]:
    """
    Trims all distances in the dataset to the same number of frames (shortest).

    Args:
        data: Output from load_tof_data.

    Returns:
        Tuple of (trimmed_data, n_frames, shortest_key) where n_frames is the number of frames
        all datasets were trimmed to, and shortest_key is the dataset that determined the trim length.
    """

    # Find the shortest dataset across all distances and zones
    min_frames = None
    shortest_key = None
    for key, zone_data in data.items():
        for param, zones in zone_data.items():
            for z, arr in zones.items():
                if arr is not None:
                    n = len(arr)
                    if min_frames is None or n < min_frames:
                        min_frames = n
                        shortest_key = key

    if min_frames is None:
        raise ValueError("No valid data found in dataset.")

    # Trim all arrays to min_frames from the end
    trimmed = {}
    for key, zone_data in data.items():
        trimmed[key] = {}
        for param, zones in zone_data.items():
            trimmed[key][param] = {}
            for z, arr in zones.items():
                if arr is not None:
                    trimmed[key][param][z] = arr[:min_frames]
                else:
                    trimmed[key][param][z] = None

    return trimmed, min_frames, shortest_key


def analyse_tof_data(data: dict) -> dict:
    """
    Calculates per-zone and whole-array statistics for each distance.

    Args:
        data: Output from trim_tof_data (or load_tof_data).

    Returns:
        A dict keyed by distance (e.g. "d110") containing:
            - "per_zone": per-zone stats (mean, std, percent_valid) for
                          distance_mm, signal_per_spad, ambient_per_spad
            - "array": whole-array stats (mean, std, percent_valid, error)
                       for distance_mm, signal_per_spad, ambient_per_spad
    """

    import numpy as np

    PARAMS = ["distance_mm", "signal_per_spad", "ambient_per_spad"]
    ZONES = range(64)

    def extract_true_distance(key: str) -> float:
        digits = re.findall(r"-?\d+", key)
        return float("".join(digits))

    results = {}

    for key, zone_data in data.items():
        true_distance = extract_true_distance(key)
        per_zone = {p: {} for p in PARAMS}
        per_zone["percent_valid"] = {}

        # Stack validity across all zones for whole-array stats
        all_valid = []

        for z in ZONES:
            valid_arr = zone_data["is_valid_range"][z]
            all_valid.append(valid_arr)

            # Per-zone validity
            if valid_arr is not None:
                per_zone["percent_valid"][z] = float(np.mean(valid_arr) * 100)
            else:
                per_zone["percent_valid"][z] = None

            # Per-zone stats for each parameter
            for param in PARAMS:
                arr = zone_data[param][z]
                if arr is not None:
                    per_zone[param][z] = {
                        "mean": float(np.mean(arr)),
                        "std":  float(np.std(arr))
                    }
                else:
                    per_zone[param][z] = {"mean": None, "std": None}

        # Whole-array stats — flatten across all zones and all frames
        array_stats = {}
        for param in PARAMS:
            all_vals = []
            for z in ZONES:
                arr = zone_data[param][z]
                if arr is not None:
                    all_vals.append(arr)
            if all_vals:
                flat = np.concatenate(all_vals)
                array_stats[param] = {
                    "mean": float(np.mean(flat)),
                    "std":  float(np.std(flat)),
                    "error": float(np.mean(flat) - true_distance) if param == "distance_mm" else None
                }
            else:
                array_stats[param] = {"mean": None, "std": None, "error": None}

        # Whole-array percent validity
        if all_valid:
            flat_valid = np.concatenate([v for v in all_valid if v is not None])
            array_stats["percent_valid"] = float(np.mean(flat_valid) * 100)
        else:
            array_stats["percent_valid"] = None

        results[key] = {
            "per_zone": per_zone,
            "array":    array_stats
        }

    return results