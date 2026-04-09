"""
object_height.py – Calibrated object-height analysis using ToF sensor data.

Data organisation
-----------------
Each sub-folder of ``data_root`` must be named ``AANNN``, where:

* ``AA``  – one or more letters (test / position identifier).
* ``NNN`` – one or more digits representing the **platform distance from the
  sensor** in the same units as the ToF distance measurements (mm).

Computation
-----------
For each ``AANNN`` folder the script:

1. Reads the ``data*.csv`` file and extracts per-zone ``distance_mm`` and
   ``is_valid_range`` columns.
2. Computes the **mean measured distance** for each zone using valid readings
   only.
3. Computes the **object height above the platform**::

       object_height = platform_distance − calibrated_distance

   where ``calibrated_distance`` is the per-zone mean measured distance.
   A *positive* value means the object surface is *closer* to the sensor
   than the empty platform, i.e. the object protrudes toward the sensor.

Sign convention note
~~~~~~~~~~~~~~~~~~~~
``origin`` is accepted for documentation / cross-check purposes and is used
to validate that the caller's reference frame is consistent.  The active
computation only needs ``platform_distance`` (from the folder name) and the
raw ToF readings.

Usage
-----
::

    python object_height.py --data-root /path/to/data --origin 330
    python object_height.py --data-root /path/to/data --origin 330 \\
                            --save-dir /path/to/output
"""

from __future__ import annotations

import argparse
import os
import re
from typing import Optional

import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ── Folder name regex ─────────────────────────────────────────────────────────
_FOLDER_RE = re.compile(r"^([A-Za-z]+)(\d+)$")


# ── Folder name parsing ───────────────────────────────────────────────────────

def parse_folder_name(name: str) -> tuple[str, int]:
    """
    Parse an ``AANNN`` folder name into a (letter_code, platform_distance) pair.

    Args:
        name: Folder name, e.g. ``"ob150"`` or ``"FF330"``.

    Returns:
        ``(letter_code, platform_distance_mm)`` – the alphabetic prefix and the
        numeric suffix interpreted as a distance in mm.

    Raises:
        ValueError: If *name* does not match the ``AANNN`` pattern (one or more
            letters followed by one or more digits).
    """
    match = _FOLDER_RE.match(name)
    if match is None:
        raise ValueError(
            f"Folder name {name!r} does not match the AANNN pattern "
            "(one or more letters followed by one or more digits)."
        )
    return match.group(1), int(match.group(2))


# ── Data loading ──────────────────────────────────────────────────────────────

def _load_zone_data_from_folder(folder_path: str) -> Optional[dict]:
    """
    Load ToF zone data from a single ``AANNN`` folder.

    The folder must contain exactly one file whose name starts with ``"data"``
    and ends with ``".csv"``.  The CSV is expected to have columns named::

        distance_mm_z{N}        (N = 0 … 63)
        .is_valid_range_z{N}    (N = 0 … 63)

    which is the format produced by the existing experiment pipeline.

    Args:
        folder_path: Absolute path to the ``AANNN`` folder.

    Returns:
        Dict with keys ``"distance_mm"`` and ``"is_valid_range"``, each
        mapping ``zone_index -> numpy.ndarray``.  Returns ``None`` if no
        data CSV is found.
    """
    ZONES = range(64)
    PARAMETERS = ["distance_mm", "is_valid_range"]

    def _col_name(param: str, zone: int) -> str:
        if param == "is_valid_range":
            return f".is_valid_range_z{zone}"
        return f"{param}_z{zone}"

    data_file: Optional[str] = None
    for entry in os.scandir(folder_path):
        if entry.name.startswith("data") and entry.name.endswith(".csv"):
            data_file = entry.path
            break

    if data_file is None:
        print(f"Warning: No data CSV found in {folder_path!r}, skipping.")
        return None

    df = pd.read_csv(data_file)

    zone_data: dict = {}
    for param in PARAMETERS:
        zone_data[param] = {}
        for z in ZONES:
            col = _col_name(param, z)
            if col in df.columns:
                zone_data[param][z] = df[col].to_numpy()
            else:
                print(f"Warning: Column {col!r} not found in {data_file!r}")
                zone_data[param][z] = None

    return zone_data


# ── Per-zone statistics ───────────────────────────────────────────────────────

def compute_zone_valid_means(zone_data: dict) -> dict[int, Optional[float]]:
    """
    Compute the per-zone mean of *valid* ``distance_mm`` readings.

    Validity is determined by the ``is_valid_range`` flag (value ``== 1``).

    Args:
        zone_data: Dict returned by :func:`_load_zone_data_from_folder`.

    Returns:
        Dict mapping ``zone_index -> mean_distance_mm``, or ``None`` for zones
        that have no valid readings.
    """
    means: dict[int, Optional[float]] = {}
    for z in range(64):
        dist_arr = zone_data["distance_mm"][z]
        valid_arr = zone_data["is_valid_range"][z]
        if dist_arr is not None and valid_arr is not None:
            mask = valid_arr == 1
            if mask.any():
                means[z] = float(np.mean(dist_arr[mask]))
            else:
                means[z] = None
        else:
            means[z] = None
    return means


# ── Object height computation ─────────────────────────────────────────────────

def compute_object_heights(
    zone_means: dict[int, Optional[float]],
    platform_distance: int,
) -> dict[int, Optional[float]]:
    """
    Compute the per-zone object height above the platform.

    **Sign convention** – the sensor points *downward* toward the platform::

        object_height = platform_distance − calibrated_distance

    * Positive  → object surface is *closer* to the sensor than the platform
      (object is sitting on the platform and protrudes toward the sensor).
    * Zero      → zone reads exactly at platform level (no object).
    * Negative  → zone reads *beyond* the platform (sensor sees past edge of
      platform, or erroneous reading).

    Args:
        zone_means:        Dict of ``zone_index -> mean measured distance (mm)``
                           as returned by :func:`compute_zone_valid_means`.
        platform_distance: Sensor-to-platform distance in mm (from folder name).

    Returns:
        Dict of ``zone_index -> object height above platform (mm)``, with
        ``None`` for zones that had no valid readings.
    """
    heights: dict[int, Optional[float]] = {}
    for z in range(64):
        mean = zone_means.get(z)
        if mean is not None:
            heights[z] = float(platform_distance - mean)
        else:
            heights[z] = None
    return heights


# ── Heatmap plotting ──────────────────────────────────────────────────────────

def plot_object_height_heatmap(
    folder_id: str,
    platform_distance: int,
    object_heights: dict[int, Optional[float]],
    output_path: Optional[str] = None,
) -> None:
    """
    Plot an 8×8 heatmap of per-zone object heights above the platform.

    Layout mirrors the existing ``draw_heatmap`` / ``plot_error_heatmaps``
    style (``FancyBboxPatch`` cells, white grid lines, zone labels inside
    cells), but uses a continuous ``viridis`` colormap and a colorbar instead
    of the categorical error colouring.

    Zone numbering orientation:
    * Z0  → bottom-left  (row 0, col 0 in matplotlib data coordinates).
    * Z63 → top-right    (row 7, col 7).

    Args:
        folder_id:         Folder identifier string (e.g. ``"ob150"``).
        platform_distance: Sensor-to-platform distance in mm.
        object_heights:    Dict of ``zone_index -> object height (mm)`` or
                           ``None``.
        output_path:       If provided, save the figure as
                           ``{output_path}/{folder_id}_ObjectHeight.png`` and
                           close it.  If ``None``, display interactively.
    """
    ROWS, COLS = 8, 8

    # Normalise colormap across valid height values
    valid_heights = [h for h in object_heights.values() if h is not None]
    if valid_heights:
        vmin = min(valid_heights)
        vmax = max(valid_heights)
    else:
        vmin, vmax = 0.0, 1.0

    # Avoid degenerate range (all cells identical)
    if abs(vmax - vmin) < 1e-6:
        vmin -= 1.0
        vmax += 1.0

    colormap = plt.cm.viridis  # type: ignore[attr-defined]
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    fig, ax = plt.subplots(figsize=(9, 9))
    fig.suptitle(
        f"Object Height Heatmap — {folder_id}\n"
        f"Platform distance: {platform_distance} mm  |  "
        f"height = platform_distance − measured_distance",
        fontsize=12,
        y=1.01,
    )

    for z in range(64):
        r = z // COLS  # physical row (0 = bottom in matplotlib data coords)
        c = z % COLS
        h = object_heights.get(z)

        if h is not None:
            rgba = colormap(norm(h))
        else:
            rgba = (0.85, 0.85, 0.85, 1.0)  # light grey for missing data

        rect = mpatches.FancyBboxPatch(
            (c, r), 1, 1,
            boxstyle="square,pad=0",
            facecolor=rgba,
            edgecolor="white",
            linewidth=0.5,
        )
        ax.add_patch(rect)

        text_val = f"{h:.1f}" if h is not None else "N/A"
        ax.text(
            c + 0.5, r + 0.5,
            f"Z{z}\n{text_val}",
            ha="center", va="center",
            fontsize=6, color="white", fontweight="bold",
        )

    ax.set_xlim(0, COLS)
    ax.set_ylim(0, ROWS)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")
    ax.set_xlabel("Column (zone % 8)", fontsize=9)
    ax.set_ylabel("Row (zone // 8)", fontsize=9)

    # Colorbar
    sm = cm.ScalarMappable(cmap=colormap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Object height above platform (mm)", fontsize=9)

    plt.tight_layout()

    if output_path is not None:
        save_path = os.path.join(output_path, f"{folder_id}_ObjectHeight.png")
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        plt.close(fig)
        print(f"Saved: {save_path}")
    else:
        plt.show()


# ── Entrypoint ────────────────────────────────────────────────────────────────

def analyse_object_heights(
    data_root: str,
    origin: float,
    save_dir: Optional[str] = None,
) -> dict[str, dict[int, Optional[float]]]:
    """
    Discover ``AANNN`` subfolders in ``data_root``, compute per-zone object
    heights, and produce a heatmap for each folder.

    Folder discovery
    ~~~~~~~~~~~~~~~~
    Only direct children of ``data_root`` that are directories *and* match the
    ``AANNN`` naming pattern (letters then digits) are processed.  Others are
    skipped with a warning.

    Args:
        data_root: Root directory containing ``AANNN`` data subfolders.
        origin:    Sensor-origin reference distance (mm).  Stored for
                   documentation / cross-check; the actual height computation
                   uses ``platform_distance`` extracted from each folder name.
        save_dir:  Directory to write output ``*_ObjectHeight.png`` files.
                   Created automatically if it does not exist.  If ``None``,
                   figures are shown interactively.

    Returns:
        Dict of ``{folder_id: {zone_index: object_height_mm}}`` for every
        successfully processed folder.

    Raises:
        ValueError: If ``data_root`` is not an existing directory.
    """
    if not os.path.isdir(data_root):
        raise ValueError(f"data_root is not a directory: {data_root!r}")

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    results: dict[str, dict[int, Optional[float]]] = {}
    skipped: list[str] = []

    for entry in sorted(os.scandir(data_root), key=lambda e: e.name):
        if not entry.is_dir():
            continue

        try:
            _letter_code, platform_distance = parse_folder_name(entry.name)
        except ValueError as exc:
            print(f"Skipping {entry.name!r}: {exc}")
            skipped.append(entry.name)
            continue

        print(
            f"Processing {entry.name!r}  "
            f"(platform distance = {platform_distance} mm, origin = {origin} mm)"
        )

        zone_data = _load_zone_data_from_folder(entry.path)
        if zone_data is None:
            skipped.append(entry.name)
            continue

        zone_means = compute_zone_valid_means(zone_data)
        heights = compute_object_heights(zone_means, platform_distance)

        plot_object_height_heatmap(
            folder_id=entry.name,
            platform_distance=platform_distance,
            object_heights=heights,
            output_path=save_dir,
        )

        results[entry.name] = heights

    if skipped:
        print(f"\nSkipped folders: {skipped}")

    return results


# ── CLI ───────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Compute and visualise per-zone object heights above a platform "
            "from ToF sensor data organised in AANNN subfolders."
        )
    )
    p.add_argument(
        "--data-root",
        required=True,
        metavar="DIR",
        help="Root directory containing AANNN data subfolders.",
    )
    p.add_argument(
        "--origin",
        type=float,
        required=True,
        metavar="MM",
        help="Sensor-origin reference distance in mm.",
    )
    p.add_argument(
        "--save-dir",
        default=None,
        metavar="DIR",
        help=(
            "Directory to save output heatmap PNGs.  "
            "If omitted, figures are shown interactively."
        ),
    )
    return p


if __name__ == "__main__":
    _args = _build_parser().parse_args()
    analyse_object_heights(
        data_root=_args.data_root,
        origin=_args.origin,
        save_dir=_args.save_dir,
    )
