"""
object_height.py – Measured-distance analysis using ToF sensor data.

Data organisation
-----------------
Each sub-folder of ``data_root`` must be named ``AANNN``, where:

* ``AA``  – one or more letters (test / position identifier).
* ``NNN`` – one or more digits (e.g. a reference distance in mm).

Computation
-----------
For each ``AANNN`` folder the script:

1. Reads the ``data*.csv`` file and extracts per-zone ``distance_mm`` and
   ``is_valid_range`` columns.
2. Computes the **mean measured distance** for each zone using valid readings
   only (``is_valid_range == 1``).

Visualisation
-------------
Produces a tiled 2×2 figure per folder that exactly matches the layout of
``plot_error_heatmaps`` in ``Visualize.py``:

* Top-left  : 8×8 individual zone heatmap.
* Top-right : 4×4 aggregated groups.
* Bottom-left : 2×2 aggregated groups.
* Bottom-right : Ring aggregations.

Usage
-----
::

    python object_height.py --data-root /path/to/data
    python object_height.py --data-root /path/to/data \\
                            --save-dir /path/to/output
"""

from __future__ import annotations

import argparse
import os
import re
from datetime import datetime
from typing import Optional

import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 – registers 3-D projection
from stl import mesh as stl_mesh

# ── Folder name regex ─────────────────────────────────────────────────────────
_FOLDER_RE = re.compile(r"^([A-Za-z]+)(\d+)$")


# ── Output folder setup ───────────────────────────────────────────────────────

def setup_output_folder(output_path: str) -> str:
    """
    Create a timestamped output subfolder inside *output_path*.

    The subfolder is named ``2D heatmaps-{YYYYMMDD_HHMMSS}`` and is created
    with ``exist_ok=False`` so each run always gets a unique directory.

    Args:
        output_path: Parent directory in which to create the subfolder.

    Returns:
        Full path to the newly created timestamped subfolder.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    full_path = os.path.join(output_path, f"2D heatmaps-{timestamp}")
    os.makedirs(full_path, exist_ok=False)
    return full_path


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

        distance_mm_z{N}        (N = 0 ... 63)
        .is_valid_range_z{N}    (N = 0 ... 63)

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


# ── 3-D heatmap computation and plotting ─────────────────────────────────────

def compute_zone_heights(
    zone_means: dict[int, Optional[float]],
    tolerance_mm: float = 2.0,
) -> dict[int, Optional[float]]:
    """
    Convert per-zone mean distances to object-height values for a 3-D plot.

    Algorithm
    ~~~~~~~~~
    1. Find the **maximum** valid mean distance (``d_max``).
    2. Any zone whose mean distance is within *tolerance_mm* of ``d_max`` is
       considered part of the background and is assigned a height of **0**.
    3. All other valid zones get ``height = d_max - zone_mean``.
       (A closer zone → larger height.)
    4. Zones with no valid data remain ``None``.

    Args:
        zone_means:    Dict of ``zone_index -> mean_distance_mm`` as returned
                       by :func:`compute_zone_valid_means`.
        tolerance_mm:  Distance tolerance for background classification.
                       Defaults to 2 mm.

    Returns:
        Dict of ``zone_index -> height_mm`` (or ``None`` for invalid zones).
    """
    valid_vals = [v for v in zone_means.values() if v is not None]
    if not valid_vals:
        return {z: None for z in zone_means}

    d_max = max(valid_vals)
    heights: dict[int, Optional[float]] = {}
    for z, mean in zone_means.items():
        if mean is None:
            heights[z] = None
        elif abs(mean - d_max) <= tolerance_mm:
            heights[z] = 0.0
        else:
            heights[z] = d_max - mean
    return heights


class ZoneHeightGrid:
    """
    Iterable view over an 8×8 ToF sensor height grid.

    Yields ``(row, col, height_mm)`` tuples when iterated, where ``None``
    heights are replaced with ``0.0``.  Supports repeated iteration.

    Args:
        zone_heights: Dict of ``zone_index -> height_mm`` as returned by
                      :func:`compute_zone_heights`.

    Example::

        grid = ZoneHeightGrid(zone_heights)
        for row, col, h in grid:
            print(row, col, h)

        # or use next() / list()
        it = iter(grid)
        first = next(it)
    """

    def __init__(self, zone_heights: dict[int, Optional[float]]) -> None:
        self._heights = zone_heights
        self._index = 0

    def __iter__(self) -> "ZoneHeightGrid":
        self._index = 0
        return self

    def __next__(self) -> tuple[int, int, float]:
        if self._index >= 64:
            raise StopIteration
        z = self._index
        self._index += 1
        row = z // 8
        col = z % 8
        h = self._heights.get(z)
        return row, col, 0.0 if h is None else h

    def as_matrix(self) -> np.ndarray:
        """Return the heights as an 8×8 numpy array (row-major, None → 0)."""
        mat = np.zeros((8, 8), dtype=float)
        for row, col, h in self:
            mat[row, col] = h
        return mat


def plot_3d_heatmap(
    folder_id: str,
    zone_heights: dict[int, Optional[float]],
    output_path: Optional[str] = None,
) -> None:
    """
    Produce two 3-D height-map visualisations for the given zone heights.

    Static PNG
        A matplotlib ``bar3d`` figure saved as
        ``{output_path}/{folder_id}_3DHeatmap.png``.

    Interactive HTML
        A plotly ``Surface`` figure saved as
        ``{output_path}/{folder_id}_3DHeatmap.html``.
        Open this file in any browser to orbit, zoom and pan the surface
        freely with the mouse.

    Both figures colour-encode height using the ``plasma`` palette.  When
    *output_path* is ``None`` the matplotlib figure is shown interactively
    and no HTML file is written.

    Args:
        folder_id:    Folder identifier string (e.g. ``"ob150"``).
        zone_heights: Dict of ``zone_index -> height_mm`` as returned by
                      :func:`compute_zone_heights`.
        output_path:  Directory in which to save the two output files.
                      If ``None``, the matplotlib plot is displayed
                      interactively.
    """
    grid = ZoneHeightGrid(zone_heights)
    height_matrix = grid.as_matrix()

    colormap = plt.cm.plasma  # type: ignore[attr-defined]
    h_max = float(height_matrix.max())
    norm = mcolors.Normalize(vmin=0.0, vmax=max(h_max, 1.0))

    # ── Static matplotlib bar-chart ──────────────────────────────────────────
    xpos = np.array([col for _, col, _ in grid], dtype=float)
    ypos = np.array([row for row, _, _ in grid], dtype=float)
    dz   = np.array([h   for _, _, h   in grid], dtype=float)
    zpos = np.zeros(64, dtype=float)
    dx = dy = np.full(64, 0.8, dtype=float)
    colors = colormap(norm(dz))

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection="3d")
    ax.bar3d(
        xpos - 0.4, ypos - 0.4, zpos,
        dx, dy, dz,
        color=colors,
        shade=True,
        edgecolor="white",
        linewidth=0.3,
    )
    ax.set_xlabel("Column (0–7)", labelpad=8)
    ax.set_ylabel("Row (0–7)", labelpad=8)
    ax.set_zlabel("Height (mm)", labelpad=8)
    ax.set_title(
        f"3-D Object Height Heatmap — {folder_id}\n"
        "(height = farthest distance − zone mean; background zones = 0)",
        fontsize=11,
    )
    ax.set_xticks(range(8))
    ax.set_yticks(range(8))
    ax.view_init(elev=30, azim=-60)
    sm = plt.cm.ScalarMappable(cmap=colormap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, shrink=0.5, pad=0.1, label="Height (mm)")
    fig.tight_layout()

    if output_path is not None:
        png_path = os.path.join(output_path, f"{folder_id}_3DHeatmap.png")
        plt.savefig(png_path, bbox_inches="tight", dpi=150)
        plt.close(fig)
        print(f"Saved: {png_path}")
    else:
        plt.show()

    # ── Interactive plotly surface (orbitable in browser) ────────────────────
    rows = list(range(8))
    cols = list(range(8))
    plotly_fig = go.Figure(
        data=go.Surface(
            z=height_matrix,
            x=cols,
            y=rows,
            colorscale="Plasma",
            colorbar=dict(title="Height (mm)"),
            hovertemplate="Col: %{x}<br>Row: %{y}<br>Height: %{z:.2f} mm<extra></extra>",
        )
    )
    plotly_fig.update_layout(
        title=dict(
            text=(
                f"3-D Object Height Heatmap — {folder_id}<br>"
                "<sup>height = farthest distance − zone mean; background zones = 0</sup>"
            ),
            x=0.5,
        ),
        scene=dict(
            xaxis_title="Column (0–7)",
            yaxis_title="Row (0–7)",
            zaxis_title="Height (mm)",
            camera=dict(eye=dict(x=1.5, y=-1.5, z=1.2)),
        ),
        margin=dict(l=0, r=0, b=0, t=60),
    )

    if output_path is not None:
        html_path = os.path.join(output_path, f"{folder_id}_3DHeatmap.html")
        plotly_fig.write_html(html_path, include_plotlyjs="cdn")
        print(f"Saved (orbitable): {html_path}")


def export_stl(
    folder_id: str,
    zone_heights: dict[int, Optional[float]],
    platform_h_mm: float,
    sensor_z_mm: float = 330.0,
    output_path: Optional[str] = None,
) -> None:
    """
    Export the 3-D height map as a binary STL file.

    Physical scaling
    ~~~~~~~~~~~~~~~~
    The ToF sensor's field-of-view means that at a stand-off distance of
    ``(sensor_z_mm - platform_h_mm)`` mm the 8×8 zone grid spans a square
    whose **diagonal** equals::

        diagonal = 1.2741 × (sensor_z_mm − platform_h_mm)   [mm]

    From that diagonal the side of the square grid is
    ``diagonal / √2``, and each of the 8 cells along one axis has side::

        cell_mm = diagonal / (8 × √2)

    Heights (z axis) are kept in millimetres as computed by
    :func:`compute_zone_heights`.

    Geometry
    ~~~~~~~~
    Each zone is extruded as a rectangular prism (box):

    * Zones with height > 0 are extruded to their full computed height.
    * Zones with height == 0 (background) receive a 0.1 mm floor tile so
      the base of the scan area is visible in the mesh.

    The resulting mesh is a collection of closed boxes – suitable for
    slicing or further CAD work.

    Args:
        folder_id:      Folder identifier (used in the output filename).
        zone_heights:   Dict of ``zone_index -> height_mm`` from
                        :func:`compute_zone_heights`.
        platform_h_mm:  Height of the measurement platform in mm (the
                        ``NNN`` component of the ``AANNN`` folder name).
        sensor_z_mm:    Distance from sensor origin to the floor in mm.
                        Defaults to 330 mm.
        output_path:    Directory in which to write
                        ``{folder_id}_3DHeatmap.stl``.  If ``None``, the
                        file is written to the current working directory.
    """
    standoff = sensor_z_mm - platform_h_mm
    if standoff <= 0:
        raise ValueError(
            f"sensor_z_mm ({sensor_z_mm}) must be greater than "
            f"platform_h_mm ({platform_h_mm})."
        )

    diagonal_mm = 1.2741 * standoff
    cell_mm = diagonal_mm / (8.0 * np.sqrt(2.0))

    FLOOR_H = 0.1  # mm – minimum height for background tiles

    grid = ZoneHeightGrid(zone_heights)

    # Build triangle list: 12 triangles per box = 12 × 3 vertices per zone
    triangles: list[np.ndarray] = []

    for row, col, h in grid:
        bar_h = h if h > 0.0 else FLOOR_H
        x0 = col * cell_mm
        x1 = x0 + cell_mm
        y0 = row * cell_mm
        y1 = y0 + cell_mm
        z0 = 0.0
        z1 = bar_h

        # 8 box corners
        v = np.array([
            [x0, y0, z0],  # 0 bottom-front-left
            [x1, y0, z0],  # 1 bottom-front-right
            [x1, y1, z0],  # 2 bottom-back-right
            [x0, y1, z0],  # 3 bottom-back-left
            [x0, y0, z1],  # 4 top-front-left
            [x1, y0, z1],  # 5 top-front-right
            [x1, y1, z1],  # 6 top-back-right
            [x0, y1, z1],  # 7 top-back-left
        ], dtype=np.float32)

        faces = [
            # bottom (outward normal −Z)
            [v[0], v[2], v[1]],
            [v[0], v[3], v[2]],
            # top (outward normal +Z)
            [v[4], v[5], v[6]],
            [v[4], v[6], v[7]],
            # front (−Y)
            [v[0], v[1], v[5]],
            [v[0], v[5], v[4]],
            # right (+X)
            [v[1], v[2], v[6]],
            [v[1], v[6], v[5]],
            # back (+Y)
            [v[2], v[3], v[7]],
            [v[2], v[7], v[6]],
            # left (−X)
            [v[3], v[0], v[4]],
            [v[3], v[4], v[7]],
        ]
        triangles.extend(faces)

    n = len(triangles)
    solid = stl_mesh.Mesh(np.zeros(n, dtype=stl_mesh.Mesh.dtype))
    for i, tri in enumerate(triangles):
        for j in range(3):
            solid.vectors[i][j] = tri[j]

    save_dir = output_path if output_path is not None else os.getcwd()
    stl_path = os.path.join(save_dir, f"{folder_id}_3DHeatmap.stl")
    solid.save(stl_path)
    print(f"Saved (STL): {stl_path}")


# ── Heatmap plotting helpers ──────────────────────────────────────────────────

def _color_for_value(
    value: float,
    vmin: float,
    vmax: float,
    colormap,
) -> tuple:
    """Return an RGBA tuple from *colormap* mapped over [*vmin*, *vmax*]."""
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    return colormap(norm(value))


def _draw_distance_heatmap(
    ax,
    zone_values: dict,
    grid_shape: tuple,
    zone_labels: list,
    cell_values: list,
    vmin: float,
    vmax: float,
    colormap,
) -> None:
    """
    Draw a heatmap panel using ``FancyBboxPatch`` cells, matching the visual
    style of ``draw_error_heatmap`` in ``Visualize.py``.

    Cell colour encodes the measured distance via *colormap*.
    Cell text shows the label and the numeric value.

    Args:
        ax:           Matplotlib axes to draw on.
        zone_values:  Dict of ``cell_index -> value`` (may contain ``None``).
        grid_shape:   ``(rows, cols)`` of the grid.
        zone_labels:  Label string for each cell (same order as flattened grid).
        cell_values:  Value for each cell (same order); ``None`` = no data.
        vmin, vmax:   Colour scale limits.
        colormap:     Matplotlib colormap instance.
    """
    rows, cols = grid_shape

    for i, (label, val) in enumerate(zip(zone_labels, cell_values)):
        r = i // cols
        c = i % cols

        if val is not None:
            color = _color_for_value(val, vmin, vmax, colormap)
        else:
            color = (0.85, 0.85, 0.85, 1.0)

        rect = mpatches.FancyBboxPatch(
            (c, r), 1, 1,
            boxstyle="square,pad=0",
            facecolor=color,
            edgecolor="white",
            linewidth=0.5,
            transform=ax.transData,
        )
        ax.add_patch(rect)

        val_text = f"{val:.1f}" if val is not None else "N/A"
        ax.text(
            c + 0.5, r + 0.5,
            f"{label}\n{val_text}",
            ha="center", va="center",
            fontsize=5.5, color="white", fontweight="bold",
        )

    ax.set_xlim(0, cols)
    ax.set_ylim(0, rows)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")


def plot_measured_distance_heatmaps(
    folder_id: str,
    zone_means: dict[int, Optional[float]],
    output_path: Optional[str] = None,
) -> None:
    """
    Plot a tiled 2×2 figure of per-zone mean measured distances, matching the
    multi-panel layout of ``plot_error_heatmaps`` in ``Visualize.py``.

    Panels:
    * Top-left     – 8×8 individual zone heatmap.
    * Top-right    – 4×4 aggregated groups.
    * Bottom-left  – 2×2 aggregated groups.
    * Bottom-right – Ring aggregations.

    Colour encodes the mean measured distance using the ``viridis`` colormap
    across a shared scale derived from the valid zone values.

    Args:
        folder_id:    Folder identifier string (e.g. ``"ob150"``).
        zone_means:   Dict of ``zone_index -> mean measured distance (mm)``
                      as returned by :func:`compute_zone_valid_means`.
        output_path:  If provided, save the figure as
                      ``{output_path}/{folder_id}_MeasuredDistance.png`` and
                      close it.  If ``None``, display interactively.
    """
    def get_ring(z: int) -> int:
        r = z // 8
        c = z % 8
        return min(r, 7 - r, c, 7 - c)

    colormap = plt.cm.viridis  # type: ignore[attr-defined]

    # Shared colour scale across all valid zone values
    valid_vals = [v for v in zone_means.values() if v is not None]
    vmin = min(valid_vals) if valid_vals else 0.0
    vmax = max(valid_vals) if valid_vals else 1.0
    if abs(vmax - vmin) < 1e-6:
        vmin -= 1.0
        vmax += 1.0

    fig, axes = plt.subplots(2, 2, figsize=(16, 16))
    fig.suptitle(
        f"Measured Distance Heatmaps — {folder_id}\n"
        "(text = mean measured distance mm, colour = distance)",
        fontsize=14,
        y=1.002,
    )

    # ── Top-left: 8×8 zones ───────────────────────────────────────────────────
    ax = axes[0][0]
    ax.set_title("8×8 Zone Mean Distance (mm)", fontsize=10)
    labels_8 = [f"Z{z}" for z in range(64)]
    cell_8   = [zone_means.get(z) for z in range(64)]
    _draw_distance_heatmap(ax, zone_means, (8, 8), labels_8, cell_8, vmin, vmax, colormap)

    # ── Top-right: 4×4 aggregated groups ─────────────────────────────────────
    ax = axes[0][1]
    ax.set_title("4×4 Group Mean Distance (mm)", fontsize=10)
    group_vals_4x4: dict[int, Optional[float]] = {}
    labels_4x4: list[str] = []
    cell_4x4:   list[Optional[float]] = []

    for sg in range(16):
        sr = sg // 4
        sc = sg % 4
        vals: list[float] = []
        for dr in range(2):
            for dc in range(2):
                z = (sr * 2 + dr) * 8 + (sc * 2 + dc)
                val_z = zone_means.get(z)
                if val_z is not None:
                    vals.append(val_z)
        group_vals_4x4[sg] = float(np.mean(vals)) if vals else None
        labels_4x4.append(f"G{sg}")
        cell_4x4.append(group_vals_4x4[sg])

    _draw_distance_heatmap(ax, group_vals_4x4, (4, 4), labels_4x4, cell_4x4, vmin, vmax, colormap)

    # ── Bottom-left: 2×2 aggregated groups ───────────────────────────────────
    ax = axes[1][0]
    ax.set_title("2×2 Group Mean Distance (mm)", fontsize=10)
    group_vals_2x2: dict[int, Optional[float]] = {}
    labels_2x2: list[str] = []
    cell_2x2:   list[Optional[float]] = []

    for sg in range(4):
        sr = sg // 2
        sc = sg % 2
        vals = []
        for dr in range(4):
            for dc in range(4):
                z = (sr * 4 + dr) * 8 + (sc * 4 + dc)
                val_z = zone_means.get(z)
                if val_z is not None:
                    vals.append(val_z)
        group_vals_2x2[sg] = float(np.mean(vals)) if vals else None
        labels_2x2.append(f"G{sg}")
        cell_2x2.append(group_vals_2x2[sg])

    _draw_distance_heatmap(ax, group_vals_2x2, (2, 2), labels_2x2, cell_2x2, vmin, vmax, colormap)

    # ── Bottom-right: Ring aggregations ──────────────────────────────────────
    ax = axes[1][1]
    ax.set_title("Ring Mean Distance (mm)", fontsize=10)

    ring_vals_all: dict[int, list[float]] = {0: [], 1: [], 2: [], 3: []}
    for z in range(64):
        ring = get_ring(z)
        val_z = zone_means.get(z)
        if val_z is not None:
            ring_vals_all[ring].append(val_z)

    ring_means: dict[int, Optional[float]] = {
        r: (float(np.mean(v)) if v else None)
        for r, v in ring_vals_all.items()
    }

    for z in range(64):
        r    = z // 8
        c    = z % 8
        ring = get_ring(z)
        val  = ring_means.get(ring)

        color = (
            _color_for_value(val, vmin, vmax, colormap)
            if val is not None
            else (0.85, 0.85, 0.85, 1.0)
        )

        rect = mpatches.FancyBboxPatch(
            (c, r), 1, 1,
            boxstyle="square,pad=0",
            facecolor=color,
            edgecolor="white",
            linewidth=0.5,
        )
        ax.add_patch(rect)

        # Label each ring once at its top-left cell
        if (c == ring) and (r == (7 - ring)):
            val_text = f"{val:.1f}" if val is not None else "N/A"
            ax.text(
                c + 0.5, r + 0.5,
                f"R{ring}\n{val_text}",
                ha="center", va="center",
                fontsize=7, color="white", fontweight="bold",
            )

    # Ring boundary lines
    for ring in range(1, 4):
        o = ring
        ax.plot(
            [o, 8 - o, 8 - o, o, o],
            [o, o, 8 - o, 8 - o, o],
            color="white", linewidth=1.5, linestyle="--",
        )

    ax.set_xlim(0, 8)
    ax.set_ylim(0, 8)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")

    fig.subplots_adjust(right=0.88, top=0.94, hspace=0.25, wspace=0.15)

    # Shared colorbar on the right edge
    cbar_ax = fig.add_axes([0.91, 0.1, 0.02, 0.75])
    sm = plt.cm.ScalarMappable(cmap=colormap, norm=mcolors.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    fig.colorbar(sm, cax=cbar_ax, label="Mean measured distance (mm)")

    if output_path is not None:
        save_path = os.path.join(output_path, f"{folder_id}_MeasuredDistance.png")
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        plt.close(fig)
        print(f"Saved: {save_path}")
    else:
        plt.show()


# ── Entrypoint ────────────────────────────────────────────────────────────────

def analyse_measured_distances(
    data_root: str,
    save_dir: Optional[str] = None,
    sensor_z_mm: float = 330.0,
) -> dict[str, dict[int, Optional[float]]]:
    """
    Discover ``AANNN`` subfolders in ``data_root``, compute per-zone mean
    measured distances, and produce a multi-panel heatmap for each folder.

    Folder discovery
    ~~~~~~~~~~~~~~~~
    Only direct children of ``data_root`` that are directories *and* match the
    ``AANNN`` naming pattern (letters then digits) are processed.  Others are
    skipped with a warning.

    For each folder the ``NNN`` numeric suffix is used as the platform height
    (in mm) when computing the physical grid size for the STL export.

    Args:
        data_root:   Root directory containing ``AANNN`` data subfolders.
        save_dir:    Directory to write output files.  Created automatically if
                     it does not exist.  If ``None``, figures are shown
                     interactively and no STL files are written.
        sensor_z_mm: Distance from the sensor origin to the floor in mm.
                     Used to compute the physical size of the STL grid via
                     ``diagonal = 1.2741 × (sensor_z_mm − platform_h_mm)``.
                     Defaults to 330 mm.

    Returns:
        Dict of ``{folder_id: {zone_index: mean_distance_mm}}`` for every
        successfully processed folder.

    Raises:
        ValueError: If ``data_root`` is not an existing directory.
    """
    if not os.path.isdir(data_root):
        raise ValueError(f"data_root is not a directory: {data_root!r}")

    actual_save_dir: Optional[str] = None
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        actual_save_dir = setup_output_folder(save_dir)

    results: dict[str, dict[int, Optional[float]]] = {}
    skipped: list[str] = []

    for entry in sorted(os.scandir(data_root), key=lambda e: e.name):
        if not entry.is_dir():
            continue

        try:
            _letter_code, platform_h = parse_folder_name(entry.name)
        except ValueError as exc:
            print(f"Skipping {entry.name!r}: {exc}")
            skipped.append(entry.name)
            continue

        print(f"Processing {entry.name!r}")

        zone_data = _load_zone_data_from_folder(entry.path)
        if zone_data is None:
            skipped.append(entry.name)
            continue

        zone_means = compute_zone_valid_means(zone_data)

        plot_measured_distance_heatmaps(
            folder_id=entry.name,
            zone_means=zone_means,
            output_path=actual_save_dir,
        )

        zone_heights = compute_zone_heights(zone_means)
        plot_3d_heatmap(
            folder_id=entry.name,
            zone_heights=zone_heights,
            output_path=actual_save_dir,
        )

        if actual_save_dir is not None:
            export_stl(
                folder_id=entry.name,
                zone_heights=zone_heights,
                platform_h_mm=float(platform_h),
                sensor_z_mm=sensor_z_mm,
                output_path=actual_save_dir,
            )

        results[entry.name] = zone_means

    if skipped:
        print(f"\nSkipped folders: {skipped}")

    return results


def analyse_object_heights(
    data_root: str,
    origin: Optional[float] = None,
    save_dir: Optional[str] = None,
    sensor_z_mm: float = 330.0,
) -> dict[str, dict[int, Optional[float]]]:
    """
    Backwards-compatible alias for :func:`analyse_measured_distances`.

    The *origin* parameter is accepted but no longer used in the computation;
    raw measured distances are displayed instead of derived object heights.

    Args:
        data_root:   Root directory containing ``AANNN`` data subfolders.
        origin:      Accepted for backwards compatibility; ignored.
        save_dir:    Directory to write output PNG files.  If ``None``, figures
                     are shown interactively.
        sensor_z_mm: Sensor origin height in mm; forwarded to
                     :func:`analyse_measured_distances`.

    Returns:
        Dict of ``{folder_id: {zone_index: mean_distance_mm}}``.
    """
    return analyse_measured_distances(
        data_root=data_root, save_dir=save_dir, sensor_z_mm=sensor_z_mm
    )


# ── CLI ───────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Visualise per-zone mean measured distances from ToF sensor data "
            "organised in AANNN subfolders.  Produces a tiled 2×2 heatmap "
            "(8×8, 4×4, 2×2, rings) per folder, an orbitable interactive HTML "
            "3-D surface, and an STL mesh scaled to the sensor field-of-view."
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
        default=None,
        metavar="MM",
        help="Accepted for backwards compatibility; not used in the computation.",
    )
    p.add_argument(
        "--save-dir",
        default=None,
        metavar="DIR",
        help=(
            "Directory to save output files (PNG, HTML, STL).  "
            "If omitted, the matplotlib figure is shown interactively."
        ),
    )
    p.add_argument(
        "--sensor-z",
        type=float,
        default=330.0,
        metavar="MM",
        help=(
            "Sensor origin height above the floor in mm.  Used to compute the "
            "physical grid size for the STL export via "
            "diagonal = 1.2741 × (sensor-z − platform_h).  Default: 330."
        ),
    )
    return p


if __name__ == "__main__":
    _args = _build_parser().parse_args()
    analyse_measured_distances(
        data_root=_args.data_root,
        save_dir=_args.save_dir,
        sensor_z_mm=_args.sensor_z,
    )
