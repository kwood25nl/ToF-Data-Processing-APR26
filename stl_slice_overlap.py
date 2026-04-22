from __future__ import annotations
import os
from datetime import datetime
from typing import List, Optional, Tuple

import numpy as np
import trimesh
import matplotlib.pyplot as plt

try:
    import mplcursors as _mplcursors
except ImportError:
    _mplcursors = None  # type: ignore[assignment]

try:
    from shapely.geometry import Polygon
    from shapely.ops import unary_union
except Exception:
    Polygon = None
    unary_union = None


# ----------------------------
# EDIT THESE PATHS
# ----------------------------
EXPERIMENT_STL_PATH = r"Z:\Kaitie\ToF and Mono Camera Experiments APR26\Calibration Object Data 2APR2026\04-00004\sc144\outputs-20260421_135119_FOV55\sc144_3DHeatmap.stl"
TRUE_STL_PATH = r"Z:\Kaitie\LiDAR\Calibration Objects\10cmCAL_pyd.STL"

# ----------------------------
# RUN-TIME FLAGS
# Set these to True to enable the corresponding behaviour.
# ----------------------------
# Save aligned STL files to disk on every run.
EXPORT_STL: bool = False
# Save the overlap plot to a PNG file on every run.
SAVE_PNG: bool = False
# Pixel radius within which the fallback cursor snaps to the nearest data point.
CURSOR_SNAP_THRESHOLD_PX: int = 20


PLANE_NORMAL = {
    "xy": np.array([0.0, 0.0, 1.0]),
    "xz": np.array([0.0, 1.0, 0.0]),
    "yz": np.array([1.0, 0.0, 0.0]),
}

PLANE_AXES_IDX = {
    "xy": (0, 1),
    "xz": (0, 2),
    "yz": (1, 2),
}

PLANE_AXES_LABEL = {
    "xy": ("X (mm)", "Y (mm)"),
    "xz": ("X (mm)", "Z (mm)"),
    "yz": ("Y (mm)", "Z (mm)"),
}


def _ts() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def _normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < eps:
        return v * 0.0
    return v / n


def _set_equal(ax):
    ax.set_aspect("equal", adjustable="box")


def _rotation_matrix_from_vectors(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    4x4 rotation matrix that rotates vector a onto vector b.
    Handles parallel and anti-parallel cases.
    """
    a = _normalize(a)
    b = _normalize(b)

    dot = float(np.clip(np.dot(a, b), -1.0, 1.0))

    if dot > 1.0 - 1e-10:
        return np.eye(4)

    if dot < -1.0 + 1e-10:
        candidate = np.array([1.0, 0.0, 0.0])
        if abs(np.dot(candidate, a)) > 0.9:
            candidate = np.array([0.0, 1.0, 0.0])
        axis = _normalize(np.cross(a, candidate))
        angle = np.pi
        return trimesh.transformations.rotation_matrix(angle, axis)

    axis = _normalize(np.cross(a, b))
    angle = float(np.arccos(dot))
    return trimesh.transformations.rotation_matrix(angle, axis)


def cluster_faces_by_normal(
    face_normals: np.ndarray,
    face_areas: np.ndarray,
    dot_threshold: float = 0.98,
) -> list[dict]:
    """
    Greedy clustering of faces by similar normals.

    cluster dict:
      rep: representative normal
      idx: face indices
      area: summed face area
    """
    clusters: list[dict] = []
    for i, n in enumerate(face_normals):
        assigned = False
        for c in clusters:
            if float(np.dot(n, c["rep"])) >= dot_threshold:
                c["idx"].append(i)
                c["area"] += float(face_areas[i])
                assigned = True
                break
        if not assigned:
            clusters.append({"rep": n.copy(), "idx": [i], "area": float(face_areas[i])})
    return clusters


def detect_bottom_normal(mesh: trimesh.Trimesh, dot_threshold: float = 0.98) -> Tuple[np.ndarray, list[int], list[dict]]:
    # 1) face normals + areas
    face_normals = np.asarray(mesh.face_normals, dtype=float)
    face_areas = np.asarray(mesh.area_faces, dtype=float)

    # 2) cluster by similar normals
    clusters = cluster_faces_by_normal(face_normals, face_areas, dot_threshold=dot_threshold)

    # 3-4) largest area cluster
    best = max(clusters, key=lambda c: c["area"])
    idx = best["idx"]

    # 5) area-weighted avg normal
    n = (face_normals[idx] * face_areas[idx, None]).sum(axis=0)
    n = _normalize(n)

    return n, idx, clusters


def align_mesh_bottom_to_xy(
    mesh: trimesh.Trimesh,
    dot_threshold: float = 0.98,
    centroid_mode: str = "bbox",  # "bbox" or "mean"
) -> Tuple[trimesh.Trimesh, dict]:
    """
    1-10 as requested:
      rotate bottom normal -> +Z
      translate so minZ=0
      center in XY
    """
    m = mesh.copy()

    bottom_n, bottom_faces, clusters = detect_bottom_normal(m, dot_threshold=dot_threshold)

    # 6-7) rotate to +Z
    R = _rotation_matrix_from_vectors(bottom_n, np.array([0.0, 0.0, 1.0]))
    m.apply_transform(R)

    # 8) translate so lowest Z = 0
    min_z = float(m.vertices[:, 2].min())
    Tz = trimesh.transformations.translation_matrix([0.0, 0.0, -min_z])
    m.apply_transform(Tz)

    # 9) XY centroid
    if centroid_mode == "bbox":
        c = m.bounding_box.centroid
        xy_center = np.array([c[0], c[1], 0.0], dtype=float)
    elif centroid_mode == "mean":
        c = m.vertices.mean(axis=0)
        xy_center = np.array([c[0], c[1], 0.0], dtype=float)
    else:
        raise ValueError("centroid_mode must be 'bbox' or 'mean'")

    # 10) translate so centroid at (0,0)
    Txy = trimesh.transformations.translation_matrix([-xy_center[0], -xy_center[1], 0.0])
    m.apply_transform(Txy)

    info = {
        "bottom_normal_original": bottom_n,
        "bottom_faces_count": len(bottom_faces),
        "clusters_count": len(clusters),
        "largest_cluster_area": float(max(c["area"] for c in clusters)) if clusters else 0.0,
        "min_z_after_rotation": min_z,
        "xy_center_removed": xy_center,
    }
    return m, info


# ----------------------------
# Slicing + plotting check
# ----------------------------
def _section_3d_polylines(mesh: trimesh.Trimesh, origin_xyz: np.ndarray, normal_xyz: np.ndarray) -> List[np.ndarray]:
    section = mesh.section(plane_origin=origin_xyz, plane_normal=normal_xyz)
    if section is None:
        return []
    discrete = section.discrete
    if callable(discrete):
        discrete = discrete()
    out = []
    for seg in discrete:
        arr = np.asarray(seg, dtype=float)
        if arr.ndim == 2 and arr.shape[1] == 3 and arr.shape[0] >= 2:
            out.append(arr)
    return out


def _project_3d_to_2d(polylines3d: List[np.ndarray], plane_key: str) -> List[np.ndarray]:
    i0, i1 = PLANE_AXES_IDX[plane_key]
    return [pl[:, [i0, i1]] for pl in polylines3d]


def _plot_polylines2d(ax, polylines2d: List[np.ndarray], color: str, lw: float = 1.5, label: Optional[str] = None) -> list:
    """Plot 2-D polylines on *ax* and return the list of Line2D objects."""
    lines = []
    first = True
    for pl in polylines2d:
        if pl.shape[0] < 2:
            continue
        if first and label is not None:
            (ln,) = ax.plot(pl[:, 0], pl[:, 1], color=color, lw=lw, label=label)
            first = False
        else:
            (ln,) = ax.plot(pl[:, 0], pl[:, 1], color=color, lw=lw)
        lines.append(ln)
    return lines


def _attach_cursor(fig, all_lines: list) -> None:
    """
    Attach an interactive data cursor to *all_lines*.

    Tries mplcursors first (pip install mplcursors).
    Falls back to a lightweight matplotlib motion-event annotation.
    """
    if _mplcursors is not None:
        cursor = _mplcursors.cursor(all_lines, hover=True)

        @cursor.connect("add")
        def _on_add(sel):
            x, y = sel.target
            sel.annotation.set_text(f"x={x:.3f}\ny={y:.3f}")
            sel.annotation.get_bbox_patch().set(fc="lightyellow", alpha=0.85)

        return  # mplcursors handles everything

    # --- Fallback: plain matplotlib event-based cursor ---
    axes_set = {ln.axes for ln in all_lines if ln.axes is not None}
    annots: dict = {}
    for ax in axes_set:
        ann = ax.annotate(
            "",
            xy=(0, 0),
            xytext=(12, 12),
            textcoords="offset points",
            bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", ec="gray", alpha=0.85),
            fontsize=8,
            visible=False,
        )
        annots[ax] = ann

    def _on_motion(event):
        if event.inaxes is None:
            for ann in annots.values():
                if ann.get_visible():
                    ann.set_visible(False)
            fig.canvas.draw_idle()
            return

        ax = event.inaxes
        ann = annots.get(ax)
        if ann is None:
            return

        closest_dist = float("inf")
        found_x = found_y = None
        for ln in all_lines:
            if ln.axes is not ax:
                continue
            xd = ln.get_xdata()
            yd = ln.get_ydata()
            if len(xd) == 0:
                continue
            # distance in display coords
            try:
                pts_display = ax.transData.transform(np.column_stack([xd, yd]))
                cursor_display = np.array([event.x, event.y])
                dists = np.linalg.norm(pts_display - cursor_display, axis=1)
                idx = int(np.argmin(dists))
                d = float(dists[idx])
                if d < closest_dist:
                    closest_dist = d
                    found_x, found_y = float(xd[idx]), float(yd[idx])
            except (ValueError, TypeError):
                pass

        if found_x is not None and closest_dist < CURSOR_SNAP_THRESHOLD_PX:
            ann.xy = (found_x, found_y)
            ann.set_text(f"x={found_x:.3f}\ny={found_y:.3f}")
            ann.set_visible(True)
        else:
            ann.set_visible(False)
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("motion_notify_event", _on_motion)


def show_overlap_outlines_3planes(
    exp_mesh: trimesh.Trimesh,
    true_mesh: trimesh.Trimesh,
    save_png: bool = False,
    out_dir: str = "outputs",
    filename_prefix: str = "overlap_check",
) -> Optional[str]:
    """
    Slice both meshes along XY, XZ, YZ planes through TRUE centroid and overlay
    outlines in an interactive matplotlib window.

    Parameters
    ----------
    exp_mesh, true_mesh : trimesh.Trimesh
        Aligned meshes to compare.
    save_png : bool
        When True the figure is also saved to *out_dir* as a timestamped PNG.
    out_dir : str
        Directory for the optional PNG output.
    filename_prefix : str
        Prefix for the PNG filename.

    Returns
    -------
    str or None
        Path to the saved PNG if *save_png* is True, else None.
    """
    origin = true_mesh.bounding_box.centroid

    fig = plt.figure(figsize=(10, 10))
    fig.suptitle("Outline overlap check (Cartesian mm) — slices through TRUE centroid", fontsize=13)
    gs = fig.add_gridspec(2, 2)

    ax_xy = fig.add_subplot(gs[:, 0])  # col 0, spans both rows
    ax_xz = fig.add_subplot(gs[0, 1])  # col 1, row 0
    ax_yz = fig.add_subplot(gs[1, 1])  # col 1, row 1


    all_lines: list = []

    for ax, plane_key, title in zip([ax_xy, ax_xz, ax_yz], ["xy", "xz", "yz"], ["XY", "XZ", "YZ"]):
        normal = PLANE_NORMAL[plane_key]
        xlab, ylab = PLANE_AXES_LABEL[plane_key]

        exp_3d = _section_3d_polylines(exp_mesh, origin, normal)
        true_3d = _section_3d_polylines(true_mesh, origin, normal)

        exp_2d = _project_3d_to_2d(exp_3d, plane_key)
        true_2d = _project_3d_to_2d(true_3d, plane_key)

        all_lines.extend(_plot_polylines2d(ax, true_2d, color="#18A5AA", lw=2.0, label="Actual"))
        all_lines.extend(_plot_polylines2d(ax, exp_2d, color="#E97122", lw=1.2, label="Measured"))

        ax.set_title(title)
        ax.set_xlabel(xlab)
        ax.set_ylabel(ylab)
        ax.legend(loc="best")
        _set_equal(ax)

    plt.tight_layout()
    _attach_cursor(fig, all_lines)

    out_path: Optional[str] = None
    if save_png:
        _ensure_dir(out_dir)
        out_path = os.path.join(out_dir, f"{filename_prefix}_{_ts()}.png")
        fig.savefig(out_path, dpi=200)
        print("Wrote:", out_path)

    plt.show()
    return out_path


def save_overlap_outlines_3planes(
    exp_mesh: trimesh.Trimesh,
    true_mesh: trimesh.Trimesh,
    out_dir: str,
    filename_prefix: str = "overlap_check",
) -> str:
    """
    Backwards-compatible wrapper: always saves a PNG and closes the figure.

    Prefer *show_overlap_outlines_3planes* for interactive use.
    """
    out_dir = _ensure_dir(out_dir)
    origin = true_mesh.bounding_box.centroid

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Outline overlap check (Cartesian mm) — slices through TRUE centroid", fontsize=13)

    for ax, plane_key, title in zip(axes, ["xy", "xz", "yz"], ["XY", "XZ", "YZ"]):
        normal = PLANE_NORMAL[plane_key]
        xlab, ylab = PLANE_AXES_LABEL[plane_key]

        exp_3d = _section_3d_polylines(exp_mesh, origin, normal)
        true_3d = _section_3d_polylines(true_mesh, origin, normal)

        exp_2d = _project_3d_to_2d(exp_3d, plane_key)
        true_2d = _project_3d_to_2d(true_3d, plane_key)

        _plot_polylines2d(ax, true_2d, color="#18A5AA", lw=2.0, label="Actual")
        _plot_polylines2d(ax, exp_2d, color="#E97122", lw=1.2, label="Measured")

        ax.set_title(title)
        ax.set_xlabel(xlab)
        ax.set_ylabel(ylab)
        ax.legend(loc="best")
        _set_equal(ax)

    plt.tight_layout()
    out_path = os.path.join(out_dir, f"{filename_prefix}_{_ts()}.png")
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


# ----------------------------
# Example run
# ----------------------------
if __name__ == "__main__":
    out_dir = "outputs"

    true_mesh = trimesh.load(TRUE_STL_PATH, force="mesh")
    exp_mesh = trimesh.load(EXPERIMENT_STL_PATH, force="mesh")

    true_aligned, true_info = align_mesh_bottom_to_xy(true_mesh, dot_threshold=0.98, centroid_mode="bbox")
    exp_aligned, exp_info = align_mesh_bottom_to_xy(exp_mesh, dot_threshold=0.98, centroid_mode="bbox")

    # Show interactive overlap plot (optionally saves PNG when SAVE_PNG=True).
    show_overlap_outlines_3planes(
        exp_aligned,
        true_aligned,
        save_png=SAVE_PNG,
        out_dir=out_dir,
        filename_prefix="aligned_overlap_3planes",
    )
