import os
import tempfile
from datetime import datetime

import numpy as np
import streamlit as st

try:
    import trimesh
except Exception as e:  # pragma: no cover
    trimesh = None


AXES = {
    "X": np.array([1.0, 0.0, 0.0]),
    "Y": np.array([0.0, 1.0, 0.0]),
    "Z": np.array([0.0, 0.0, 1.0]),
}

def _now_ts() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def _save_upload_to_temp(uploaded_file, suffix=".stl") -> str:
    """Persist Streamlit uploaded file to a temporary file and return the path."""
    if uploaded_file is None:
        raise ValueError("No file uploaded")
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(uploaded_file.getbuffer())
    tmp.flush()
    tmp.close()
    return tmp.name

def _mesh_summary(mesh) -> dict:
    bb = mesh.bounding_box
    extents = bb.extents
    center = bb.centroid
    return {
        "faces": int(getattr(mesh, "faces", np.empty((0, 3))).shape[0]),
        "vertices": int(getattr(mesh, "vertices", np.empty((0, 3))).shape[0]),
        "extents_xyz": [float(extents[0]), float(extents[1]), float(extents[2])],
        "center_xyz": [float(center[0]), float(center[1]), float(center[2])],
    }

def _rotation_about_axis(axis_unit: np.ndarray, angle_rad: float) -> np.ndarray:
    """Return a 3x3 rotation matrix rotating by angle about a unit axis."""
    a = axis_unit / np.linalg.norm(axis_unit)
    K = np.array(
        [[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]],
        dtype=float,
    )
    I = np.eye(3)
    return I + np.sin(angle_rad) * K + (1 - np.cos(angle_rad)) * (K @ K)

def _section_polygons(mesh, plane_origin: np.ndarray, plane_normal: np.ndarray):
    """Return shapely polygons for a mesh section, or empty list."""
    section = mesh.section(plane_origin=plane_origin, plane_normal=plane_normal)
    if section is None:
        return []
    planar, _ = section.to_planar()
    return list(planar.polygons)

def _overlap_area(polys_a, polys_b) -> float:
    """Total intersection area between two polygon sets."""
    if not polys_a or not polys_b:
        return 0.0
    total = 0.0
    for pa in polys_a:
        for pb in polys_b:
            inter = pa.intersection(pb)
            if inter.is_empty:
                continue
            total += float(inter.area)
    return total

def compute_overlaps(
    experiment_mesh,
    true_mesh,
    axis_label: str,
    n_slices: int,
    slice_span_deg: float,
):
    """Compute overlap areas.

    Interpretation (matches your description):
    - All slices pass through the experimental mesh midpoint.
    - We choose a rotation axis (X/Y/Z). That axis is the *rotation axis*.
    - We rotate the slicing plane around that axis.
    - The trueBody is sliced once at angle=0 (same plane as experimental slice 0).

    slice_span_deg:
      - Total angular span covered.
      - 180 means unique planes (because plane at theta and theta+180 are the same).
    """
    axis_unit = AXES[axis_label]

    midpoint = experiment_mesh.bounding_box.centroid

    candidate = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(candidate, axis_unit)) > 0.9:
        candidate = np.array([0.0, 1.0, 0.0])
    base_normal = np.cross(axis_unit, candidate)
    base_normal = base_normal / np.linalg.norm(base_normal)

    true_polys = _section_polygons(true_mesh, plane_origin=midpoint, plane_normal=base_normal)

    overlaps = []
    angles = []

    span_rad = np.deg2rad(slice_span_deg)

    if n_slices <= 0:
        raise ValueError("n_slices must be >= 1")

    for i in range(n_slices):
        theta = 0.0 if n_slices == 1 else (i * span_rad / n_slices)
        R = _rotation_about_axis(axis_unit, theta)
        n_i = R @ base_normal

        exp_polys = _section_polygons(experiment_mesh, plane_origin=midpoint, plane_normal=n_i)
        ov = _overlap_area(exp_polys, true_polys)
        overlaps.append(ov)
        angles.append(theta)

    overlaps = np.array(overlaps, dtype=float)
    angles = np.array(angles, dtype=float)

    stats = {
        "mean": float(np.mean(overlaps)) if overlaps.size else 0.0,
        "min": float(np.min(overlaps)) if overlaps.size else 0.0,
        "max": float(np.max(overlaps)) if overlaps.size else 0.0,
        "median": float(np.median(overlaps)) if overlaps.size else 0.0,
    }
    return angles, overlaps, stats

def main():
    st.set_page_config(page_title="STL Slice Overlap", layout="wide")

    st.title("STL slice overlap")
    st.caption("Upload experimentBody and trueBody STL, choose axis, choose number of slices, and compute overlap statistics.")

    if trimesh is None:
        st.error("Missing dependency: trimesh. Install requirements and restart.")
        st.stop()

    with st.sidebar:
        st.header("Inputs")
        exp_file = st.file_uploader("Experiment STL (experimentBody)", type=["stl"], key="exp")
        true_file = st.file_uploader("True STL (trueBody)", type=["stl"], key="true")

        st.header("Slicing")
        axis_label = st.selectbox("Rotation axis", options=list(AXES.keys()), index=2)  # default Z
        n_slices = st.slider("Number of experimental slices", min_value=1, max_value=90, value=6, step=1)
        slice_span_deg = st.selectbox("Angular span", options=[90, 180, 360], index=1)

        st.header("Output")
        out_dir = st.text_input("Output directory (server-side)", value="outputs")
        out_base = st.text_input("Output basename", value="overlap")
        do_save = st.checkbox("Save CSV + plot to output directory", value=True)

        compute_btn = st.button("Compute overlap", type="primary")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Experiment STL")
        if exp_file is None:
            st.info("Upload an experiment STL to preview it.")
        else:
            exp_path = _save_upload_to_temp(exp_file)
            exp_mesh = trimesh.load(exp_path, force="mesh")
            st.json(_mesh_summary(exp_mesh))
            try:
                glb = exp_mesh.export(file_type="glb")
                st.download_button("Download experiment as GLB", data=glb, file_name="experiment.glb")
                st._legacy_v1.write(st._legacy_v1.html(""))  # no-op to avoid lint
                st.subheader("3D preview")
                st.write("If your Streamlit version supports it, you can use st.model_viewer. Otherwise download GLB.")
                if hasattr(st, "model_viewer"):
                    st.model_viewer(glb)
            except Exception as e:
                st.warning(f"Could not create 3D preview (GLB export failed): {e}")

    with col2:
        st.subheader("True STL")
        if true_file is None:
            st.info("Upload a true STL to preview it.")
        else:
            true_path = _save_upload_to_temp(true_file)
            true_mesh = trimesh.load(true_path, force="mesh")
            st.json(_mesh_summary(true_mesh))
            try:
                glb = true_mesh.export(file_type="glb")
                st.download_button("Download true as GLB", data=glb, file_name="true.glb")
                st.subheader("3D preview")
                st.write("If your Streamlit version supports it, you can use st.model_viewer. Otherwise download GLB.")
                if hasattr(st, "model_viewer"):
                    st.model_viewer(glb)
            except Exception as e:
                st.warning(f"Could not create 3D preview (GLB export failed): {e}")

    if compute_btn:
        if exp_file is None or true_file is None:
            st.error("Please upload BOTH STLs.")
            st.stop()

        exp_path = _save_upload_to_temp(exp_file)
        true_path = _save_upload_to_temp(true_file)
        exp_mesh = trimesh.load(exp_path, force="mesh")
        true_mesh = trimesh.load(true_path, force="mesh")

        angles, overlaps, stats = compute_overlaps(
            experiment_mesh=exp_mesh,
            true_mesh=true_mesh,
            axis_label=axis_label,
            n_slices=n_slices,
            slice_span_deg=float(slice_span_deg),
        )

        st.subheader("Results")
        if n_slices <= 2:
            st.metric("Mean overlap area", f"{stats['mean']:.6g}")
        else:
            c1, c2, c3 = st.columns(3)
            c1.metric("Min overlap area", f"{stats['min']:.6g}")
            c2.metric("Median overlap area", f"{stats['median']:.6g}")
            c3.metric("Max overlap area", f"{stats['max']:.6g}")

        st.line_chart({"overlap_area": overlaps})

        if do_save:
            os.makedirs(out_dir, exist_ok=True)
            ts = _now_ts()
            csv_path = os.path.join(out_dir, f"{out_base}_{ts}.csv")
            png_path = os.path.join(out_dir, f"{out_base}_{ts}.png")

            import pandas as pd
            df = pd.DataFrame({"angle_rad": angles, "overlap_area": overlaps})
            df.to_csv(csv_path, index=False)

            import matplotlib.pyplot as plt
            plt.figure()
            plt.plot(overlaps)
            plt.xlabel("slice index")
            plt.ylabel("overlap area")
            plt.title(f"overlap areas (axis={axis_label}, N={n_slices})")
            plt.tight_layout()
            plt.savefig(png_path, dpi=200)
            plt.close()

            st.success(f"Saved: {csv_path} and {png_path}")


if __name__ == "__main__":
    main()