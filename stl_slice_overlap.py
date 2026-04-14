import trimesh
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

# Function to load STL files
def load_stl(file_path):
    return trimesh.load(file_path)

# Function to render cross-sections
def render_cross_sections(mesh, rotation_axis, starting_plane, num_slices, out_dir):
    # Compute angles for slices
    angles = np.linspace(0, 180, num_slices)
    for angle in angles:
        # Rotate mesh
        mesh_copy = mesh.copy()
        mesh_copy.apply_transform(trimesh.transformations.rotation_matrix(np.radians(angle), rotation_axis))
        slice = mesh_copy.section(plane=starting_plane)
        # Extract 2D polylines
        contours = slice.to_polygons()
        # Plot and save
        plt.figure()
        for contour in contours:
            plt.plot(contour[:, 0], contour[:, 1])
        plt.title(f'Slice at {angle} degrees')
        plt.savefig(os.path.join(out_dir, f'slice_{angle}.png'))
        plt.close()

# Main function
def main(experiment_stl_path, true_stl_path, out_dir):
    experiment_mesh = load_stl(experiment_stl_path)
    true_mesh = load_stl(true_stl_path)
    # Prompt for rotation axes and other parameters
    exp_rotation_axis = np.array([1,0,0])  # Example axis for rotation
    true_rotation_axis = np.array([0,1,0])  # Example axis for true mesh
    starting_plane = 'xy'  # Example starting plane
    num_slices = 10  # Example number of slices
    # Render cross-sections
    render_cross_sections(experiment_mesh, exp_rotation_axis, starting_plane, num_slices, out_dir)
    # Additional processing for true mesh
    true_slice = true_mesh.section(plane=starting_plane)
    # Save true slice as PNG
    plt.figure()
    plt.plot(true_slice[:, 0], true_slice[:, 1])
    plt.title('True Slice')
    plt.savefig(os.path.join(out_dir, f'true_slice.png'))
    plt.close()

# Example usage
# main('path_to_experiment.stl', 'path_to_true.stl', 'output_directory')
