import argparse
import numpy as np
import trimesh
import matplotlib.pyplot as plt
import os
import csv
from shapely.geometry import Polygon

# Requirements:
# trimesh
# matplotlib
# shapely

def load_mesh(file_path):
    return trimesh.load(file_path)

def create_static_preview(experiment_mesh, true_mesh):
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))
    # Mid-plane cross-section for experiment
    for i, plane in enumerate(['xy', 'xz', 'yz']):
        section = experiment_mesh.section(plane='mid', plane_normal=[1, 0, 0], transform=None)
        if section and hasattr(section, 'polygons'):  # Handle Path2D
            for polygon in section.polygons:
                L = plt.Line2D(*zip(*polygon), label='Experiment', color='blue')
                axs[0, i].add_line(L)
                axs[0, i].set_title(f'Experiment {plane}')
                axs[0, i].set_xlabel(f'{plane} axis')
                axs[0, i].set_ylabel('Height')

        section_true = true_mesh.section(plane='mid', plane_normal=[1, 0, 0], transform=None)
        if section_true and hasattr(section_true, 'polygons'):
            for polygon in section_true.polygons:
                L = plt.Line2D(*zip(*polygon), label='True', color='red')
                axs[1, i].add_line(L)
                axs[1, i].set_title(f'True {plane}')
                axs[1, i].set_xlabel(f'{plane} axis')
                axs[1, i].set_ylabel('Height')

    plt.tight_layout()
    plt.savefig('static_preview.png')
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='STL Slice Overlap Tool')
    parser.add_argument('--experiment-stl', type=str, help='Path to experiment STL file')
    parser.add_argument('--true-stl', type=str, help='Path to true STL file')
    parser.add_argument('--out-dir', type=str, default='outputs', help='Output directory')
    args = parser.parse_args()

    # Fallback if paths are not provided
    if not args.experiment_stl:
        args.experiment_stl = input('Enter experiment STL file path: ')
    if not args.true_stl:
        args.true_stl = input('Enter true STL file path: ')

    experiment_mesh = load_mesh(args.experiment_stl)
    true_mesh = load_mesh(args.true_stl)

    # Create static preview
    create_static_preview(experiment_mesh, true_mesh)

    # Additional interactions and (slice computations omitted for brevity)...

if __name__ == '__main__':
    main()