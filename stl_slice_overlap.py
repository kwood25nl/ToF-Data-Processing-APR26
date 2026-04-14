import argparse
import trimesh
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import MultiPolygon, Polygon


def compute_overlap(experimental_slice, true_slice):
    """Compute the overlap area between two slices."""
    if experimental_slice is None or true_slice is None:
        return 0
    # Convert to polygons
    experimental_polygons = experimental_slice.to_planar().polygons
    true_polygons = true_slice.to_planar().polygons
    total_area = 0
    for exp_poly in experimental_polygons:
        for true_poly in true_polygons:
            intersection = exp_poly.intersection(true_poly)
            if isinstance(intersection, MultiPolygon):
                total_area += sum(poly.area for poly in intersection.geoms)
            else:
                total_area += intersection.area
    return total_area


def main():
    parser = argparse.ArgumentParser(description='Compute STL slice overlap areas.')
    parser.add_argument('--N', type=int, default=1, help='Number of slices for experimentBody')
    parser.add_argument('--axis', choices=['x', 'y', 'z'], required=True, help='Slicing plane orientation')
    parser.add_argument('--out', type=str, default='overlap_summary.png', help='Output plot file path')
    parser.add_argument('--csv', action='store_true', help='Save per-slice overlap areas to CSV')
    args = parser.parse_args()

    # Load STL meshes
    experiment_body = trimesh.load('experimentBody.stl')
    true_body = trimesh.load('trueBody.stl')

    # Compute mesh midpoint
    midpoint = experiment_body.bounding_box.centroid
    normal = {'x': [1, 0, 0], 'y': [0, 1, 0], 'z': [0, 0, 1]}[args.axis]

    overlap_areas = []

    for i in range(args.N):
        angle = i * (np.pi / args.N)
        rotation = trimesh.transformations.rotation_matrix(angle, normal)
        rotated_midpoint = np.dot(rotation[:3, :3], midpoint)
        experimental_slice = experiment_body.section(plane_origin=rotated_midpoint, plane_normal=normal)
        true_slice = true_body.section(plane_origin=midpoint, plane_normal=np.array([0, 0, 1]))

        overlap_area = compute_overlap(experimental_slice, true_slice)
        overlap_areas.append(overlap_area)

    # Summary statistics
    if args.N <= 2:
        mean_overlap = np.mean(overlap_areas)
        print(f'Mean overlap area: {mean_overlap}')
        plt.plot(overlap_areas)
        plt.xlabel('Slice Index')
        plt.ylabel('Overlap Area')
        plt.title('Overlap Areas for N <= 2')
        plt.savefig(args.out)
    else:
        min_overlap = np.min(overlap_areas)
        max_overlap = np.max(overlap_areas)
        median_overlap = np.median(overlap_areas)
        print(f'Min: {min_overlap}, Max: {max_overlap}, Median: {median_overlap}')
        plt.plot(overlap_areas)
        plt.xlabel('Slice Index')
        plt.ylabel('Overlap Area')
        plt.title('Overlap Areas for N >= 3')
        plt.savefig(args.out)

    # Save to CSV if requested
    if args.csv:
        import pandas as pd
        df = pd.DataFrame({'Angle': [i * (np.pi / args.N) for i in range(args.N)], 'Overlap Area': overlap_areas})
        df.to_csv('overlap_areas.csv', index=False)


if __name__ == '__main__':
    main()