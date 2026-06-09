"""
Time-of-Flight Sensor Data Surface Reconstruction Tool
Converts 8x8 sensor data to 3D meshes and visualizations

Inspired by object_height.py approach for proper mesh generation
"""

import os
import sys
import csv
from datetime import datetime
from pathlib import Path
from typing import Tuple, List, Optional
import numpy as np
from collections import deque
import warnings

warnings.filterwarnings('ignore')

try:
    import pandas as pd
    from scipy.spatial import KDTree
    from scipy.interpolate import griddata
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import matplotlib.colors as mcolors
    from mpl_toolkits.mplot3d import Axes3D
    import cv2
    from stl import mesh as stl_mesh
    import plotly.graph_objects as go
except ImportError as e:
    print(f"Missing required package: {e}")
    print("Install with: pip install pandas scipy matplotlib opencv-python numpy-stl plotly")
    sys.exit(1)


class ToFSensorProcessor:
    """Process Time-of-Flight sensor data and generate surface reconstructions"""
    
    def __init__(self, csv_path: str):
        """
        Initialize processor with CSV file
        
        Args:
            csv_path: Path to input CSV file
        """
        self.csv_path = Path(csv_path)
        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        
        # Sensor specifications
        self.grid_size = 8
        self.fov_diagonal = 65  # degrees
        self.fov_half = 45 / 2  # half angle in each direction
        
        # Load data
        self.df = pd.read_csv(csv_path)
        self.num_frames = len(self.df)
        
        # Create output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = self.csv_path.parent / f"tof_output_{timestamp}"
        self.output_dir.mkdir(exist_ok=True)
        
        print(f"Loaded {self.num_frames} frames from {csv_path}")
        print(f"Output directory: {self.output_dir}")
    
    def _extract_frame_data(self, frame_idx: int) -> np.ndarray:
        """
        Extract and validate distance data for a frame
        
        Args:
            frame_idx: Frame index
            
        Returns:
            Array of 64 distances in mm, with invalid ranges filled
        """
        row = self.df.iloc[frame_idx]
        distances = []
        valid_flags = []
        
        # Extract distance and validity for each pixel
        for z in range(64):
            dist_col = f'distance_mm_z{z}'
            valid_col = f'.is_valid_range_z{z}'
            
            if dist_col not in self.df.columns or valid_col not in self.df.columns:
                distances.append(0)
                valid_flags.append(False)
            else:
                dist = float(row[dist_col]) if pd.notna(row[dist_col]) else 0
                valid = bool(row[valid_col]) if pd.notna(row[valid_col]) else False
                distances.append(dist)
                valid_flags.append(valid)
        
        distances = np.array(distances)
        valid_flags = np.array(valid_flags)
        
        # Fill invalid pixels with nearest valid neighbor (up to 2 zones away)
        filled_distances = self._fill_invalid_pixels(distances, valid_flags)
        
        return filled_distances
    
    def _fill_invalid_pixels(self, distances: np.ndarray, valid_flags: np.ndarray) -> np.ndarray:
        """
        Fill invalid pixels with nearest valid neighbor value
        Search up to 2 zones away
        
        Args:
            distances: Array of 64 distances
            valid_flags: Array of 64 validity flags
            
        Returns:
            Array with filled distances
        """
        filled = distances.copy()
        grid = distances.reshape(8, 8)
        valid_grid = valid_flags.reshape(8, 8)
        
        for row in range(8):
            for col in range(8):
                if not valid_grid[row, col]:
                    best_dist = None
                    best_distance_metric = float('inf')
                    
                    for dr in range(-2, 3):
                        for dc in range(-2, 3):
                            if dr == 0 and dc == 0:
                                continue
                            nr, nc = row + dr, col + dc
                            if 0 <= nr < 8 and 0 <= nc < 8 and valid_grid[nr, nc]:
                                dist_metric = max(abs(dr), abs(dc))
                                if dist_metric < best_distance_metric:
                                    best_distance_metric = dist_metric
                                    best_dist = grid[nr, nc]
                    
                    if best_dist is not None:
                        filled[row * 8 + col] = best_dist
                    else:
                        filled[row * 8 + col] = 0
        
        return filled
    
    def _compute_zone_heights(self, distances: np.ndarray) -> Tuple[dict, float, float]:
        """
        Convert distances to heights using a platform approach
        
        Args:
            distances: Array of 64 distances
            
        Returns:
            Tuple of (zone_heights_dict, platform_height, cell_size_mm)
        """
        # Find maximum distance (platform level)
        valid_distances = distances[distances > 0]
        if len(valid_distances) == 0:
            max_dist = 100
        else:
            max_dist = float(np.max(valid_distances))
        
        # Platform is 1mm below the farthest point
        platform_height = max_dist + 1.0
        
        # Heights are distance from object to platform
        zone_heights = {}
        for z in range(64):
            if distances[z] > 0:
                zone_heights[z] = platform_height - distances[z]
            else:
                zone_heights[z] = None
        
        # Calculate cell size based on FOV
        diagonal_mm = 2 * platform_height * np.tan(np.deg2rad(self.fov_half))
        cell_mm = diagonal_mm / (8.0 * np.sqrt(2.0))
        
        return zone_heights, platform_height, cell_mm
    
    def _create_mesh_from_heights(self, zone_heights: dict, cell_mm: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create 3D mesh using platform approach (like object_height.py)
        
        Args:
            zone_heights: Dictionary of zone heights
            cell_mm: Cell size in mm
            
        Returns:
            Tuple of (vertices, faces)
        """
        # Build 8x8 height matrix
        heights = np.zeros((8, 8), dtype=np.float32)
        for z in range(64):
            row = z // 8
            col = z % 8
            h = zone_heights.get(z)
            heights[row, col] = h if h is not None and h > 0 else 0.1
        
        triangles = []
        
        # Floor covering entire 8x8 grid
        gx0, gy0 = 0.0, 0.0
        gx1, gy1 = 8 * cell_mm, 8 * cell_mm
        triangles += [
            [np.array([gx0, gy0, 0.0], dtype=np.float32),
             np.array([gx1, gy0, 0.0], dtype=np.float32),
             np.array([gx1, gy1, 0.0], dtype=np.float32)],
            [np.array([gx0, gy0, 0.0], dtype=np.float32),
             np.array([gx1, gy1, 0.0], dtype=np.float32),
             np.array([gx0, gy1, 0.0], dtype=np.float32)],
        ]
        
        # Each zone as a rectangular column
        for row in range(8):
            for col in range(8):
                h = heights[row, col]
                x0, x1 = col * cell_mm, (col + 1) * cell_mm
                y0, y1 = row * cell_mm, (row + 1) * cell_mm
                
                # Top face
                triangles += [
                    [np.array([x0, y0, h], dtype=np.float32),
                     np.array([x1, y0, h], dtype=np.float32),
                     np.array([x1, y1, h], dtype=np.float32)],
                    [np.array([x0, y0, h], dtype=np.float32),
                     np.array([x1, y1, h], dtype=np.float32),
                     np.array([x0, y1, h], dtype=np.float32)],
                ]
                
                # Side walls (only exposed sides)
                # Front wall
                n_h = heights[row - 1, col] if row > 0 else 0.0
                if h > n_h:
                    triangles += [
                        [np.array([x0, y0, n_h], dtype=np.float32),
                         np.array([x0, y0, h], dtype=np.float32),
                         np.array([x1, y0, h], dtype=np.float32)],
                        [np.array([x0, y0, n_h], dtype=np.float32),
                         np.array([x1, y0, h], dtype=np.float32),
                         np.array([x1, y0, n_h], dtype=np.float32)],
                    ]
                
                # Back wall
                n_h = heights[row + 1, col] if row < 7 else 0.0
                if h > n_h:
                    triangles += [
                        [np.array([x1, y1, n_h], dtype=np.float32),
                         np.array([x1, y1, h], dtype=np.float32),
                         np.array([x0, y1, h], dtype=np.float32)],
                        [np.array([x1, y1, n_h], dtype=np.float32),
                         np.array([x0, y1, h], dtype=np.float32),
                         np.array([x0, y1, n_h], dtype=np.float32)],
                    ]
                
                # Left wall
                n_h = heights[row, col - 1] if col > 0 else 0.0
                if h > n_h:
                    triangles += [
                        [np.array([x0, y1, n_h], dtype=np.float32),
                         np.array([x0, y1, h], dtype=np.float32),
                         np.array([x0, y0, h], dtype=np.float32)],
                        [np.array([x0, y1, n_h], dtype=np.float32),
                         np.array([x0, y0, h], dtype=np.float32),
                         np.array([x0, y0, n_h], dtype=np.float32)],
                    ]
                
                # Right wall
                n_h = heights[row, col + 1] if col < 7 else 0.0
                if h > n_h:
                    triangles += [
                        [np.array([x1, y0, n_h], dtype=np.float32),
                         np.array([x1, y0, h], dtype=np.float32),
                         np.array([x1, y1, h], dtype=np.float32)],
                        [np.array([x1, y0, n_h], dtype=np.float32),
                         np.array([x1, y1, h], dtype=np.float32),
                         np.array([x1, y1, n_h], dtype=np.float32)],
                    ]
        
        # Convert to vertices and faces
        n = len(triangles)
        vertices_list = []
        faces = []
        vertex_map = {}
        
        for tri in triangles:
            face_indices = []
            for vertex in tri:
                v_tuple = tuple(vertex)
                if v_tuple not in vertex_map:
                    vertex_map[v_tuple] = len(vertices_list)
                    vertices_list.append(vertex)
                face_indices.append(vertex_map[v_tuple])
            faces.append(face_indices)
        
        return np.array(vertices_list), np.array(faces)
    
    def _write_ply(self, vertices: np.ndarray, faces: np.ndarray, output_path: str):
        """Write mesh to PLY file"""
        num_vertices = len(vertices)
        num_faces = len(faces)
        
        with open(output_path, 'w') as f:
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {num_vertices}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write(f"element face {num_faces}\n")
            f.write("property list uchar int vertex_indices\n")
            f.write("end_header\n")
            
            for v in vertices:
                f.write(f"{v[0]:.2f} {v[1]:.2f} {v[2]:.2f}\n")
            
            for face in faces:
                f.write(f"3 {int(face[0])} {int(face[1])} {int(face[2])}\n")
        
        print(f"Saved PLY: {output_path}")
    
    def _write_stl(self, vertices: np.ndarray, faces: np.ndarray, output_path: str):
        """Write mesh to STL file (binary)"""
        # Create mesh object
        mesh_obj = stl_mesh.Mesh(np.zeros(len(faces), dtype=stl_mesh.Mesh.dtype))
        
        # Fill in the mesh
        for i, face in enumerate(faces):
            for j in range(3):
                mesh_obj.vectors[i][j] = vertices[face[j]]
        
        # Write to file
        mesh_obj.save(output_path)
        print(f"Saved STL: {output_path}")
    
    def _create_html_3d(self, distances: np.ndarray, output_path: str, frame_idx: int):
        """Create interactive 3D HTML visualization using Plotly"""
        zone_heights, _, cell_mm = self._compute_zone_heights(distances)
        
        # Build mesh geometry
        all_x, all_y, all_z = [], [], []
        all_i, all_j, all_k = [], [], []
        hover_text = []
        colors_intensity = []
        
        # Heights matrix for reference
        heights = np.zeros((8, 8), dtype=np.float32)
        for z in range(64):
            row = z // 8
            col = z % 8
            h = zone_heights.get(z)
            heights[row, col] = h if h is not None and h > 0 else 0.1
        
        # Build mesh for each zone
        for row in range(8):
            for col in range(8):
                h = heights[row, col]
                x0, x1 = col * cell_mm, (col + 1) * cell_mm
                y0, y1 = row * cell_mm, (row + 1) * cell_mm
                z0, z1 = 0.0, h
                
                base = len(all_x)
                
                # 8 vertices
                all_x.extend([x0, x1, x1, x0, x0, x1, x1, x0])
                all_y.extend([y0, y0, y1, y1, y0, y0, y1, y1])
                all_z.extend([z0, z0, z0, z0, z1, z1, z1, z1])
                colors_intensity.extend([h] * 8)
                
                # 12 triangles (6 faces)
                face_indices = [
                    (0, 1, 2), (0, 2, 3),  # bottom
                    (4, 5, 6), (4, 6, 7),  # top
                    (0, 1, 5), (0, 5, 4),  # front
                    (2, 3, 7), (2, 7, 6),  # back
                    (0, 3, 7), (0, 7, 4),  # left
                    (1, 2, 6), (1, 6, 5),  # right
                ]
                
                for a, b, c in face_indices:
                    all_i.append(base + a)
                    all_j.append(base + b)
                    all_k.append(base + c)
                    zone = row * 8 + col
                    hover_text.append(f"Zone: Z{zone}<br>Col: {col}<br>Row: {row}<br>Height: {h:.2f} mm")
        
        # Create Plotly figure
        fig = go.Figure(data=[
            go.Mesh3d(
                x=all_x, y=all_y, z=all_z,
                i=all_i, j=all_j, k=all_k,
                intensity=colors_intensity,
                intensitymode='vertex',
                colorscale='Viridis',
                colorbar=dict(title='Height (mm)'),
                showscale=True,
                flatshading=True,
                hovertemplate='%{text}<extra></extra>',
                text=hover_text[:len(all_i)],
            )
        ])
        
        fig.update_layout(
            title=f'ToF Sensor 3D Reconstruction - Frame {frame_idx}',
            scene=dict(
                xaxis_title='X (mm)',
                yaxis_title='Y (mm)',
                zaxis_title='Height (mm)',
                camera=dict(eye=dict(x=1.5, y=-1.5, z=1.2)),
            ),
            width=1000,
            height=800,
        )
        
        fig.write_html(output_path, include_plotlyjs='cdn')
        print(f"Saved HTML visualization: {output_path}")
    
    def _create_heatmap_2d(self, distances: np.ndarray, output_path: str, frame_idx: int):
        """Create 2D heatmap visualization"""
        heatmap = distances.reshape(8, 8)
        
        fig, ax = plt.subplots(figsize=(12, 12), dpi=100)
        
        im = ax.imshow(heatmap, cmap='viridis', aspect='equal', origin='upper')
        ax.set_title(f'Distance Heatmap - Frame {frame_idx}', fontsize=14)
        ax.set_xlabel('Column', fontsize=12)
        ax.set_ylabel('Row', fontsize=12)
        
        # Add grid and labels
        ax.set_xticks(np.arange(8))
        ax.set_yticks(np.arange(8))
        ax.grid(True, alpha=0.3, color='white', linewidth=0.5)
        
        # Add text annotations
        for i in range(8):
            for j in range(8):
                zone = i * 8 + j
                text = ax.text(j, i, f'Z{zone}\n{heatmap[i, j]:.0f}',
                             ha="center", va="center", color="white", fontsize=7)
        
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Distance (mm)', fontsize=12)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=100, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved heatmap: {output_path}")
    
    def _create_animation_frame(self, frame_idx: int, dpi: int = 150) -> np.ndarray:
        """Create a single 3D plot frame as image"""
        distances = self._extract_frame_data(frame_idx)
        zone_heights, _, cell_mm = self._compute_zone_heights(distances)
        
        fig = plt.figure(figsize=(12, 10), dpi=dpi)
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot bars for each zone
        heights = np.zeros((8, 8), dtype=np.float32)
        for z in range(64):
            row = z // 8
            col = z % 8
            h = zone_heights.get(z)
            heights[row, col] = h if h is not None and h > 0 else 0.1
        
        # Create color map
        colors = plt.cm.viridis(heights / heights.max())
        
        # Plot bars
        label_offset = max(float(heights.max()) * 0.02, 0.2)
        for row in range(8):
            for col in range(8):
                h = heights[row, col]
                ax.bar3d(col, row, 0, 1, 1, h, color=colors[row, col], 
                        shade=True, edgecolor='black', linewidth=0.3)
                zone = row * 8 + col
                ax.text(col + 0.5, row + 0.5, h + label_offset, f'Z{zone}',
                        ha='center', va='bottom', fontsize=6, color='black')
        
        ax.set_xlabel('Column')
        ax.set_ylabel('Row')
        ax.set_zlabel('Height (mm)')
        ax.set_title(f'Frame {frame_idx}')
        ax.view_init(elev=30, azim=-60)
        
        # Render to image
        temp_path = self.output_dir / ".temp_frame.png"
        fig.savefig(str(temp_path), dpi=dpi, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        
        image = cv2.imread(str(temp_path))
        temp_path.unlink()
        
        return image
    
    def _create_heatmap_frame(self, frame_idx: int, dpi: int = 150) -> np.ndarray:
        """Create a 2D heatmap frame for video"""
        distances = self._extract_frame_data(frame_idx)
        heatmap = distances.reshape(8, 8)
        
        fig, ax = plt.subplots(figsize=(12, 12), dpi=dpi)
        
        im = ax.imshow(heatmap, cmap='viridis', aspect='equal', origin='upper')
        ax.set_title(f'Distance Heatmap - Frame {frame_idx}', fontsize=14)
        ax.set_xlabel('Column', fontsize=12)
        ax.set_ylabel('Row', fontsize=12)
        ax.set_xticks(np.arange(8))
        ax.set_yticks(np.arange(8))
        ax.grid(True, alpha=0.3, color='white')
        
        plt.colorbar(im, ax=ax, label='Distance (mm)')
        
        temp_path = self.output_dir / ".temp_heatmap.png"
        fig.savefig(str(temp_path), dpi=dpi, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        
        image = cv2.imread(str(temp_path))
        temp_path.unlink()
        
        return image
    
    def _generate_video(self, frame_range, video_type: str):
        """Generate video from frames"""
        frames = []
        frame_list = list(frame_range)
        
        print(f"Rendering {video_type} frames...")
        
        dpi = 150  # High DPI for better resolution
        for i, frame_idx in enumerate(frame_list):
            print(f"  Frame {frame_idx} ({i+1}/{len(frame_list)})")
            if video_type == "3d":
                frame_img = self._create_animation_frame(frame_idx, dpi=dpi)
            else:
                frame_img = self._create_heatmap_frame(frame_idx, dpi=dpi)
            frames.append(frame_img)
        
        if not frames:
            print("No frames to create video")
            return
        
        height, width = frames[0].shape[:2]
        print(f"Video dimensions: {width}×{height}")
        
        normalized_frames = []
        for frame in frames:
            if frame.shape[:2] != (height, width):
                frame = cv2.resize(frame, (width, height))
            normalized_frames.append(frame)
        
        fps = 10
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        
        if video_type == "3d":
            output_path = self.output_dir / "reconstruction_3d.mp4"
        else:
            output_path = self.output_dir / "heatmap_2d.mp4"
        
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        
        success_count = 0
        for frame in normalized_frames:
            ret = out.write(frame)
            if ret:
                success_count += 1
        
        out.release()
        print(f"Saved video: {output_path}")
        print(f"Frames written: {success_count}/{len(normalized_frames)}")
    
    def _generate_plotly_3d_animation(self, frame_range):
        """Generate interactive Plotly 3D animation video"""
        print("Generating Plotly 3D animation...")
        
        frame_list = list(frame_range)
        frames_data = []
        
        # Collect mesh data for all frames
        for frame_idx in frame_list:
            print(f"  Processing frame {frame_idx}")
            distances = self._extract_frame_data(frame_idx)
            zone_heights, _, cell_mm = self._compute_zone_heights(distances)
            
            all_x, all_y, all_z = [], [], []
            all_i, all_j, all_k = [], [], []
            colors_intensity = []
            
            heights = np.zeros((8, 8), dtype=np.float32)
            for z in range(64):
                row = z // 8
                col = z % 8
                h = zone_heights.get(z)
                heights[row, col] = h if h is not None and h > 0 else 0.1
            
            for row in range(8):
                for col in range(8):
                    h = heights[row, col]
                    x0, x1 = col * cell_mm, (col + 1) * cell_mm
                    y0, y1 = row * cell_mm, (row + 1) * cell_mm
                    z0, z1 = 0.0, h
                    
                    base = len(all_x)
                    
                    all_x.extend([x0, x1, x1, x0, x0, x1, x1, x0])
                    all_y.extend([y0, y0, y1, y1, y0, y0, y1, y1])
                    all_z.extend([z0, z0, z0, z0, z1, z1, z1, z1])
                    colors_intensity.extend([h] * 8)
                    
                    face_indices = [
                        (0, 1, 2), (0, 2, 3),
                        (4, 5, 6), (4, 6, 7),
                        (0, 1, 5), (0, 5, 4),
                        (2, 3, 7), (2, 7, 6),
                        (0, 3, 7), (0, 7, 4),
                        (1, 2, 6), (1, 6, 5),
                    ]
                    
                    for a, b, c in face_indices:
                        all_i.append(base + a)
                        all_j.append(base + b)
                        all_k.append(base + c)
            
            frames_data.append({
                'x': all_x, 'y': all_y, 'z': all_z,
                'i': all_i, 'j': all_j, 'k': all_k,
                'intensity': colors_intensity,
                'frame_idx': frame_idx
            })
        
        # Create initial frame
        initial = frames_data[0]
        fig = go.Figure(
            data=[go.Mesh3d(
                x=initial['x'], y=initial['y'], z=initial['z'],
                i=initial['i'], j=initial['j'], k=initial['k'],
                intensity=initial['intensity'],
                intensitymode='vertex',
                colorscale='Viridis',
                colorbar=dict(title='Height (mm)'),
                showscale=True,
                flatshading=True,
            )]
        )
        
        # Create frames for animation
        animation_frames = []
        for data in frames_data:
            animation_frames.append(
                go.Frame(
                    data=[go.Mesh3d(
                        x=data['x'], y=data['y'], z=data['z'],
                        i=data['i'], j=data['j'], k=data['k'],
                        intensity=data['intensity'],
                        intensitymode='vertex',
                        colorscale='Viridis',
                        showscale=True,
                        flatshading=True,
                    )],
                    name=str(data['frame_idx'])
                )
            )
        
        fig.frames = animation_frames
        
        # Add play/pause controls
        fig.update_layout(
            title='ToF Sensor 3D Reconstruction - Orbitable Animation',
            scene=dict(
                xaxis_title='X (mm)',
                yaxis_title='Y (mm)',
                zaxis_title='Height (mm)',
                camera=dict(eye=dict(x=1.5, y=-1.5, z=1.2)),
            ),
            updatemenus=[dict(
                type='buttons',
                showactive=False,
                y=0.8,
                x=0.05,
                xanchor='left',
                yanchor='top',
                buttons=[
                    dict(label='Play',
                         method='animate',
                         args=[None, dict(frame=dict(duration=100, redraw=True),
                                        fromcurrent=True)]),
                    dict(label='Pause',
                         method='animate',
                         args=[[None], dict(frame=dict(duration=0, redraw=False),
                                          mode='immediate')])
                ]
            )],
            sliders=[dict(
                active=0,
                yanchor='top',
                y=0,
                xanchor='left',
                x=0.1,
                currentvalue=dict(
                    prefix='Frame: ',
                    visible=True,
                    xanchor='center'
                ),
                transition=dict(duration=50),
                pad=dict(b=10, t=50),
                len=0.8,
                steps=[
                    dict(args=[[f.name], dict(frame=dict(duration=100, redraw=True),
                                             mode='immediate')],
                         method='animate',
                         label=str(f.name))
                    for f in animation_frames
                ]
            )],
            width=1200,
            height=900,
        )
        
        output_path = self.output_dir / "reconstruction_3d_animation.html"
        fig.write_html(output_path, include_plotlyjs='cdn')
        print(f"Saved orbitable 3D animation: {output_path}")
    
    def process(self):
        """Main processing pipeline"""
        
        print("\n" + "="*60)
        print("ToF Sensor Surface Reconstruction Tool")
        print("="*60 + "\n")
        
        print(f"Total frames available: {self.num_frames}")
        print("Select frame range to process:")
        
        while True:
            try:
                start_frame = int(input("Start frame (0-indexed): "))
                end_frame = int(input("End frame (inclusive, 0-indexed): "))
                
                if 0 <= start_frame <= end_frame < self.num_frames:
                    break
                else:
                    print(f"Invalid range. Please enter frames between 0 and {self.num_frames - 1}")
            except ValueError:
                print("Please enter valid integers")
        
        frame_range = range(start_frame, end_frame + 1)
        
        print("\nSelect output types (enter comma-separated numbers):")
        print("1. PLY mesh file(s)")
        print("2. STL mesh file(s) - solid")
        print("3. Orbitable HTML visualization(s) for single frames")
        print("4. 3D reconstruction video (moving through frames)")
        print("5. 2D distance heatmap video")
        print("6. 2D distance heatmap image(s)")
        print("7. Orbitable 3D animation (interactive Plotly)")
        
        while True:
            try:
                output_choice = input("Enter choices (e.g., 1,2,3,4,5,6,7): ").strip()
                choices = [int(x.strip()) for x in output_choice.split(',')]
                if all(c in [1, 2, 3, 4, 5, 6, 7] for c in choices):
                    break
                else:
                    print("Invalid choice. Enter numbers 1-7 separated by commas")
            except ValueError:
                print("Invalid input. Please enter numbers separated by commas")
        
        # Process selections
        if 1 in choices:
            print("\n[1] Generating PLY mesh files...")
            for frame_idx in frame_range:
                distances = self._extract_frame_data(frame_idx)
                zone_heights, _, cell_mm = self._compute_zone_heights(distances)
                vertices, faces = self._create_mesh_from_heights(zone_heights, cell_mm)
                
                output_path = self.output_dir / f"frame_{frame_idx:04d}.ply"
                self._write_ply(vertices, faces, str(output_path))
        
        if 2 in choices:
            print("\n[2] Generating STL mesh files...")
            for frame_idx in frame_range:
                distances = self._extract_frame_data(frame_idx)
                zone_heights, _, cell_mm = self._compute_zone_heights(distances)
                vertices, faces = self._create_mesh_from_heights(zone_heights, cell_mm)
                
                output_path = self.output_dir / f"frame_{frame_idx:04d}.stl"
                self._write_stl(vertices, faces, str(output_path))
        
        if 3 in choices:
            print("\n[3] Generating orbitable HTML visualizations...")
            for frame_idx in frame_range:
                distances = self._extract_frame_data(frame_idx)
                output_path = self.output_dir / f"frame_{frame_idx:04d}.html"
                self._create_html_3d(distances, str(output_path), frame_idx)
        
        if 4 in choices:
            print("\n[4] Generating 3D reconstruction video...")
            self._generate_video(frame_range, "3d")
        
        if 5 in choices:
            print("\n[5] Generating heatmap video...")
            self._generate_video(frame_range, "heatmap")
        
        if 6 in choices:
            print("\n[6] Generating 2D heatmap images...")
            for frame_idx in frame_range:
                distances = self._extract_frame_data(frame_idx)
                output_path = self.output_dir / f"heatmap_{frame_idx:04d}.png"
                self._create_heatmap_2d(distances, str(output_path), frame_idx)
        
        if 7 in choices:
            print("\n[7] Generating orbitable 3D animation...")
            self._generate_plotly_3d_animation(frame_range)
        
        print(f"\nAll outputs saved to: {self.output_dir}")
        print("Processing complete!")


def main():
    """Main entry point"""
    print("ToF Sensor Surface Reconstruction Tool")
    print("-" * 60)
    
    csv_path = input("Enter path to CSV file: ").strip()
    
    if not csv_path:
        print("Please provide a valid file path")
        sys.exit(1)
    
    try:
        processor = ToFSensorProcessor(csv_path)
        processor.process()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
