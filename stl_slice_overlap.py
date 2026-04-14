import argparse
import os
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt

def _timestamp() -> str:
    return datetime.now().strftime('%Y%m%d_%H%M%S')

def _ensure_out_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path

def _choose_axis_interactive(default: str = 'z') -> str:
    axes = ['x', 'y', 'z']
    while True:
        print('Select slicing axis:')
        for i, a in enumerate(axes, start=1):
            print(f'  {i}) {a}')
        raw = input(f'Enter choice [1-3] (default {default}): ').strip()
        if raw == '':
            return default
        if raw.isdigit() and 1 <= int(raw) <= 3:
            return axes[int(raw) - 1]
        if raw.lower() in axes:
            return raw.lower()
        print('Invalid choice, try again.\n')

def _pick_file_interactive(prompt: str, default: str) -> str:
    raw = input(f'{prompt} (default: {default}): ').strip()
    return raw if raw else default

def _pick_out_dir_interactive(default: str = 'outputs') -> str:
    raw = input(f'Output directory (default: {default}): ').strip()
    return raw if raw else default

def main():
    parser = argparse.ArgumentParser(
        description='Slice two STL files and compute/plot overlap areas.'
    )

    parser.add_argument('--experiment-stl', type=str, default=None, help='Path to experiment STL file')
    parser.add_argument('--true-stl', type=str, default=None, help='Path to true STL file')
    parser.add_argument('--out-dir', type=str, default=None, help='Output directory')

    parser.add_argument('--N', type=int, default=1, help='Number of experimental slices')
    parser.add_argument('--axis', choices=['x', 'y', 'z'], default=None, help='Slicing axis')
    parser.add_argument('--csv', action='store_true', help='Write CSV output')
    parser.add_argument('--out', type=str, default='overlap_summary', help='Basename for output files (no extension)')

    parser.add_argument('--interactive', action='store_true', help='Prompt for STL paths and axis in the terminal')

    args = parser.parse_args()

    # Interactive selection (requested)
    if args.interactive:
        exp_path = _pick_file_interactive('Experiment STL path', args.experiment_stl or 'experimentBody.stl')
        true_path = _pick_file_interactive('True STL path', args.true_stl or 'trueBody.stl')
        axis = _choose_axis_interactive(default=(args.axis or 'z'))
        out_dir = _pick_out_dir_interactive(default=(args.out_dir or 'outputs'))
    else:
        exp_path = args.experiment_stl or 'experimentBody.stl'
        true_path = args.true_stl or 'trueBody.stl'
        axis = args.axis or 'z'
        out_dir = args.out_dir or '.'.

    out_dir = _ensure_out_dir(out_dir)

    # For now, we just "display" what was selected and produce a timestamped placeholder plot.
    # (This keeps the interactive UI working while overlap computation is developed/refined.)
    print('\nSelected inputs:')
    print(f'  experiment STL: {exp_path}')
    print(f'  true STL      : {true_path}')
    print(f'  axis          : {axis}')
    print(f'  N slices      : {args.N}')
    print(f'  out dir       : {out_dir}')

    # Timestamped output names
    ts = _timestamp()
    img_path = os.path.join(out_dir, f'{args.out}_{ts}.png')
    csv_path = os.path.join(out_dir, f'{args.out}_{ts}.csv')

    # Placeholder plot
    y = np.zeros(args.N, dtype=float)
    plt.figure()
    plt.plot(y)
    plt.xlabel('Slice Index')
    plt.ylabel('Overlap Area (placeholder)')
    plt.title('Overlap Areas (placeholder)')
    plt.tight_layout()
    plt.savefig(img_path, dpi=200)
    plt.close()
    print(f'Wrote image: {img_path}')

    if args.csv:
        import pandas as pd
        df = pd.DataFrame({'slice_index': list(range(args.N)), 'overlap_area': y})
        df.to_csv(csv_path, index=False)
        print(f'Wrote CSV: {csv_path}')


if __name__ == '__main__':
    main()