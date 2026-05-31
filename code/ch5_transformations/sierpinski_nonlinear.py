"""Triangle de Sierpinski non lineaire par iteration deterministe (chapitre 5).

SFI non lineaire sur le triangle droit T = {(x,y) | x >= 0, y >= 0, x + y <= 1},
puis affichage sur un triangle equilateral par une transformation affine.

Transformations sur T (d = 2) :
    w1(x,y) = (x^2/d, y^2/d)
    w2(x,y) = (1 - (1-x)^2/d, y^2/d)
    w3(x,y) = (x^2/d, 1 - (1-y)^2/d)

Sortie : figures/ch5_transformations/sierpinski_nonlinear_n{n}.png

Usage :
    python sierpinski_nonlinear.py
"""

import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from figmeta import timed_savefig

OUTPUT_DIR = Path(__file__).resolve().parents[2] / "figures" / "ch5_transformations"
DEFAULT_N = 10
COLORS = ["#e74c3c", "#3498db", "#2ecc71"]
D_VALUE = 2.0
SIDE_LENGTH = 1.0
HEIGHT = np.sqrt(3) * SIDE_LENGTH / 2.0


def w1(x, y, d=D_VALUE):
    return x * x / d, y * y / d


def w2(x, y, d=D_VALUE):
    return 1.0 - (1.0 - x) * (1.0 - x) / d, y * y / d


def w3(x, y, d=D_VALUE):
    return x * x / d, 1.0 - (1.0 - y) * (1.0 - y) / d


TRANSFORMS = (w1, w2, w3)


def generate_points(iterations, start=(SIDE_LENGTH / 3.0, SIDE_LENGTH / 3.0)):
    points = [(start[0], start[1], -1)]
    for _ in range(iterations):
        new_points = []
        for x, y, _ in points:
            for idx, transform in enumerate(TRANSFORMS):
                nx, ny = transform(x, y)
                new_points.append((nx, ny, idx))
        points = new_points
    return points


def _marker_size(num_points):
    if num_points < 100:
        return 14
    if num_points < 1000:
        return 6
    if num_points < 10000:
        return 2.5
    if num_points < 30000:
        return 1.0
    return 0.6


def _to_equilateral(points):
    data = np.array(points)
    xs = data[:, 0]
    ys = data[:, 1]
    mapped_x = xs + ys / 2.0
    mapped_y = (np.sqrt(3) / 2.0) * ys
    return np.column_stack((mapped_x, mapped_y))


def plot_points(points_data, output_path, started_at=None):
    t0 = started_at if started_at is not None else time.perf_counter()
    fig, ax = plt.subplots(figsize=(6, 6), dpi=220)
    size = _marker_size(len(points_data))

    for i in range(len(TRANSFORMS)):
        group = [(x, y) for x, y, idx in points_data if idx == i or (idx == -1 and i == 0)]
        if not group:
            continue
        data = _to_equilateral(group)
        ax.plot(data[:, 0], data[:, 1], marker="o", color=COLORS[i], markersize=size, linestyle="None")

    ax.set_xlim(0.0, SIDE_LENGTH)
    ax.set_ylim(0.0, HEIGHT)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")
    fig.tight_layout(pad=0)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    timed_savefig(fig, output_path, t0, pad_inches=0)
    plt.close(fig)


def main(n=DEFAULT_N, output_dir=OUTPUT_DIR):
    output_dir.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    points = generate_points(n)
    out_path = output_dir / f"sierpinski_nonlinear_n{n}.png"
    plot_points(points, out_path, started_at=start)
    print(f"  {out_path.name} ({len(points)} pts, {time.perf_counter() - start:.2f}s)")


if __name__ == "__main__":
    main()
