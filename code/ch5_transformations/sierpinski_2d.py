"""Triangle de Sierpinski par itération polygonale déterministe (chapitre 5).

Itère l'opérateur de Hutchinson A_{k+1} = U_i w_i(A_k) avec les trois homothéties
de rapport 1/2 centrées aux sommets d'un triangle équilatéral. Deux ensembles
initiaux sont illustrés : le triangle lui-même et un A_0 arbitraire (forme + cercle)
pour montrer la convergence indépendante de A_0. Méthode déterministe.

Sorties : figures/ch5_transformations/sierpinski_tri_n{n}.png
          figures/ch5_transformations/sierpinski_alt_n{n}.png

Usage :
    python sierpinski_2d.py
"""

import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PolyCollection

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from figmeta import timed_savefig
from ifs_common.affine import AffineMap
from ifs_common.hutchinson import iterate_polygons

FIGURES_DIR = Path(__file__).resolve().parents[2] / "figures" / "ch5_transformations"
DEFAULT_STEPS = [0, 1, 2, 8]

PLOT_X_MIN, PLOT_X_MAX = -0.5, 1.1
PLOT_Y_MIN, PLOT_Y_MAX = -0.1, 1.1


TRANSFORMS = (
    AffineMap(0.5, 0.0, 0.0, 0.5, 0.0, 0.0),
    AffineMap(0.5, 0.0, 0.0, 0.5, 0.5, 0.0),
    AffineMap(0.5, 0.0, 0.0, 0.5, 0.25, np.sqrt(3) / 4),
)

TRIANGLE_A0 = [np.array([[0.0, 0.0], [1.0, 0.0], [0.5, np.sqrt(3) / 2]], dtype=float)]

_theta = np.linspace(0, 2 * np.pi, 100)
ALT_A0 = [
    np.array(
        [[-0.4, 0.2], [-0.2, 0.1], [-0.05, 0.25], [-0.1, 0.6], [-0.2, 0.75], [-0.35, 0.65], [-0.45, 0.4]],
        dtype=float,
    ),
    np.column_stack((0.1 + 0.1 * np.cos(_theta), 1 + 0.1 * np.sin(_theta))),
]


def generate_sierpinski_polygons(iterations, initial_polygons):
    return iterate_polygons(initial_polygons, TRANSFORMS, iterations)


def _resolve_colors(polygons, color):
    if not isinstance(color, list):
        return color
    per_color = max(1, len(polygons) // len(color))
    resolved = []
    for c in color:
        resolved.extend([c] * per_color)
    return (resolved * (len(polygons) // max(1, len(resolved)) + 1))[: len(polygons)]


def plot_sierpinski(polygons, output_path, color="#1f77b4", started_at=None):
    t0 = started_at if started_at is not None else time.perf_counter()
    fig, ax = plt.subplots(figsize=(6, 5), dpi=220)
    colors = _resolve_colors(polygons, color)
    ax.add_collection(
        PolyCollection(polygons, facecolors=colors, edgecolors=colors, linewidths=0.2, alpha=0.9)
    )
    ax.set_xlim(PLOT_X_MIN, PLOT_X_MAX)
    ax.set_ylim(PLOT_Y_MIN, PLOT_Y_MAX)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")
    fig.tight_layout(pad=0)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    timed_savefig(fig, output_path, t0, pad_inches=0)
    plt.close(fig)


def render_series(initial_polygons, prefix, color, steps=DEFAULT_STEPS, output_dir=FIGURES_DIR):
    for n in steps:
        t0 = time.perf_counter()
        polys = generate_sierpinski_polygons(n, initial_polygons)
        out = output_dir / f"{prefix}_n{n}.png"
        plot_sierpinski(polys, out, color=color, started_at=t0)
        print(f"Generated {out}")


def main(output_dir=FIGURES_DIR):
    render_series(TRIANGLE_A0, "sierpinski_tri", color="#ff0000", output_dir=output_dir)
    render_series(ALT_A0, "sierpinski_alt", color=["#66cc66", "#003399"], output_dir=output_dir)


if __name__ == "__main__":
    main()
