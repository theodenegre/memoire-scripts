"""Fougère de Barnsley par itération polygonale déterministe (chapitre 5).

Construit l'attracteur via l'opérateur de Hutchinson A_{k+1} = U_i w_i(A_k), en
transportant un polygone initial A_0 (un carré par défaut) par les quatre
transformations affines de Barnsley. Méthode déterministe, conforme au mémoire
(le jeu du chaos n'est pas utilisé ici).


Sorties par défaut :
    figures/ch5_transformations/fern_steps_1_14/fern_n13.png
    figures/ch5_transformations/fern_steps_1_14/fern_n14_colored.png
        -> figure fig:fougère_de_barnsley du mémoire.

Usage :
    python fougere_poly.py                 # régénère la figure du mémoire
    python fougere_poly.py 14 --colored    # une itération précise
"""

import argparse
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
STEPS_DIR = FIGURES_DIR / "fern_steps_1_14"

# Fenêtre fixe pour garantir la même échelle sur toutes les étapes.
PLOT_X_MIN, PLOT_X_MAX = -4.5, 3.0
PLOT_Y_MIN, PLOT_Y_MAX = -0.2, 10.5

# Carré initial A_0.
DEFAULT_A0 = np.array(
    [[-3.0, 9.0], [-3.0, 10.0], [-4.0, 10.0], [-4.0, 9.0]], dtype=float
)
# Palette : tige (rouge), corps (vert), branche droite (bleu), branche gauche (violet).
COLORED_PALETTE = ["#ff4d4d", "#4dff4d", "#4d4dff", "#b33dc6"]


TRANSFORMS = (
    AffineMap(0.0, 0.0, 0.0, 0.16, 0.0, 0.0, name="w1"),
    AffineMap(0.85, 0.04, -0.04, 0.85, 0.0, 1.6, name="w2"),
    AffineMap(0.20, -0.26, 0.23, 0.22, 0.0, 1.6, name="w3"),
    AffineMap(-0.15, 0.28, 0.26, 0.24, 0.0, 0.44, name="w4"),
)


def generate_fern_polygons(iterations, initial_polygon=None):
    initial = DEFAULT_A0 if initial_polygon is None else initial_polygon
    return iterate_polygons([initial], TRANSFORMS, iterations)


def _resolve_colors(polygons, color):
    if not isinstance(color, list):
        return color
    n = len(polygons)
    return (color * (n // len(color) + 1))[:n]


def plot_fern(polygons, output_path, show_vertices=False, color="#1f7a3f", started_at=None):
    t0 = started_at if started_at is not None else time.perf_counter()
    fig, ax = plt.subplots(figsize=(6, 10), dpi=220)
    colors = _resolve_colors(polygons, color)

    if show_vertices:
        points = np.vstack(polygons)
        ax.scatter(points[:, 0], points[:, 1], s=0.2, marker=".",
                   c=colors[0] if isinstance(colors, list) else colors)
    else:
        ax.add_collection(
            PolyCollection(polygons, facecolors=colors, edgecolors=colors, linewidths=0.08, alpha=0.9)
        )

    ax.set_xlim(PLOT_X_MIN, PLOT_X_MAX)
    ax.set_ylim(PLOT_Y_MIN, PLOT_Y_MAX)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")
    fig.tight_layout(pad=0)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    timed_savefig(fig, output_path, t0, pad_inches=0)
    plt.close(fig)


def render_memoire_figure(steps_dir=STEPS_DIR):
    """Régénère fern_n13 (vert) et fern_n14_colored (palette par transformation)."""
    t0 = time.perf_counter()
    plot_fern(generate_fern_polygons(13), steps_dir / "fern_n13.png", color="#1f7a3f", started_at=t0)
    t1 = time.perf_counter()
    plot_fern(generate_fern_polygons(14), steps_dir / "fern_n14_colored.png", color=COLORED_PALETTE, started_at=t1)
    print(f"Generated {steps_dir / 'fern_n13.png'} and fern_n14_colored.png")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("n", type=int, nargs="?", help="Nombre d'itérations (défaut : figure mémoire).")
    parser.add_argument("--vertices", action="store_true")
    parser.add_argument("--colored", action="store_true")
    parser.add_argument("--output", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.n is None:
        render_memoire_figure()
        return

    output = Path(args.output) if args.output else FIGURES_DIR / f"fern_n{args.n}.png"
    t0 = time.perf_counter()
    polygons = generate_fern_polygons(args.n)
    color = COLORED_PALETTE if args.colored else "#1f7a3f"
    plot_fern(polygons, output, show_vertices=args.vertices, color=color, started_at=t0)
    print(f"Image générée : {output.resolve()} ({len(polygons)} polygones)")


if __name__ == "__main__":
    main()
