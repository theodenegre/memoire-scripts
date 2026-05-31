"""Fractales classiques : étapes déterministes polygone + nuage de points (chapitre 5).

Pour Cantor, Koch, triangle et tapis de Sierpinski, et arbre fractal : on itère l'opérateur de Hutchinson de manière déterministe.
Les trois premières étapes sont rendues comme polygones pleins (avec étiquettes w_i à l'étape 1), et l'état quasi-final (n=8) comme nuage de points.
Méthode déterministe (pas de jeu du chaos).

Sortie : figures/ch5_transformations/steps/{name}_{n}_poly.png (n=0,1,2)
         figures/ch5_transformations/steps/{name}_{n}_point.png (n=0,8)

Usage :
    python fractales_etapes.py
"""

import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects
import numpy as np
from matplotlib.patches import Polygon

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from figmeta import timed_savefig
from ifs_common.affine import affine_dict as _affine
from ifs_common.hutchinson import apply_ifs_to_points, apply_ifs_to_polys

OUTPUT_DIR = Path(__file__).resolve().parents[2] / "figures" / "ch5_transformations" / "steps"
_STROKE = [matplotlib.patheffects.withStroke(linewidth=3, foreground="white")]

_sqrt3 = np.sqrt(3)
_c, _s = np.cos(np.pi / 6), np.sin(np.pi / 6)
_tree_ratio, _trunk_scale, _trunk_thickness = 0.5, 0.5, 0.05


def _sierpinski_carpet_ifs():
    return [
        _affine(1 / 3, 0, 0, 1 / 3, i / 3, j / 3)
        for i in range(3)
        for j in range(3)
        if not (i == 1 and j == 1)
    ]


FRACTALS = {
    "cantor": {
        "ifs": [_affine(1 / 3, 0, 0, 1.0, 0, 0), _affine(1 / 3, 0, 0, 1.0, 2 / 3, 0)],
        "init_poly": [np.array([[0, 0], [1, 0], [1, 0.05], [0, 0.05]])],
        "singleton": np.array([[0.5, 0.0]]),
    },
    "koch": {
        "ifs": [
            _affine(1 / 3, 0, 0, 1 / 3, 0, 0),
            _affine(1 / 6, -_sqrt3 / 6, _sqrt3 / 6, 1 / 6, 1 / 3, 0),
            _affine(1 / 6, _sqrt3 / 6, -_sqrt3 / 6, 1 / 6, 1 / 2, _sqrt3 / 6),
            _affine(1 / 3, 0, 0, 1 / 3, 2 / 3, 0),
        ],
        "init_poly": [np.array([[0, 0], [1, 0], [1, 0.05], [0, 0.05]])],
        "singleton": np.array([[0.5, 0.0]]),
    },
    "sierpinski_tri": {
        "ifs": [
            _affine(1 / 2, 0, 0, 1 / 2, 0, 0),
            _affine(1 / 2, 0, 0, 1 / 2, 1 / 2, 0),
            _affine(1 / 2, 0, 0, 1 / 2, 1 / 4, _sqrt3 / 4),
        ],
        "init_poly": [np.array([[0, 0], [1, 0], [0.5, _sqrt3 / 2]])],
        "singleton": np.array([[0.5, 0.0]]),
    },
    "sierpinski_carpet": {
        "ifs": _sierpinski_carpet_ifs(),
        "init_poly": [np.array([[0, 0], [1, 0], [1, 1], [0, 1]])],
        "singleton": np.array([[0.5, 0.5]]),
    },
    "tree": {
        "ifs": [
            _affine(0.0, 0, 0, _trunk_scale, 0, 0),
            _affine(_tree_ratio * _c, -_tree_ratio * _s, _tree_ratio * _s, _tree_ratio * _c, 0, _trunk_scale),
            _affine(_tree_ratio * _c, _tree_ratio * _s, -_tree_ratio * _s, _tree_ratio * _c, 0, _trunk_scale),
        ],
        "init_poly": [
            np.array([[-_trunk_thickness, 0], [_trunk_thickness, 0], [_trunk_thickness, 1], [-_trunk_thickness, 1]])
        ],
        "singleton": np.array([[0.0, 0.0]]),
    },
}
LABELLED = {"cantor", "koch", "sierpinski_tri", "tree"}


def plot_points(pts, filename):
    t0 = time.perf_counter()
    n = len(pts)
    size = 15.0 / (n**0.5) if n > 1 else 5.0
    fig, ax = plt.subplots(figsize=(4, 4), dpi=200)
    ax.scatter(pts[:, 0], pts[:, 1], s=size, c="black", marker="o", edgecolors="none", linewidths=0)
    ax.set_aspect("equal")
    ax.axis("off")
    if "tree" in str(filename):
        ax.set_xlim(-0.6, 0.6)
        ax.set_ylim(-0.05, 1.05)
    else:
        min_xy, max_xy = np.min(pts, axis=0), np.max(pts, axis=0)
        dx = (max_xy[0] - min_xy[0]) * 0.1 or 0.1
        dy = (max_xy[1] - min_xy[1]) * 0.1 or 0.1
        ax.set_xlim(min_xy[0] - dx, max_xy[0] + dx)
        ax.set_ylim(min_xy[1] - dy, max_xy[1] + dy)
    timed_savefig(fig, filename, t0, bbox_inches="tight", pad_inches=0.1, transparent=True)
    plt.close(fig)


def plot_polys(name, ifs, polys, palette, n, filename):
    t0 = time.perf_counter()
    fig, ax = plt.subplots(figsize=(5, 5), dpi=200)
    if palette and n > 0:
        fcs = (palette * (len(polys) // len(palette) + 1))[: len(polys)]
    elif palette:
        fcs = ["#95a5a6"]
    else:
        fcs = ["black"] * len(polys)

    for i, poly in enumerate(polys):
        ax.add_patch(Polygon(poly, facecolor=fcs[i % len(fcs)], edgecolor="black", linewidth=0.5))
        if n == 1 and name in LABELLED:
            k = i % len(ifs)
            center = np.mean(poly, axis=0)
            y_off = 0.1 if (name == "cantor" or (name == "koch" and k in (1, 2))) else 0
            ax.text(center[0], center[1] + y_off, f"$w_{k+1}$", color="black", fontsize=14,
                    ha="center", va="center", fontweight="bold", path_effects=_STROKE)

    ax.set_aspect("equal")
    ax.axis("off")
    if name == "tree":
        ax.set_xlim(-0.6, 0.6)
        ax.set_ylim(-0.05, 1.05)
    else:
        all_pts = np.concatenate(polys)
        min_xy, max_xy = np.min(all_pts, axis=0), np.max(all_pts, axis=0)
        dx = (max_xy[0] - min_xy[0]) * 0.15
        dy = (max_xy[1] - min_xy[1]) * 0.15
        ax.set_xlim(min_xy[0] - dx, max_xy[0] + dx)
        ax.set_ylim(min_xy[1] - dy, max_xy[1] + dy)
    timed_savefig(fig, filename, t0, bbox_inches="tight", pad_inches=0.1, transparent=True)
    plt.close(fig)


PALETTE = ["#e74c3c", "#3498db", "#2ecc71", "#f1c40f", "#9b59b6", "#1abc9c", "#e67e22", "#34495e"]


def render_fractal(name, data, output_dir):
    ifs = data["ifs"]
    polys = data["init_poly"]
    pts = data["singleton"]

    print(f"Traitement de {name}...")
    for n in range(9):
        if n <= 2:
            plot_polys(name, ifs, polys, PALETTE, n, output_dir / f"{name}_{n}_poly.png")
        if n in (0, 8):
            plot_points(pts, output_dir / f"{name}_{n}_point.png")
        if n < 8:
            if n < 2:
                polys = apply_ifs_to_polys(ifs, polys)
            pts = apply_ifs_to_points(pts, ifs)


def main(output_dir=OUTPUT_DIR):
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, data in FRACTALS.items():
        render_fractal(name, data, output_dir)
    print("Termine.")


if __name__ == "__main__":
    main()
