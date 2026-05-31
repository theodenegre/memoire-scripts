"""IFS simples annotés : carré plein et union de deux intervalles (chapitre 5).

Itération déterministe de l'opérateur de Hutchinson sur des polygones pleins, avec
étiquettes LaTeX montrant l'action de chaque transformation w_i (et des compositions
w_i(w_j(A_0))). Un nuage de points (``*_point.png``) est aussi produit en parallèle
pour chaque étape. Méthode déterministe.

Sorties :
    figures/ch5_transformations/steps/square_{n}_poly.png, square_{n}_point.png
    figures/ch3_ensemble_fractales/steps/two_intervals_{n}_poly.png, *_point.png

Usage :
    python ifs_labels.py
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
from ifs_common.hutchinson import apply_ifs_to_points, apply_ifs_to_polys

FIGURES_DIR_CH5 = Path(__file__).resolve().parents[2] / "figures" / "ch5_transformations"
FIGURES_DIR_CH3 = Path(__file__).resolve().parents[2] / "figures" / "ch3_ensemble_fractales"

_STROKE = [matplotlib.patheffects.withStroke(linewidth=4, foreground="white")]


FRACTALS = {
    "square": {
        "ifs": [
            {"a": 1 / 2, "b": 0, "c": 0, "d": 1 / 2, "e": 0, "f": 0},
            {"a": 1 / 2, "b": 0, "c": 0, "d": 1 / 2, "e": 1 / 2, "f": 0},
            {"a": 1 / 2, "b": 0, "c": 0, "d": 1 / 2, "e": 0, "f": 1 / 2},
            {"a": 1 / 2, "b": 0, "c": 0, "d": 1 / 2, "e": 1 / 2, "f": 1 / 2},
        ],
        "init_poly": [np.array([[0, 0], [1, 0], [1, 1], [0, 1]])],
        "singleton": np.array([[0.5, 0.5]]),
        "colors": [
            "#e74c3c", "#3498db", "#2ecc71", "#f1c40f", "#9b59b6", "#1abc9c", "#e67e22", "#e84393",
            "#d35400", "#27ae60", "#2980b9", "#8e44ad", "#f39c12", "#c0392b", "#16a085", "#2c3e50",
        ],
        "output_dir": "ch5",
    },
    "two_intervals": {
        "ifs": [
            {"a": 1 / 4, "b": 0, "c": 0, "d": 1, "e": 0, "f": 0},
            {"a": 1 / 4, "b": 0, "c": 0, "d": 1, "e": 1 / 4, "f": 0},
            {"a": 1 / 4, "b": 0, "c": 0, "d": 1, "e": 2, "f": 0},
            {"a": 1 / 4, "b": 0, "c": 0, "d": 1, "e": 9 / 4, "f": 0},
        ],
        "init_poly": [
            np.array([[0, 0], [1, 0], [1, 0.2], [0, 0.2]]),
            np.array([[2, 0], [3, 0], [3, 0.2], [2, 0.2]]),
        ],
        "singleton": np.array([[1.5, 0.0]]),
        "colors": ["#e74c3c", "#3498db", "#2ecc71", "#f1c40f"],
        "output_dir": "ch3",
    },
}


def output_dir_for(data):
    base = FIGURES_DIR_CH3 if data.get("output_dir") == "ch3" else FIGURES_DIR_CH5
    return base / "steps"


def plot_points(pts, filename):
    t0 = time.perf_counter()
    n = len(pts)
    size = 15.0 / (n**0.5) if n > 1 else 5.0
    fig, ax = plt.subplots(figsize=(4, 4), dpi=200)
    ax.scatter(pts[:, 0], pts[:, 1], s=size, c="black", marker="o", edgecolors="none", linewidths=0)
    ax.set_aspect("equal")
    ax.axis("off")
    min_xy, max_xy = np.min(pts, axis=0), np.max(pts, axis=0)
    dx = (max_xy[0] - min_xy[0]) * 0.05 or 0.1
    dy = (max_xy[1] - min_xy[1]) * 0.05 or 0.1
    ax.set_xlim(min_xy[0] - dx, max_xy[0] + dx)
    ax.set_ylim(min_xy[1] - dy, max_xy[1] + dy)
    timed_savefig(fig, filename, t0, bbox_inches="tight", pad_inches=0, transparent=True)
    plt.close(fig)


def _facecolors(palette, polys, n):
    if not palette:
        return ["black"] * len(polys)
    if n == 0:
        return ["#95a5a6"]
    return [palette[i % len(palette)] for i in range(len(polys))]


def _annotate(ax, name, ifs, polys, n):
    """Étiquettes w_i(A_0) / w_i(w_j(A_0)) / w_i(A_1) selon l'étape (rendu)."""
    for i, poly in enumerate(polys):
        k = i % len(ifs)
        j = (i // len(ifs)) % len(ifs)
        center = np.mean(poly, axis=0)

        if n == 1:
            label = f"$w_{k+1}(A_0)$"
            y_offset = (0.25 if k % 2 == 0 else -0.3) if name == "two_intervals" else 0
            ax.text(center[0], center[1] + y_offset, label, color="black", fontsize=20,
                    ha="center", va="center", fontweight="bold", path_effects=_STROKE)
        elif n == 2 and name == "square":
            ax.text(center[0], center[1], f"$w_{k+1}(w_{j+1}(A_0))$", color="black", fontsize=13,
                    ha="center", va="center", fontweight="bold", path_effects=_STROKE)
        elif n == 2 and name == "two_intervals" and i < len(ifs):
            quad = np.concatenate([polys[idx] for idx in range(len(polys)) if idx % len(ifs) == k])
            c = np.mean(quad, axis=0)
            y_offset = 0.5 if k % 2 == 0 else -0.6
            ax.text(c[0], c[1] + y_offset, f"$w_{k+1}(A_1)$", color="black", fontsize=20,
                    ha="center", va="center", fontweight="bold", path_effects=_STROKE)


def render_fractal(name, data, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    ifs = data["ifs"]
    polys = data["init_poly"]
    pts = data["singleton"]
    palette = data.get("colors")
    is_wide = name == "two_intervals"

    print(f"Traitement de {name}...")
    for n in range(3):
        t0 = time.perf_counter()
        fig, ax = plt.subplots(figsize=(10, 4) if is_wide else (8, 8), dpi=200)
        fcs = _facecolors(palette, polys, n)
        for i, poly in enumerate(polys):
            ax.add_patch(Polygon(poly, facecolor=fcs[i % len(fcs)], edgecolor="black", linewidth=1.5))
        if n > 0:
            _annotate(ax, name, ifs, polys, n)

        ax.set_aspect("equal")
        ax.axis("off")
        all_pts = np.concatenate(polys)
        min_xy, max_xy = np.min(all_pts, axis=0), np.max(all_pts, axis=0)
        dx = (max_xy[0] - min_xy[0]) * 0.1
        dy = (max_xy[1] - min_xy[1]) * (0.8 if is_wide else 0.1)
        ax.set_xlim(min_xy[0] - dx, max_xy[0] + dx)
        ax.set_ylim(min_xy[1] - dy, max_xy[1] + dy)
        timed_savefig(fig, output_dir / f"{name}_{n}_poly.png", t0, bbox_inches="tight", pad_inches=0.1, transparent=True)
        plt.close(fig)

        plot_points(pts, output_dir / f"{name}_{n}_point.png")

        polys = apply_ifs_to_polys(ifs, polys)
        pts = apply_ifs_to_points(pts, ifs)


def main():
    for name, data in FRACTALS.items():
        render_fractal(name, data, output_dir_for(data))
    print("Termine.")


if __name__ == "__main__":
    main()
