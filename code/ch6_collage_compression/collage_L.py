"""Théorème du collage sur une forme en L (chapitre 6).

Illustre le collage parfait et imparfait. Le compact cible K (forme en L de 3 carrés
unitaires) est l'union de 4 copies de lui-même contractées de 1/2 par les
transformations w_i (homothéties, symétries, translations) données dans le mémoire.

- Collage parfait   : W(K) = K, erreur de collage h(K, W(K)) = 0.
- Collage imparfait : transformations perturbées -> vides/recouvrements, erreur > 0.

Sorties : figures/ch6_collage_compression/collage_parfait.png
          figures/ch6_collage_compression/collage_imparfait.png

Usage :
    python collage_L.py
"""

import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from figmeta import timed_savefig

OUT_DIR = Path(__file__).resolve().parents[2] / "figures" / "ch6_collage_compression"

L_SHAPE = np.array([[0.0, 0.0], [2.0, 0.0], [2.0, 1.0], [1.0, 1.0], [1.0, 2.0], [0.0, 2.0]])
COLORS = ["#ff7675", "#74b9ff", "#55efc4", "#fdcb6e"]
EDGE_COLORS = ["#d63031", "#0984e3", "#00b894", "#d35400"]

# (matrice 2x2, translation) des 4 transformations du collage parfait.
PERFECT = [
    ([[0.5, 0.0], [0.0, 0.5]], [0.0, 0.0]),
    ([[-0.5, 0.0], [0.0, 0.5]], [2.0, 0.0]),
    ([[0.5, 0.0], [0.0, -0.5]], [0.0, 2.0]),
    ([[0.5, 0.0], [0.0, 0.5]], [0.5, 0.5]),
]
# Variante imparfaite : w1, w2, w3 perturbées, w4 correcte.
IMPERFECT = [
    ([[0.48, 0.0], [0.0, 0.48]], [0.05, 0.05]),
    ([[-0.52, 0.0], [0.0, 0.52]], [2.05, -0.05]),
    ([[0.45, 0.0], [0.0, -0.45]], [-0.05, 2.05]),
    ([[0.5, 0.0], [0.0, 0.5]], [0.5, 0.5]),
]


def transform_polygon(vertices, matrix, translation):
    return vertices @ np.array(matrix).T + np.array(translation)


def _draw(transforms, title, label_prefix, output_path):
    t0 = time.perf_counter()
    fig, ax = plt.subplots(figsize=(5, 5), dpi=200)
    ax.add_patch(
        patches.Polygon(L_SHAPE, closed=True, facecolor="#f5f6fa", edgecolor="#7f8c8d",
                        linestyle="--", linewidth=1.5, zorder=1)
    )
    for i, (matrix, translation) in enumerate(transforms):
        ax.add_patch(
            patches.Polygon(
                transform_polygon(L_SHAPE, matrix, translation), closed=True,
                facecolor=COLORS[i], edgecolor=EDGE_COLORS[i], alpha=0.8, linewidth=1.5,
                label=rf"${label_prefix}_{i + 1}(K)$", zorder=2,
            )
        )
    ax.set_xlim(-0.2, 2.2)
    ax.set_ylim(-0.2, 2.2)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.legend(loc="upper right", fontsize=12, frameon=True, facecolor="white", edgecolor="#dfe6e9")
    ax.set_title(title, fontsize=12, pad=10, color="#2d3436")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    timed_savefig(fig, output_path, t0, bbox_inches="tight")
    plt.close(fig)
    print(f"Generated {output_path.name}")


def main(out_dir=OUT_DIR):
    _draw(PERFECT, "Collage parfait : $W(K) = K$\nErreur de collage $h(K, W(K)) = 0$",
          "w", out_dir / "collage_parfait.png")
    _draw(IMPERFECT, "Collage imparfait : $W(K) \\approx K$\nErreur de collage $h(K, W(K)) > 0$",
          "w'", out_dir / "collage_imparfait.png")


if __name__ == "__main__":
    main()
