"""Comparaison fougère de Barnsley avec / sans condensation (chapitre 5).

Construit l'attracteur du SFI complet de Barnsley (tige w1 incluse) à partir d'un
A_0 carré, en deux colonnes : sans condensation, et avec un ensemble de
condensation C (fin segment vertical = la tige) réinjecté à chaque itération.
On affiche l'étape 1 et l'étape n. Chaque polygone est coloré par la dernière
transformation appliquée. Méthode déterministe.

Note d'implémentation : pour rester en mémoire, les polygones devenus plus petits
que ``PRUNE_LIMIT`` ne sont plus subdivisés (ils sont figés et dessinés tels quels).
C'est une optimisation de rendu, sans incidence visible sur l'attracteur final.
Elle ne fait pas partie de la définition mathématique de la condensation.

Sortie : figures/ch5_transformations/fern_condensation_compare_square.png

Usage :
    python fougere_condensation.py [-n 18]
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

DEFAULT_OUTPUT = (
    Path(__file__).resolve().parents[2]
    / "figures"
    / "ch5_transformations"
    / "fern_condensation_compare_square.png"
)

PLOT_X_MIN, PLOT_X_MAX = -5.0, 3.5
PLOT_Y_MIN, PLOT_Y_MAX = -0.5, 11.0
PRUNE_LIMIT = 0.008

A0_SQUARE = np.array([[-3.0, 9.0], [-3.0, 10.0], [-4.0, 10.0], [-4.0, 9.0]], dtype=float)
STEM_CONDENSATION = np.array([[-0.02, 0.0], [0.02, 0.0], [0.02, 1.6], [-0.02, 1.6]], dtype=float)

COLOR_BY_TRANSFORM = {
    "w0": "#8b4513",  # condensation
    "w1": "#c85a1a",  # tige
    "w2": "#4dff4d",
    "w3": "#4d4dff",
    "w4": "#b33dc6",
}
INITIAL_COLOR = "#888888"


TRANSFORMS = (
    AffineMap(0.0, 0.0, 0.0, 0.16, 0.0, 0.0, name="w1"),
    AffineMap(0.85, 0.04, -0.04, 0.85, 0.0, 1.6, name="w2"),
    AffineMap(0.20, -0.26, 0.23, 0.22, 0.0, 1.6, name="w3"),
    AffineMap(-0.15, 0.28, 0.26, 0.24, 0.0, 0.44, name="w4"),
)


def attractor_at_step(step, initial_polygon, condensation):
    """Polygones (+ couleurs) de l'attracteur après ``step`` itérations.

    Les petits polygones (taille < PRUNE_LIMIT) sont retirés de la subdivision et
    conservés dans une réserve pour le rendu final.
    """
    polys = initial_polygon[np.newaxis, ...]
    colors = np.array([INITIAL_COLOR], dtype=object)
    frozen_polys, frozen_colors = [], []

    for _ in range(step):
        if polys.shape[0] == 0:
            break
        count = polys.shape[0]
        mapped_polys = np.concatenate([m.apply(polys) for m in TRANSFORMS], axis=0)
        mapped_colors = np.concatenate(
            [np.full(count, COLOR_BY_TRANSFORM[m.name], dtype=object) for m in TRANSFORMS]
        )

        sizes = np.max(np.max(mapped_polys, axis=1) - np.min(mapped_polys, axis=1), axis=1)
        active = sizes >= PRUNE_LIMIT
        polys, colors = mapped_polys[active], mapped_colors[active]
        frozen_polys.append(mapped_polys[~active])
        frozen_colors.append(mapped_colors[~active])

        if condensation is not None:
            polys = np.concatenate([polys, condensation[np.newaxis, ...]], axis=0)
            colors = np.concatenate([colors, np.array([COLOR_BY_TRANSFORM["w0"]], dtype=object)])

    all_polys = [polys] + [p for p in frozen_polys if p.shape[0] > 0]
    all_colors = [colors] + [c for c in frozen_colors if c.shape[0] > 0]
    return np.concatenate(all_polys, axis=0), np.concatenate(all_colors, axis=0)


def draw_panel(axis, polys, colors, caption):
    axis.add_collection(PolyCollection(polys, facecolors=colors, edgecolors=colors, linewidths=1.0, alpha=0.9))
    axis.set_xlim(PLOT_X_MIN, PLOT_X_MAX)
    axis.set_ylim(PLOT_Y_MIN, PLOT_Y_MAX)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_visible(False)
    axis.set_xlabel(caption, fontsize=10, labelpad=6)
    axis.xaxis.set_label_position("bottom")


def build_legend(figure):
    labels = [
        ("w0", "$w_0$ (condensation)"), ("w1", "$w_1$ (tige)"),
        ("w2", "$w_2$"), ("w3", "$w_3$"), ("w4", "$w_4$"),
    ]
    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=COLOR_BY_TRANSFORM[key], edgecolor="none", label=label)
        for key, label in labels
    ]
    handles.append(
        plt.Rectangle((0, 0), 1, 1, facecolor=INITIAL_COLOR, edgecolor="none", label="$A_0$ (aucune transformation)")
    )
    figure.legend(handles=handles, loc="lower center", ncol=6, fontsize=9, frameon=False,
                  bbox_to_anchor=(0.5, 0.0))


def make_figure(a0, n, output_path):
    t0 = time.perf_counter()
    columns = (("Sans condensation", None), ("Avec condensation $C$", STEM_CONDENSATION))
    figure, axes = plt.subplots(2, 2, figsize=(10, 11), dpi=200)

    for col, (col_label, condensation) in enumerate(columns):
        for row, step in enumerate((1, n)):
            polys, colors = attractor_at_step(step, a0, condensation)
            draw_panel(axes[row, col], polys, colors, caption=f"{col_label}\nÉtape {step}")

    figure.suptitle(f"Construction de l'attracteur (tige incluse), $A_0$ carré, itérations 1 et {n}", fontsize=12)
    build_legend(figure)
    figure.tight_layout(pad=0.5, rect=(0, 0.05, 1, 0.96))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    timed_savefig(figure, output_path, t0, pad_inches=0.05, bbox_inches="tight")
    plt.close(figure)
    print(f"Image générée : {output_path.resolve()} (n={n})")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("-n", "--iterations", type=int, default=18)
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()
    make_figure(A0_SQUARE, args.iterations, Path(args.output))


if __name__ == "__main__":
    main()
