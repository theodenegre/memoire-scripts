"""Les 8 isométries du groupe diédral D4 (chapitre 6).

Affiche les 8 transformations (identité, rotations 90/180/270, réflexions H/V et
sur les deux diagonales) appliquées à une lettre F dans une grille 8x8. Ces
isométries sont la composante spatiale des transformations PIFS.

Sortie : figures/ch6_collage_compression/symmetries_d4.png

Usage :
    python isometries_d4.py
"""

import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from figmeta import timed_savefig
from pifs_core import get_symmetries_list

DEFAULT_OUTPUT = (
    Path(__file__).resolve().parents[2] / "figures" / "ch6_collage_compression" / "symmetries_d4.png"
)


def make_letter_f(size=8):
    b = np.zeros((size, size), dtype=np.uint8)
    b[1:7, 2:3] = 255  # barre verticale
    b[1:2, 2:6] = 255  # barre supérieure
    b[3:4, 2:5] = 255  # barre du milieu
    return b


# Mêmes 8 isométries que ``pifs_core.get_symmetries_list`` (même ordre), enrichies
# d'un libellé pour l'affichage.
D4_LABELS = [
    "Identité (sans changement)",
    "Rotation de 90° (antihoraire)",
    "Rotation de 180°",
    "Rotation de 270°",
    "Réflexion horizontale (gauche/droite)",
    "Réflexion verticale (haut/bas)",
    "Transposition diagonale principale",
    "Transposition diagonale secondaire",
]


def d4_symmetries(block):
    return list(zip(D4_LABELS, get_symmetries_list(block)))


def main(output_path=DEFAULT_OUTPUT):
    t0 = time.perf_counter()
    symmetries = d4_symmetries(make_letter_f())

    fig, axes = plt.subplots(2, 4, figsize=(12, 6.5), dpi=300)
    plt.subplots_adjust(wspace=0.3, hspace=0.4)

    for idx, (desc, img) in enumerate(symmetries):
        ax = axes[idx // 4, idx % 4]
        ax.imshow(img, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
        for i in range(9):
            ax.axhline(i - 0.5, color="#7f8c8d", linewidth=0.5, alpha=0.5)
            ax.axvline(i - 0.5, color="#7f8c8d", linewidth=0.5, alpha=0.5)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"$\\iota_{idx}$\n{desc}", fontsize=8.5, fontweight="bold", color="#2c3e50", pad=8)
        for spine in ax.spines.values():
            spine.set_edgecolor("#34495e")
            spine.set_linewidth(1.2)

    plt.suptitle("Les 8 transformations isométriques du groupe diédral $D_4$ appliquées à un bloc 8x8",
                 fontsize=12, fontweight="bold", color="#2c3e50", y=0.98)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    timed_savefig(plt, output_path, t0, bbox_inches="tight", pad_inches=0.1)
    plt.close()
    print(f"Figure des symétries D4 sauvegardée : {output_path}")


if __name__ == "__main__":
    main()
