"""Éponge de Menger par itération déterministe dans R^3 (chapitre 5).

SFI de 20 homothéties de rapport 1/3 : on part d'un cube et on conserve, parmi les
27 sous-cubes (3x3x3), les 20 qui ne sont ni le centre ni les centres de faces
(c.-à-d. au plus une coordonnée nulle). On itère l'opérateur de Hutchinson sur les
cubes (méthode déterministe), puis on rend les faces colorées par hauteur.

Sortie : figures/ch5_transformations/steps/menger_sponge.png

Usage :
    python menger.py
"""

import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from figmeta import timed_savefig

OUTPUT_DIR = Path(__file__).resolve().parents[2] / "figures" / "ch5_transformations" / "steps"

CUBE_FACES = [[0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 5, 4], [2, 3, 7, 6], [0, 3, 7, 4], [1, 2, 6, 5]]
UNIT_CUBE = np.array(
    [[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
     [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]],
    dtype=float,
)


def menger_ifs():
    ifs = []
    for x in (-1, 0, 1):
        for y in (-1, 0, 1):
            for z in (-1, 0, 1):
                zeros = (x == 0) + (y == 0) + (z == 0)
                if zeros <= 1:
                    ifs.append({"M": (1 / 3) * np.eye(3), "b": (2 / 3) * np.array([x, y, z])})
    return ifs


def iterate_3d(ifs, initial_solids, depth):
    solids = list(initial_solids)
    for d in range(depth):
        print(f"  Iteration {d + 1}/{depth}...")
        solids = [(t["M"] @ solid.T).T + t["b"] for solid in solids for t in ifs]
    return solids


def plot_cubes(cubes, filename, title, faces=CUBE_FACES, lims=(-1, 1), started_at=None):
    t0 = started_at if started_at is not None else time.perf_counter()
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="3d")

    z_means = [np.mean(c[:, 2]) for c in cubes]
    z_min, z_max = min(z_means), max(z_means)
    z_range = z_max - z_min or 1.0
    cmap = plt.cm.viridis

    all_faces, all_colors = [], []
    for cube, z in zip(cubes, z_means):
        color = cmap((z - z_min) / z_range)
        for face in faces:
            all_faces.append(cube[face])
            all_colors.append(color)

    print(f"Rendu de {len(all_faces)} faces...")
    ax.add_collection3d(
        Poly3DCollection(all_faces, facecolors=all_colors, edgecolors="black", linewidths=0.05, alpha=0.9)
    )
    ax.set_xlim(*lims)
    ax.set_ylim(*lims)
    ax.set_zlim(*lims)
    ax.set_title(title)
    ax.axis("off")
    ax.view_init(elev=25, azim=45)
    fig.patch.set_alpha(0.0)
    ax.patch.set_alpha(0.0)
    fig.tight_layout()
    filename.parent.mkdir(parents=True, exist_ok=True)
    timed_savefig(fig, filename, t0, dpi=300, bbox_inches="tight", transparent=True)
    plt.close(fig)


def main(depth=4, output_dir=OUTPUT_DIR):
    print("Generation de l'eponge de Menger...")
    t0 = time.perf_counter()
    cubes = iterate_3d(menger_ifs(), [UNIT_CUBE], depth=depth)
    plot_cubes(cubes, output_dir / "menger_sponge.png", "Eponge de Menger", started_at=t0)
    print("Termine.")


if __name__ == "__main__":
    main()
