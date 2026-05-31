"""Tétraèdre de Sierpinski par itération déterministe dans R^3 (chapitre 5).

SFI de 4 homothéties de rapport 1/2 centrées aux sommets d'un tétraèdre régulier
(généralisation 3D du triangle de Sierpinski). On itère l'opérateur de Hutchinson
sur le tétraèdre initial (méthode déterministe), puis on rend les faces colorées
par hauteur.

Sortie : figures/ch5_transformations/steps/sierpinski_tetrahedron.png

Usage :
    python sierpinski_3d.py
"""

import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ifs_common.hutchinson import iterate_3d
from ifs_common.plotting3d import plot_solids

OUTPUT_DIR = Path(__file__).resolve().parents[2] / "figures" / "ch5_transformations" / "steps"

TETRA_FACES = [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]]
VERTICES = np.array(
    [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.5, np.sqrt(3) / 2, 0.0],
        [0.5, np.sqrt(3) / 6, np.sqrt(6) / 3],
    ]
)


def sierpinski_tetrahedron_ifs(vertices=VERTICES):
    return [{"M": 0.5 * np.eye(3), "b": 0.5 * v} for v in vertices]


def main(depth=6, output_dir=OUTPUT_DIR):
    print("Generation du tetraedre de Sierpinski...")
    t0 = time.perf_counter()
    tetrahedrons = iterate_3d(sierpinski_tetrahedron_ifs(), [VERTICES.copy()], depth=depth)
    plot_solids(
        tetrahedrons,
        TETRA_FACES,
        output_dir / "sierpinski_tetrahedron.png",
        "Tetraedre de Sierpinski",
        view=(25, 45),
        cube_limits=(0, 1),
        started_at=t0,
    )
    print(f"Termine ({len(tetrahedrons)} tetraedres).")


if __name__ == "__main__":
    main()
