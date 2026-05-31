"""Produit cartésien Cantor x Koch dans R^3 par itération déterministe (chapitre 5).

Le produit cartésien de deux attracteurs est l'attracteur du SFI produit. On
combine ici le triadique de Cantor (axe x, 2 contractions de rapport 1/3) et la
courbe de Koch (plan yz, 4 transformations), soit 2x4 = 8 transformations affines
de R^3. On itère l'opérateur de Hutchinson sur une boîte initiale (déterministe).

Sortie (référencée par le LaTeX) :
    figures/ch3_ensemble_fractales/steps/cantor_koch.png

Usage :
    python produit_cantor_koch.py
"""

import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ifs_common.hutchinson import iterate_3d
from ifs_common.plotting3d import plot_solids

OUTPUT_PATH = (
    Path(__file__).resolve().parents[2]
    / "figures"
    / "ch3_ensemble_fractales"
    / "steps"
    / "cantor_koch.png"
)

BOX_FACES = [[0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 5, 4], [2, 3, 7, 6], [0, 3, 7, 4], [1, 2, 6, 5]]


def cantor_koch_ifs():
    sqrt3 = np.sqrt(3)
    cantor_x = [{"k": 1 / 3, "t": 0.0}, {"k": 1 / 3, "t": 2 / 3}]
    koch_yz = [
        (1 / 3, 0, 0, 1 / 3, 0, 0),
        (1 / 6, -sqrt3 / 6, sqrt3 / 6, 1 / 6, 1 / 3, 0),
        (1 / 6, sqrt3 / 6, -sqrt3 / 6, 1 / 6, 1 / 2, sqrt3 / 6),
        (1 / 3, 0, 0, 1 / 3, 2 / 3, 0),
    ]

    ifs = []
    for cx in cantor_x:
        for a, b, c, d, e, f in koch_yz:
            M = np.array([[cx["k"], 0, 0], [0, a, b], [0, c, d]])
            ifs.append({"M": M, "b": np.array([cx["t"], e, f])})
    return ifs


def initial_box(z_thick=0.05):
    return np.array(
        [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
         [0, 0, z_thick], [1, 0, z_thick], [1, 1, z_thick], [0, 1, z_thick]],
        dtype=float,
    )


def main(depth=6, output_path=OUTPUT_PATH):
    print("Generation du produit Cantor-Koch (3D)...")
    t0 = time.perf_counter()
    boxes = iterate_3d(cantor_koch_ifs(), [initial_box()], depth=depth)
    plot_solids(
        boxes,
        BOX_FACES,
        output_path,
        "Produit de Cantor et Koch",
        color_axis=1,
        cmap=plt.cm.magma,
        view=(20, -60),
        started_at=t0,
    )
    print("Termine.")


if __name__ == "__main__":
    main()
