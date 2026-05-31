"""Chemin continu entre deux attracteurs dans H(R^2) (chapitre 4).

Illustre que H(R^2) est connexe par arcs : on relie l'ensemble de Cantor (vu
dans le plan) au triangle de Sierpinski par le segment
    gamma(t) = (1-t)*Cantor + t*Sierpinski,   t in [0, 1].

Les nuages de points Cantor et Sierpinski sont eux-mêmes produits par itération
des IFS correspondants (méthode déterministe sur un nuage de points initial).

Sorties :
    figures/ch4_proprietes_hx/cantor_to_sierpinski.png      (panneau multi-t)
    figures/ch4_proprietes_hx/steps/cantor_sierpinski_tXXX.png (un t par image)

Usage :
    python chemin_cantor_sierpinski.py
"""

from pathlib import Path

import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from figmeta import timed_savefig

RNG = np.random.default_rng(42)
N_POINTS = 8000
N_IFS_ITER = 14
FIGURES_DIR = Path(__file__).resolve().parents[2] / "figures" / "ch4_proprietes_hx"
STEPS_DIR = FIGURES_DIR / "steps"
DEFAULT_TS = [0.0, 0.25, 0.5, 0.75, 1.0]


def cantor_ifs_points(n):
    pts = np.zeros((n, 2))
    pts[:, 0] = RNG.random(n)
    for _ in range(N_IFS_ITER):
        choice = RNG.integers(0, 2, size=n)
        pts[:, 0] = np.where(choice == 0, pts[:, 0] / 3, pts[:, 0] / 3 + 2 / 3)
    return pts


def sierpinski_ifs_points(n):
    v = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, np.sqrt(3) / 2]])
    pts = RNG.random((n, 2))
    for _ in range(N_IFS_ITER):
        choice = RNG.integers(0, 3, size=n)
        for i in range(3):
            pts[choice == i] = (pts[choice == i] + v[i]) / 2
    return pts


def interpolate(A, B, t):
    idx_a = RNG.integers(0, len(A), size=N_POINTS)
    idx_b = RNG.integers(0, len(B), size=N_POINTS)
    return (1 - t) * A[idx_a] + t * B[idx_b]


def make_figure(ts, output_path):
    t0 = time.perf_counter()
    A = cantor_ifs_points(N_POINTS)
    B = sierpinski_ifs_points(N_POINTS)

    fig, axes = plt.subplots(1, len(ts), figsize=(3 * len(ts), 3.2))
    if len(ts) == 1:
        axes = [axes]

    for ax, t in zip(axes, ts):
        C = interpolate(A, B, t)
        ax.scatter(C[:, 0], C[:, 1], s=0.3, c="steelblue", rasterized=True)
        ax.set_aspect("equal")
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.1, 1.0)
        ax.set_title(f"$t = {t:.2f}$", fontsize=11)
        ax.axis("off")

    fig.suptitle(
        r"$\gamma(t) = (1-t)\,\mathrm{Cantor} + t\,\mathrm{Sierpi\'{n}ski}$ dans $\mathcal{H}(\mathbb{R}^2)$",
        fontsize=12,
        y=1.02,
    )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    timed_savefig(fig, output_path, t0, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_path}")


def main(ts=DEFAULT_TS):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    make_figure(ts, FIGURES_DIR / "cantor_to_sierpinski.png")
    for t in ts:
        make_figure([t], STEPS_DIR / f"cantor_sierpinski_t{int(t * 100):03d}.png")


if __name__ == "__main__":
    main()
