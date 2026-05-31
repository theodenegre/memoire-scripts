"""Fougère de Barnsley engendrée à partir d'un singleton (chapitre 5).

Itère l'opérateur de Hutchinson de manière déterministe sur l'ensemble initial
A_0 = {(0,0)} : à chaque étape les quatre transformations affines sont appliquées
à tous les points courants, d'où 4^n points après n itérations. Méthode
déterministe (pas de jeu du chaos). La taille des points décroît avec n pour le
rendu, conformément au mémoire (fig:barnsley_singleton_evolution).

Sortie : figures/ch5_transformations/fern_singleton/fern_singleton_n{n}.png

Usage :
    python fougere_singleton.py
"""

import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from figmeta import timed_savefig

OUTPUT_DIR = (
    Path(__file__).resolve().parents[2] / "figures" / "ch5_transformations" / "fern_singleton"
)
DEFAULT_STEPS = [0, 1, 2, 3, 4, 5, 6, 7, 14]

PLOT_X_MIN, PLOT_X_MAX = -4.5, 3.0
PLOT_Y_MIN, PLOT_Y_MAX = -0.2, 10.5

# (a, b, c, d, e, f) des quatre transformations de Barnsley.
TRANSFORMS = (
    (0.0, 0.0, 0.0, 0.16, 0.0, 0.0),
    (0.85, 0.04, -0.04, 0.85, 0.0, 1.6),
    (0.20, -0.26, 0.23, 0.22, 0.0, 1.6),
    (-0.15, 0.28, 0.26, 0.24, 0.0, 0.44),
)


# Au-dela de ~20 millions de points, l'attracteur est visuellement sature : on
# sous-echantillonne l'ensemble courant pour eviter une explosion memoire (4^14
# points feraient ~8 Go). 20 M de points (~320 Mo) donnent un rendu dense.
MAX_POINTS = 20_000_000


def hutchinson_points(points):
    x, y = points[:, 0], points[:, 1]
    new_x = np.concatenate([a * x + b * y + e for a, b, _, _, e, _ in TRANSFORMS])
    new_y = np.concatenate([c * x + d * y + f for _, _, c, d, _, f in TRANSFORMS])
    return np.column_stack((new_x, new_y))


def generate_fern_points(iterations, max_points=MAX_POINTS, seed=0):
    rng = np.random.default_rng(seed)
    points = np.array([[0.0, 0.0]], dtype=float)
    for _ in range(iterations):
        points = hutchinson_points(points)
        if len(points) > max_points:
            idx = rng.choice(len(points), size=max_points, replace=False)
            points = points[idx]
    return points


def _marker_for(num_points):
    if num_points < 50:
        return dict(marker="o", markersize=20)
    if num_points < 500:
        return dict(marker="o", markersize=10)
    if num_points < 5000:
        return dict(marker="o", markersize=4)
    if num_points < 100000:
        return dict(marker=".", markersize=2)
    return dict(marker=",", markersize=0.1, alpha=0.5)


def plot_fern_points(points, output_path, color="#1f7a3f", started_at=None):
    t0 = started_at if started_at is not None else time.perf_counter()
    fig, ax = plt.subplots(figsize=(6, 10), dpi=220)
    ax.plot(points[:, 0], points[:, 1], color=color, linestyle="None", **_marker_for(len(points)))
    ax.set_xlim(PLOT_X_MIN, PLOT_X_MAX)
    ax.set_ylim(PLOT_Y_MIN, PLOT_Y_MAX)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")
    fig.tight_layout(pad=0)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    timed_savefig(fig, output_path, t0, pad_inches=0)
    plt.close(fig)


def main(steps=DEFAULT_STEPS, output_dir=OUTPUT_DIR):
    output_dir.mkdir(parents=True, exist_ok=True)
    for n in steps:
        start = time.perf_counter()
        points = generate_fern_points(n)
        out_path = output_dir / f"fern_singleton_n{n}.png"
        plot_fern_points(points, out_path, started_at=start)
        print(f"  {out_path.name} ({len(points)} pts, {time.perf_counter() - start:.2f}s)")


if __name__ == "__main__":
    main()
