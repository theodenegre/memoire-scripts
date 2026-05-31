"""Triadique de Cantor empilé par itération déterministe de l'opérateur de Hutchinson.

Génère ``figures/ch3_ensemble_fractales/cantor_stacked.png`` : les ensembles
A_0, A_1, ..., A_n = W^n(A_0) empilés verticalement, où W est l'union des deux
contractions w_1(x) = x/3 et w_2(x) = x/3 + 2/3 appliquées à des intervalles.

Conforme au chapitre 3 (méthode déterministe, aucun jeu du chaos).

Usage :
    python cantor_etapes.py
"""

import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from figmeta import timed_savefig

DEFAULT_OUTPUT = (
    Path(__file__).resolve().parents[2]
    / "figures"
    / "ch3_ensemble_fractales"
    / "cantor_stacked.png"
)


def w1(intervals):
    return [(a / 3, b / 3) for a, b in intervals]


def w2(intervals):
    return [(a / 3 + 2 / 3, b / 3 + 2 / 3) for a, b in intervals]


def hutchinson(intervals):
    return w1(intervals) + w2(intervals)


def generate_cantor(iterations):
    intervals = [(0.0, 1.0)]
    for _ in range(iterations):
        intervals = hutchinson(intervals)
    return intervals


def step_label(i):
    if i == 0:
        return "$A_0$"
    if i == 1:
        return "$A_1 = W(A_0)$"
    return f"$A_{{{i}}} = W^{{{i}}}(A_0)$"


def plot_cantor_stacked(steps, output_path, color="#000000"):
    t0 = time.perf_counter()
    fig, ax = plt.subplots(figsize=(10, 4), dpi=220)

    for y_idx, i in enumerate(steps):
        ax.text(-0.04, -y_idx, step_label(i), va="center", ha="right", fontsize=16, color=color)
        for a, b in generate_cantor(i):
            ax.plot([a, b], [-y_idx, -y_idx], color=color, linewidth=12, solid_capstyle="butt")

    ax.set_xlim(-0.4, 1.05)
    ax.set_ylim(-len(steps) + 0.5, 0.5)
    ax.axis("off")
    fig.tight_layout(pad=0)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    timed_savefig(fig, output_path, t0, pad_inches=0, bbox_inches="tight")
    plt.close(fig)


def main(output_path=DEFAULT_OUTPUT):
    plot_cantor_stacked([0, 1, 2, 3, 4, 6], output_path)
    print(f"Generated {output_path}")


if __name__ == "__main__":
    main()
