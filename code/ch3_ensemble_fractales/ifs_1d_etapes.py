"""Itérations déterministes d'IFS unidimensionnels (chapitre 3).

Génère dans ``figures/ch3_ensemble_fractales/steps/`` les étapes A_n = W^n(A_0)
de trois systèmes, chacun comme union de copies contractées d'intervalles :

- ``two_intervals_n_poly``   : attracteur [0,1/3] u [2/3,1], A_0 = attracteur (point fixe).
- ``two_intervals_n_conv_dyn``: même IFS, A_0 arbitraire (convergence vers l'attracteur).
- ``cantor_n_poly``          : triadique de Cantor, A_0 = [0,1].

Chaque transformation w(x) = k*x + shift agit sur des segments. Méthode purement
déterministe (itération de l'opérateur de Hutchinson), conforme au mémoire.

Usage :
    python ifs_1d_etapes.py
"""

import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from figmeta import timed_savefig

STEPS_DIR = (
    Path(__file__).resolve().parents[2] / "figures" / "ch3_ensemble_fractales" / "steps"
)

# IFS [0,1/3] u [2/3,1] : 4 contractions de rapport 1/4 (cf. mémoire).
TWO_INTERVALS_TRANSFORMS = [(1 / 4, 0), (1 / 4, 1 / 12), (1 / 4, 2 / 3), (1 / 4, 3 / 4)]
# IFS du triadique de Cantor : 2 contractions de rapport 1/3.
CANTOR_TRANSFORMS = [(1 / 3, 0), (1 / 3, 2 / 3)]


def apply_ifs(intervals, transforms):
    return [
        (a * k + shift, b * k + shift) for a, b in intervals for k, shift in transforms
    ]


def iterate(initial, transforms, n_steps):
    """Renvoie A_{n_steps} = W^{n_steps}(initial)."""
    intervals = initial
    for _ in range(n_steps):
        intervals = apply_ifs(intervals, transforms)
    return intervals


def plot_intervals(intervals, output_path, color="#2c3e50", x_range=None):
    t0 = time.perf_counter()
    fig, ax = plt.subplots(figsize=(6, 1.2), dpi=200)
    for a, b in intervals:
        if a == b:
            ax.plot([a, a], [0, 0], marker="|", markersize=20, color=color, markeredgewidth=2)
        else:
            ax.plot([a, b], [0, 0], color=color, linewidth=20, solid_capstyle="butt")

    if x_range:
        ax.set_xlim(*x_range)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_position(("outward", 10))
    ax.get_yaxis().set_visible(False)
    ax.tick_params(axis="x", labelsize=10, colors="#7f8c8d")

    ax.set_ylim(-0.5, 0.5)
    fig.tight_layout(pad=0)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    timed_savefig(fig, output_path, t0, pad_inches=0.1, bbox_inches="tight")
    plt.close(fig)


def render_sequence(initial, transforms, n_max, name_for, color, x_range, output_dir):
    """``name_for(n)`` donne le nom de fichier (sans extension) de l'étape n."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for n in range(n_max):
        intervals = iterate(initial, transforms, n)
        plot_intervals(intervals, output_dir / f"{name_for(n)}.png", color=color, x_range=x_range)


def main(output_dir=STEPS_DIR):
    render_sequence(
        [(0, 1 / 3), (2 / 3, 1)],
        TWO_INTERVALS_TRANSFORMS,
        n_max=3,
        name_for=lambda n: f"two_intervals_{n}_poly",
        color="#e67e22",
        x_range=(-0.1, 1.1),
        output_dir=output_dir,
    )
    render_sequence(
        [(-0.5, -0.25), (0.25, 0.5), (0.75, 0.75)],
        TWO_INTERVALS_TRANSFORMS,
        n_max=6,
        name_for=lambda n: f"two_intervals_{n}_conv_dyn",
        color="#d35400",
        x_range=(-0.6, 1.1),
        output_dir=output_dir,
    )
    render_sequence(
        [(0, 1)],
        CANTOR_TRANSFORMS,
        n_max=7,
        name_for=lambda n: f"cantor_{n}_poly",
        color="#2c3e50",
        x_range=(-0.1, 1.1),
        output_dir=output_dir,
    )


if __name__ == "__main__":
    main()
