"""Rendu commun des solides 3D d'un attracteur, colores par hauteur.

Mutualise le panneau 3D de ``sierpinski_3d`` (tetraedres) et
``produit_cantor_koch`` (boites) : chaque solide est un tableau de sommets
``(k, 3)``, ses faces sont donnees par une liste d'index ``faces`` ; la couleur
est determinee par la moyenne du solide le long d'un axe (``color_axis``).
"""

import time

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from figmeta import timed_savefig


def plot_solids(
    solids,
    faces,
    output_path,
    title,
    *,
    color_axis=2,
    cmap=None,
    view=(25, 45),
    cube_limits=None,
    started_at=None,
):
    """Dessine ``solids`` (faces colorees par hauteur) et sauvegarde le PNG.

    - ``faces`` : liste de listes d'index de sommets definissant chaque face.
    - ``color_axis`` : axe (0=x, 1=y, 2=z) servant au degrade de couleur.
    - ``cube_limits`` : ``(lo, hi)`` pour des bornes fixes identiques sur x/y/z ;
      si ``None``, les bornes sont calculees pour un cube englobant centre.
    """
    t0 = started_at if started_at is not None else time.perf_counter()
    cmap = cmap or plt.cm.viridis

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="3d")

    means = [float(np.mean(s[:, color_axis])) for s in solids]
    c_min, c_max = min(means), max(means)
    c_range = c_max - c_min or 1.0

    all_faces, all_colors = [], []
    for solid, value in zip(solids, means):
        color = cmap((value - c_min) / c_range)
        for face in faces:
            all_faces.append(solid[face])
            all_colors.append(color)

    print(f"Rendu de {len(all_faces)} faces...")
    ax.add_collection3d(
        Poly3DCollection(
            all_faces, facecolors=all_colors, edgecolors="black", linewidths=0.1, alpha=0.9
        )
    )

    if cube_limits is None:
        pts = np.concatenate(solids)
        center = (pts.min(axis=0) + pts.max(axis=0)) / 2
        half = np.max(pts.max(axis=0) - pts.min(axis=0)) / 2
        lims = [(center[i] - half, center[i] + half) for i in range(3)]
    else:
        lo, hi = cube_limits
        lims = [(lo, hi)] * 3
    ax.set_xlim(*lims[0])
    ax.set_ylim(*lims[1])
    ax.set_zlim(*lims[2])

    ax.set_title(title)
    ax.axis("off")
    ax.view_init(elev=view[0], azim=view[1])
    fig.patch.set_alpha(0.0)
    ax.patch.set_alpha(0.0)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    timed_savefig(fig, output_path, t0, dpi=300, bbox_inches="tight", transparent=True)
    plt.close(fig)
