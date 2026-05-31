"""Itération déterministe de l'opérateur de Hutchinson W = U_i w_i.

Trois variantes selon la représentation de l'ensemble courant :

- polygones 2D transportés par des ``AffineMap`` (``iterate_polygons``),
- polygones / nuages 2D transportés par des transformations dict
  (``apply_ifs_to_polys`` pour une liste de polygones, ``apply_ifs_to_points``
  pour un seul nuage empilé),
- solides 3D transportés par ``{"M": matrice 3x3, "b": vecteur 3}``
  (``iterate_3d``).

Méthode purement déterministe (aucun jeu du chaos), conforme au mémoire.
"""

import numpy as np

from .affine import transform_points


def iterate_polygons(polygons, transforms, iterations):
    """Renvoie W^iterations appliqué à une liste de polygones (``AffineMap``).

    Chaque polygone est un tableau ``(k, 2)``. À chaque étape, tout polygone est
    remplacé par ses images par chaque transformation.
    """
    if iterations < 0:
        raise ValueError("Le nombre d'itérations doit être positif ou nul.")
    polys = [np.asarray(p, dtype=float).copy() for p in polygons]
    for _ in range(iterations):
        polys = [m.apply(poly) for poly in polys for m in transforms]
    return polys


def apply_ifs_to_polys(transforms, polys):
    """Une itération de W sur une liste de polygones (transformations dict)."""
    return [transform_points(poly, t) for poly in polys for t in transforms]


def apply_ifs_to_points(pts, transforms):
    """Une itération de W sur un nuage de points empilé (transformations dict)."""
    return np.concatenate([transform_points(pts, t) for t in transforms])


def iterate_3d(ifs, initial_solids, depth, verbose=True):
    """Renvoie W^depth appliqué à des solides 3D.

    Chaque solide est un tableau ``(k, 3)`` de sommets, chaque transformation un
    dict ``{"M": (3,3), "b": (3,)}``.
    """
    solids = list(initial_solids)
    for d in range(depth):
        if verbose:
            print(f"  Itération {d + 1}/{depth}...")
        solids = [(t["M"] @ solid.T).T + t["b"] for solid in solids for t in ifs]
    return solids
