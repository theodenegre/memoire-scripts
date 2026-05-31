"""Transformation affine 2D w(x,y) = (a x + b y + e, c x + d y + f).

Deux représentations coexistent dans les scripts du mémoire :

- ``AffineMap`` : dataclass nommée, pratique pour les SFI codés en dur
  (fougère de Barnsley, Sierpiński). ``apply`` agit sur un tableau de points
  ``(..., 2)`` et préserve la forme d'entrée (polygone ou nuage).
- le format ``dict`` ``{"a", "b", "c", "d", "e", "f"}`` utilisé par les scripts
  pilotés par données (``ifs_labels``, ``fractales_etapes``), avec les helpers
  ``transform_points`` / ``apply_ifs_to_polys`` / ``apply_ifs_to_points``.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AffineMap:
    """Application affine du plan, optionnellement nommée (``name``)."""

    a: float
    b: float
    c: float
    d: float
    e: float
    f: float
    name: str = ""

    def apply(self, points):
        """Applique w à un tableau de points ``(..., 2)``, forme préservée."""
        pts = np.asarray(points, dtype=float)
        x, y = pts[..., 0], pts[..., 1]
        return np.stack(
            (self.a * x + self.b * y + self.e, self.c * x + self.d * y + self.f),
            axis=-1,
        )


def affine_dict(a, b, c, d, e, f):
    """Construit la représentation dict d'une transformation affine."""
    return {"a": a, "b": b, "c": c, "d": d, "e": e, "f": f}


def transform_points(pts, t):
    """Applique la transformation dict ``t`` à un tableau de points ``(N, 2)``."""
    pts = np.asarray(pts, dtype=float)
    x, y = pts[:, 0], pts[:, 1]
    return np.column_stack(
        (t["a"] * x + t["b"] * y + t["e"], t["c"] * x + t["d"] * y + t["f"])
    )
