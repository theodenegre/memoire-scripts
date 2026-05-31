"""Décodage PIFS du pentagone de Sierpinski depuis un bruit (chapitre 6).

Illustre l'algorithme de DÉCODAGE PIFS (algo:decodage_pifs) et la convergence
garantie par le théorème de Banach :

1. On construit une image cible : le pentagone de Sierpinski (rastérisé en niveaux
   de gris). Cette cible n'est qu'une donnée d'entrée.
2. On l'ENCODE par PIFS quadtree (pifs_core) : pour chaque bloc range, on cherche le
   bloc domaine 2B->B et l'isométrie minimisant l'erreur de collage par moindres
   carrés (s, o).
3. On DÉCODE en itérant l'opérateur de collage W depuis une image de bruit
   aléatoire. L'attracteur émerge en quelques itérations.

Contrairement à la version précédente (qui dessinait l'attracteur par jeu du chaos),
la figure montre maintenant bien le décodage PIFS : extraction de blocs domaines,
réduction, isométrie, transformation photométrique s*z + o, répétée.

Sortie : figures/ch6_collage_compression/sierpinski_pentagon_decode_bw.png

Usage :
    python pifs_decode_pentagone.py
"""

import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pifs_core import S_MAX, downsample_half, get_symmetries_list
from figmeta import timed_savefig

DEFAULT_OUTPUT = (
    Path(__file__).resolve().parents[2]
    / "figures"
    / "ch6_collage_compression"
    / "sierpinski_pentagon_decode_bw.png"
)

N_MAPS = 5
S = 0.5
ANGLE_OFFSET = np.pi / 2
CENTERS = np.array(
    [[np.cos(ANGLE_OFFSET + 2 * np.pi * k / N_MAPS), np.sin(ANGLE_OFFSET + 2 * np.pi * k / N_MAPS)]
     for k in range(N_MAPS)],
    dtype=float,
)
XMIN, XMAX, YMIN, YMAX = -1.15, 1.15, -1.15, 1.15


def render_target(size=128, n_points=400_000, seed=0):
    """Image cible (niveaux de gris) du pentagone de Sierpinski, par jeu du chaos.

    Sert uniquement à FABRIQUER l'image à compresser. Le décodage, lui, est un PIFS.
    """
    rng = np.random.default_rng(seed)
    pts = rng.uniform(-1.0, 1.0, (n_points, 2))
    grid = np.zeros((size, size), dtype=np.float64)
    for step in range(40):
        idx = rng.integers(0, N_MAPS, size=len(pts))
        pts = S * pts + (1 - S) * CENTERS[idx]
        if step >= 15:
            xi = ((pts[:, 0] - XMIN) / (XMAX - XMIN) * (size - 1)).astype(int)
            yi = ((YMAX - pts[:, 1]) / (YMAX - YMIN) * (size - 1)).astype(int)
            mask = (xi >= 0) & (xi < size) & (yi >= 0) & (yi < size)
            np.add.at(grid, (yi[mask], xi[mask]), 1)
    img = 255.0 * (grid > 0)
    return img.astype(np.float32)


def encode_pifs(img, block=8, domain_step=2):
    """Encodage PIFS à blocs fixes : (range, domaine 2B->B, isométrie, s, o) par moindres carrés."""
    h, w = img.shape
    codes = []
    d_size = 2 * block
    step = max(1, block * domain_step)

    pool, coords = [], []
    for dy in range(0, h - d_size + 1, step):
        for dx in range(0, w - d_size + 1, step):
            reduced = downsample_half(img[dy:dy + d_size, dx:dx + d_size])
            for sym_idx, sym_d in enumerate(get_symmetries_list(reduced)):
                pool.append(sym_d.flatten())
                coords.append((dx, dy, sym_idx))
    D = np.array(pool, dtype=np.float32)
    sD, ssD = D.sum(axis=1), (D * D).sum(axis=1)
    n = block * block

    for ry in range(0, h, block):
        for rx in range(0, w, block):
            R = img[ry:ry + block, rx:rx + block].flatten().astype(np.float32)
            den = n * ssD - sD * sD
            s = np.where(den != 0, (n * (D @ R) - R.sum() * sD) / den, 0.0)
            s = np.clip(s, -S_MAX, S_MAX)
            o = (R.sum() - s * sD) / n
            mse = np.mean((R - (s[:, None] * D + o[:, None])) ** 2, axis=1)
            best = int(np.argmin(mse))
            dx, dy, sym = coords[best]
            codes.append({"rx": rx, "ry": ry, "size": block, "dx": dx, "dy": dy,
                          "s": float(s[best]), "o": float(o[best]), "sym": sym})
    return {"width": w, "height": h, "codes": codes}


def decode_pifs(data, steps, seed=42):
    """Itère l'opérateur de collage depuis un bruit, renvoie {étape: image}."""
    w, h = data["width"], data["height"]
    codes = data["codes"]
    rng = np.random.default_rng(seed)
    current = rng.uniform(0, 255, (h, w)).astype(np.float32)

    frames = {0: current.copy()}
    for it in range(1, max(steps) + 1):
        next_img = np.zeros_like(current)
        for code in codes:
            size = code["size"]
            d_block = downsample_half(current[code["dy"]:code["dy"] + 2 * size, code["dx"]:code["dx"] + 2 * size])
            d_trans = get_symmetries_list(d_block)[code["sym"]]
            next_img[code["ry"]:code["ry"] + size, code["rx"]:code["rx"] + size] = code["s"] * d_trans + code["o"]
        current = np.clip(next_img, 0, 255)
        if it in steps:
            frames[it] = current.copy()
    return frames


def make_figure(output_path=DEFAULT_OUTPUT):
    t0 = time.perf_counter()
    steps = [0, 1, 2, 12]
    labels = {0: "Étape 0\n(bruit initial)", 1: "Itération 1", 2: "Itération 2",
              12: "Itération 12\n(attracteur)"}

    target = render_target()
    codes = encode_pifs(target, block=2)
    frames = decode_pifs(codes, steps)

    # 5 panneaux : cible originale + étapes du décodage
    n_panels = 1 + len(steps)
    fig, axes = plt.subplots(1, n_panels, figsize=(2.4 * n_panels + 0.3, 3.2), dpi=200)

    # Panneau 0 : image cible originale
    axes[0].imshow(255.0 - target, cmap="gray", vmin=0, vmax=255, interpolation="bilinear")
    axes[0].set_title(f"Cible originale\n({target.shape[1]}\u00d7{target.shape[0]} px)",
                      fontsize=8, color="#1f2937")

    # Panneaux 1-4 : étapes du décodage
    for ax, step in zip(axes[1:], steps):
        ax.imshow(255.0 - frames[step], cmap="gray", vmin=0, vmax=255, interpolation="bilinear")
        ax.set_title(labels[step], fontsize=8, color="#1f2937")

    for ax in axes:
        ax.axis("off")
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_edgecolor("#aaaaaa")
            spine.set_linewidth(0.6)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    timed_savefig(fig, output_path, t0, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Figure sauvegardée : {output_path}")


if __name__ == "__main__":
    make_figure()
