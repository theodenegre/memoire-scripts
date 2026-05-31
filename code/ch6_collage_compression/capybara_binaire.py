"""Illustration du partitionnement PIFS binaire sur un capybara 32x32 (chapitre 6).

Cas binaire géométrique du mémoire : l'image est partitionnée en cibles 4x4 et
sources 8x8. Chaque source est réduite à 4x4 (moyenne 2x2 puis binarisation) et
orientée par une isométrie de D4 pour approcher la cible. Aucun ajustement
photométrique (s, o) ici : la mise en correspondance est purement géométrique
(distance de type Hausdorff), conformément au mémoire.

Ce module expose aussi ``create_capybara_pixel_art`` (RGB) et
``create_capybara_bw_pixel_art`` (binaire), réutilisés par ``pifs_couleur.py``.

Sortie : figures/ch6_collage_compression/pifs_blocks_demo_capybara.png

Usage :
    python capybara_binaire.py
"""

import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from figmeta import timed_savefig

DEFAULT_OUTPUT = (
    Path(__file__).resolve().parents[2]
    / "figures"
    / "ch6_collage_compression"
    / "pifs_blocks_demo_capybara.png"
)

# Palette indexée du pixel art (0=ciel, 1=herbe, 2=contour, 3/4=orange, 5=tache, 6=oeil).
PALETTE_RGB = {
    0: [30, 167, 242], 1: [118, 190, 83], 2: [0, 0, 0], 3: [255, 147, 0],
    4: [243, 120, 39], 5: [255, 99, 56], 6: [0, 0, 0],
}
PALETTE_BW = {0: 255, 1: 255, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0}


def _build_capybara_grid():
    grid = np.zeros((32, 32), dtype=int)
    grid[0:24, :] = 0
    grid[24:32, :] = 1
    grid[5, 8:10] = 2; grid[5, 12:14] = 2
    grid[6, 8] = 2; grid[6, 9] = 3; grid[6, 10] = 2; grid[6, 12] = 2; grid[6, 13] = 3; grid[6, 14] = 2; grid[6, 15:26] = 2
    grid[7, 7] = 2; grid[7, 8:10] = 3; grid[7, 10] = 2; grid[7, 11] = 3; grid[7, 12] = 2; grid[7, 13:26] = 3; grid[7, 26] = 2
    grid[8, 7] = 2; grid[8, 8:12] = 3; grid[8, 12] = 2; grid[8, 13:26] = 3; grid[8, 26] = 2
    grid[9, 6] = 2; grid[9, 7:14] = 3; grid[9, 14:18] = 6; grid[9, 18:23] = 3; grid[9, 23:26] = 4; grid[9, 26] = 2
    grid[10, 6] = 2; grid[10, 7:13] = 3; grid[10, 13] = 2; grid[10, 14:18] = 6; grid[10, 18:22] = 3; grid[10, 22:26] = 4; grid[10, 26] = 2
    grid[11, 5] = 2; grid[11, 6:21] = 3; grid[11, 21:26] = 4; grid[11, 26] = 2
    grid[12, 4] = 2; grid[12, 5:21] = 3; grid[12, 21:26] = 4; grid[12, 26] = 2
    grid[13, 4] = 2; grid[13, 5:25] = 3; grid[13, 25:27] = 2
    grid[14, 3] = 2; grid[14, 4:25] = 3; grid[14, 25:27] = 2
    grid[15, 3] = 2; grid[15, 4:6] = 3; grid[15, 6:8] = 5; grid[15, 8:24] = 3; grid[15, 24:26] = 2
    grid[16, 2] = 2; grid[16, 3:6] = 3; grid[16, 6:8] = 5; grid[16, 8:20] = 3; grid[16, 20:25] = 2
    grid[17, 2] = 2; grid[17, 3:19] = 3; grid[17, 19] = 2
    grid[18, 2] = 2; grid[18, 3:19] = 3; grid[18, 19] = 2
    grid[19, 2] = 2; grid[19, 3:19] = 3; grid[19, 19] = 2
    grid[20, 2] = 2; grid[20, 3:18] = 3; grid[20, 18] = 2
    grid[21, 2] = 2; grid[21, 3:18] = 3; grid[21, 18] = 2
    grid[22, 2] = 2; grid[22, 3:18] = 3; grid[22, 18] = 2
    grid[23, 2] = 2; grid[23, 3:5] = 3; grid[23, 5] = 2; grid[23, 6:14] = 3; grid[23, 14] = 2; grid[23, 15:17] = 3; grid[23, 17] = 2
    grid[24, 2] = 2; grid[24, 3:5] = 3; grid[24, 5] = 2; grid[24, 14] = 2; grid[24, 15:17] = 3; grid[24, 17] = 2
    grid[25, 2:6] = 2; grid[25, 14:18] = 2
    return grid


def create_capybara_pixel_art():
    grid = _build_capybara_grid()
    rgb = np.zeros((32, 32, 3), dtype=np.uint8)
    for k, color in PALETTE_RGB.items():
        rgb[grid == k] = color
    return rgb


def create_capybara_bw_pixel_art():
    grid = _build_capybara_grid()
    binary = np.zeros((32, 32), dtype=np.uint8)
    for k, value in PALETTE_BW.items():
        binary[grid == k] = value
    return binary


def reduce_and_orient(source_8x8, isometry):
    """Réduit un bloc source 8x8 -> 4x4 (moyenne 2x2 + binarisation), puis applique l'isométrie."""
    reduced = source_8x8.reshape(4, 2, 4, 2).mean(axis=(1, 3))
    binarized = np.where(reduced < 128, 0.0, 255.0).astype(np.float32)
    ops = [
        lambda b: b, lambda b: np.rot90(b, 1), lambda b: np.rot90(b, 2), lambda b: np.rot90(b, 3),
        np.fliplr, np.flipud, lambda b: b.T, lambda b: np.fliplr(b).T.copy(),
    ]
    return ops[isometry](binarized)


PAIRS = [
    {"name": "1", "color": "#e67e22", "tgt_rect": (8, 4), "tgt_slice": (slice(4, 8), slice(8, 12)),
     "src_rect": (0, 8), "src_slice": (slice(8, 16), slice(0, 8)), "isometry": 3,
     "isometry_name": r"$w_1$ ($\mathrm{rot}\ 270^\circ$)", "y_pos": 0.55},
    {"name": "2", "color": "#2ecc71", "tgt_rect": (16, 12), "tgt_slice": (slice(12, 16), slice(16, 20)),
     "src_rect": (8, 8), "src_slice": (slice(8, 16), slice(8, 16)), "isometry": 0,
     "isometry_name": "$w_2$ (identité)", "y_pos": 0.10},
]


def _highlight_blocks(ax, img, pairs, rect_key, slice_size, prefix):
    ax.imshow(img, cmap="gray", vmin=0, vmax=255)
    for p in pairs:
        x, y = p[rect_key]
        ax.add_patch(patches.Rectangle((x - 0.5, y - 0.5), slice_size, slice_size, linewidth=2.5,
                                       edgecolor=p["color"], facecolor="none", zorder=10))
        ax.text(x + slice_size / 2, y + slice_size / 2, f"${prefix}_{p['name']}$", color=p["color"],
                fontsize=11, fontweight="bold", ha="center", va="center",
                bbox=dict(boxstyle="square,pad=0.15", fc="white", ec=p["color"], alpha=0.9, lw=1.5))
    ax.set_xlim(-0.5, 31.5)
    ax.set_ylim(31.5, -0.5)
    ax.axis("off")


def _add_pixel_axes(parent_ax, rect_frac, pixels, border_color, title, title_below=False):
    fig = parent_ax.figure
    bb = parent_ax.get_position()
    sub = fig.add_axes([bb.x0 + rect_frac[0] * bb.width, bb.y0 + rect_frac[1] * bb.height,
                        rect_frac[2] * bb.width, rect_frac[3] * bb.height])
    sub.imshow(pixels, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
    sub.set_xticks([])
    sub.set_yticks([])
    for spine in sub.spines.values():
        spine.set_edgecolor(border_color)
        spine.set_linewidth(1.8)
    if title_below:
        sub.set_xlabel(title, fontsize=8.0, color=border_color, labelpad=4, fontweight="bold")
    else:
        sub.set_title(title, fontsize=8.0, color=border_color, pad=3, fontweight="bold")


def _draw_transformations_panel(ax, gray_img, pairs):
    ax.axis("off")
    ax.set_title("D. Exemples de transformations contractives $w_i$", fontsize=11, fontweight="bold", pad=10)
    for p in pairs:
        src = gray_img[p["src_slice"]]
        tgt = gray_img[p["tgt_slice"]]
        reconstructed = reduce_and_orient(src, p["isometry"])
        y, y_center = p["y_pos"], p["y_pos"] + 0.13

        _add_pixel_axes(ax, [0.02, y, 0.22, 0.28], src, p["color"], f"Source $S_{p['name']}$ (8×8)")
        _add_pixel_axes(ax, [0.74, y, 0.22, 0.28], tgt, p["color"], f"Cible $C_{p['name']}$ (4×4)")
        _add_pixel_axes(ax, [0.39, y, 0.22, 0.28], reconstructed, p["color"],
                        f"Reconstruit $\\hat{{C}}_{p['name']}$", title_below=True)

        ax.annotate("", xy=(0.38, y_center), xytext=(0.25, y_center), xycoords="axes fraction",
                    arrowprops=dict(arrowstyle="-|>", color=p["color"], lw=1.5, connectionstyle="arc3,rad=0.1"))
        ax.text(0.315, y_center + 0.055, p["isometry_name"], fontsize=8, color=p["color"], ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85))
        ax.annotate("", xy=(0.73, y_center), xytext=(0.62, y_center), xycoords="axes fraction",
                    arrowprops=dict(arrowstyle="<|-|>", color=p["color"], lw=1.5, linestyle="dashed",
                                    connectionstyle="arc3,rad=-0.1"))
        ax.text(0.675, y_center + 0.055, "$h$", fontsize=8, color=p["color"], ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85))


def make_figure(output_path=DEFAULT_OUTPUT):
    t0 = time.perf_counter()
    gray_img = create_capybara_bw_pixel_art().astype(np.float32)
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 9), dpi=300)
    plt.subplots_adjust(wspace=0.3, hspace=0.3)

    axes[0, 0].imshow(gray_img, cmap="gray", vmin=0, vmax=255)
    axes[0, 0].axis("off")
    axes[0, 0].set_title("A. Image Originale Binaire ($32 \\times 32$ pixels)", fontsize=11, fontweight="bold", pad=10)

    _highlight_blocks(axes[0, 1], gray_img, PAIRS, "tgt_rect", 4, "C")
    axes[0, 1].set_title("B. Grille des blocs cibles ($C_i$ de taille $4 \\times 4$)", fontsize=11, fontweight="bold", pad=10)
    for i in range(0, 33, 4):
        axes[0, 1].axhline(i - 0.5, color="#7f8c8d", linewidth=0.5, alpha=0.4)
        axes[0, 1].axvline(i - 0.5, color="#7f8c8d", linewidth=0.5, alpha=0.4)

    _highlight_blocks(axes[1, 0], gray_img, PAIRS, "src_rect", 8, "S")
    axes[1, 0].set_title("C. Grille des blocs sources ($S_j$ de taille $8 \\times 8$)", fontsize=11, fontweight="bold", pad=10)
    for i in range(0, 33, 8):
        axes[1, 0].axhline(i - 0.5, color="#7f8c8d", linewidth=0.5, alpha=0.4)
        axes[1, 0].axvline(i - 0.5, color="#7f8c8d", linewidth=0.5, alpha=0.4)

    _draw_transformations_panel(axes[1, 1], gray_img, PAIRS)

    plt.suptitle("Illustration du Partitionnement PIFS sur le Capybara Binaire (32x32)",
                 fontsize=13, fontweight="bold", color="#2c3e50", y=0.97)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    timed_savefig(plt, output_path, t0, bbox_inches="tight", pad_inches=0.1)
    plt.close()
    print(f"Figure sauvegardée : {output_path}")


def main():
    make_figure()


if __name__ == "__main__":
    main()
