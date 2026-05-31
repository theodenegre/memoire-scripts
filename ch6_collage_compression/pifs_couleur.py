"""Compression PIFS d'images en couleurs, canal par canal (chapitre 6).

Une image RGB est traitée comme trois images en niveaux de gris indépendantes
(I_R, I_G, I_B). Chaque canal est compressé puis décompressé par le même PIFS
quadtree que le cas gris (partition adaptative, domaines 2B->B par moyenne 2x2,
isométries D4, ajustement s, o par moindres carrés avec |s| <= S_MAX < 1), puis les
trois canaux reconstruits sont recombinés. Décodage itéré depuis un bruit.

Sorties : figures/ch6_collage_compression/color_channels_demo_{capybara,cat,fern}.png

Usage :
    python pifs_couleur.py
"""

import os
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from figmeta import timed_savefig
from capybara_binaire import create_capybara_pixel_art
from pifs_core import S_MAX, fit_affine, get_symmetries_list


def _best_match(r_block, domain_img, r_size, step):
    d_h, d_w = domain_img.shape
    best_error, best = float("inf"), (0, 0, 0)
    for dy in range(0, d_h - r_size + 1, step):
        for dx in range(0, d_w - r_size + 1, step):
            d_block = domain_img[dy:dy + r_size, dx:dx + r_size]
            for sym_idx, transformed_d in enumerate(get_symmetries_list(d_block)):
                _, _, error = fit_affine(r_block, transformed_d)
                if error < best_error:
                    best_error, best = error, (dx, dy, sym_idx)
    return best, best_error


def compress_channel(channel, min_size, max_size, error_threshold, step_factor):
    h, w = channel.shape
    domain_img = channel.reshape(h // 2, 2, w // 2, 2).mean(axis=(1, 3))

    queue = [
        (x, y, max_size)
        for y in range(0, h, max_size)
        for x in range(0, w, max_size)
        if y + max_size <= h and x + max_size <= w
    ]
    codes = []

    while queue:
        rx, ry, r_size = queue.pop(0)
        r_block = channel[ry:ry + r_size, rx:rx + r_size]

        if np.max(r_block) - np.min(r_block) < 0.1 or r_size == min_size:
            codes.append({"rx": rx, "ry": ry, "size": r_size, "dx": 0, "dy": 0, "sym": 0,
                          "s": 0.0, "o": float(np.mean(r_block))})
            continue

        step = max(1, int(r_size * step_factor))
        (best_dx, best_dy, best_sym), best_error = _best_match(r_block, domain_img, r_size, step)

        if best_error > error_threshold and r_size > min_size:
            half = r_size // 2
            queue += [(rx, ry, half), (rx + half, ry, half), (rx, ry + half, half), (rx + half, ry + half, half)]
            continue

        d_block = get_symmetries_list(domain_img[best_dy:best_dy + r_size, best_dx:best_dx + r_size])[best_sym]
        s, o, _ = fit_affine(r_block, d_block)
        codes.append({"rx": rx, "ry": ry, "size": r_size, "dx": best_dx, "dy": best_dy,
                      "sym": best_sym, "s": s, "o": o})
    return codes


def decompress_channel(codes, shape, iterations=16, seed=42):
    h, w = shape
    np.random.seed(seed)
    current_img = np.random.uniform(0, 255, (h, w)).astype(np.float32)

    for _ in range(iterations):
        next_img = np.zeros_like(current_img)
        domain_img = current_img.reshape(h // 2, 2, w // 2, 2).mean(axis=(1, 3))
        for c in codes:
            size = c["size"]
            d_block = get_symmetries_list(domain_img[c["dy"]:c["dy"] + size, c["dx"]:c["dx"] + size])[c["sym"]]
            next_img[c["ry"]:c["ry"] + size, c["rx"]:c["rx"] + size] = c["s"] * d_block + c["o"]
        current_img = np.clip(next_img, 0, 255)
    return current_img.astype(np.uint8)


def compress_decompress_channel(channel_img, min_size=2, max_size=32, error_threshold=30.0,
                                step_factor=1.0, iterations=16):
    channel = channel_img.astype(np.float32)
    codes = compress_channel(channel, min_size, max_size, error_threshold, step_factor)
    return decompress_channel(codes, channel.shape, iterations=iterations)


def compress_decompress_rgb(rgb, **qt_args):
    channels = [compress_decompress_channel(rgb[:, :, c].astype(np.float32), **qt_args) for c in range(3)]
    return np.stack(channels, axis=2)


def generate_color_channels_figure(orig_rgb, recon_rgb, output_path, title_prefix):
    t0 = time.perf_counter()
    decomp_r, decomp_g, decomp_b = recon_rgb[:, :, 0], recon_rgb[:, :, 1], recon_rgb[:, :, 2]
    h, w, _ = orig_rgb.shape

    def channel_view(channel, axis_index):
        vis = np.zeros((h, w, 3), dtype=np.uint8)
        vis[:, :, axis_index] = channel
        return vis

    fig, axes = plt.subplots(1, 5, figsize=(18, 4), dpi=300)
    panels = [
        (orig_rgb, "1. Originale Couleur", "#2c3e50"),
        (channel_view(decomp_r, 0), "2. Canal Rouge (R)", "#e74c3c"),
        (channel_view(decomp_g, 1), "3. Canal Vert (G)", "#2ecc71"),
        (channel_view(decomp_b, 2), "4. Canal Bleu (B)", "#3498db"),
        (recon_rgb, "5. Image Compilée (R+G+B)", "#8e44ad"),
    ]
    for ax, (img, title, color) in zip(axes, panels):
        ax.imshow(img)
        ax.set_title(title, fontsize=10, fontweight="bold", color=color)
        ax.axis("off")

    plt.suptitle(f"Compression Couleur Canal par Canal (Quadtree) : {title_prefix}",
                 fontsize=13, fontweight="bold", y=0.98, color="#2c3e50")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    timed_savefig(plt, output_path, t0, bbox_inches="tight", pad_inches=0.1)
    plt.close()


def _load_rgb_resized(path, size=(128, 128)):
    img = Image.open(path).convert("RGB").resize(size, Image.Resampling.LANCZOS)
    return np.array(img)


def _report(name, orig, recon):
    diff = orig.astype(np.float32) - recon.astype(np.float32)
    print(f"{name}, RMSE: {np.sqrt(np.mean(diff ** 2)):.2f}, MaxErr: {np.max(np.abs(diff)):.2f}")


def main():
    figures_dir = Path(__file__).resolve().parents[2] / "figures" / "ch6_collage_compression"
    ch5_dir = Path(__file__).resolve().parents[2] / "figures" / "ch5_transformations"
    sources_dir = Path(__file__).resolve().parents[2] / "figures" / "sources"
    figures_dir.mkdir(parents=True, exist_ok=True)

    print("Processing Capybara...")
    capy_orig = create_capybara_pixel_art()
    capy_recon = compress_decompress_rgb(
        capy_orig, min_size=2, max_size=8, error_threshold=30.0, step_factor=2.0, iterations=16
    )
    generate_color_channels_figure(
        capy_orig, capy_recon, figures_dir / "color_channels_demo_capybara.png", "Capybara (32x32, Quadtree 2-8)"
    )
    _report("Capybara", capy_orig, capy_recon)

    natural = [
        ("Chat", sources_dir / "cat_source.png", figures_dir / "color_channels_demo_cat.png", "Chat (Quadtree 2-32)"),
        ("Fougère", ch5_dir / "test_fern_colored.png", figures_dir / "color_channels_demo_fern.png", "Fougère (Quadtree 2-32)"),
    ]
    qt_args = dict(min_size=2, max_size=32, error_threshold=50.0, step_factor=2.0, iterations=16)
    for name, src, out, title in natural:
        if not os.path.exists(src):
            continue
        print(f"Processing {name}...")
        orig = _load_rgb_resized(src)
        recon = compress_decompress_rgb(orig, **qt_args)
        generate_color_channels_figure(orig, recon, out, title)
        _report(name, orig, recon)


if __name__ == "__main__":
    main()
