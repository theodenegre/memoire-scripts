"""Démonstration PIFS quadtree sur images en niveaux de gris (chapitre 6).

Compresse puis décompresse trois images (fougère, triangle de Sierpinski, chat) par
l'algorithme FICANRP : partition adaptative par quadtree, domaines 2B->B (moyenne
2x2), isométries D4, ajustement s, o par moindres carrés (|s| <= S_MAX < 1).
La compression utilise une version vectorisée numpy du même algorithme que
``pifs_core`` (résultats équivalents, beaucoup plus rapide). Le décodage itère
l'opérateur de collage depuis un bruit jusqu'à l'attracteur (théorème de Banach).

Sorties :
    figures/ch6_collage_compression/decompression_bruit_initial.png
    figures/ch6_collage_compression/decompression_{fern,sierpinski,cat}_comparison.png
    figures/ch6_collage_compression/decompression_grilles_comparison.png

Usage :
    python pifs_quadtree_demo.py
"""

import sys
import time
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pifs_core import S_MAX, downsample_half, get_symmetries_list, load_image_grayscale
from figmeta import timed_savefig

FIGURES_DIR = Path(__file__).resolve().parents[2] / "figures" / "ch6_collage_compression"
SOURCES_DIR = Path(__file__).resolve().parents[2] / "figures" / "sources"


def _build_domain_pool(img, r_size, d_step):
    """Pré-calcule tous les domaines candidats (réduits + isométries) pour une taille de range."""
    h, w = img.shape
    d_size = 2 * r_size
    step = max(1, int(r_size * d_step))
    cands, coords = [], []
    for dy in range(0, h - d_size + 1, step):
        for dx in range(0, w - d_size + 1, step):
            d_block = downsample_half(img[dy:dy + d_size, dx:dx + d_size])
            for sym_idx, sym_d in enumerate(get_symmetries_list(d_block)):
                cands.append(sym_d.flatten())
                coords.append((dx, dy, sym_idx))
    D = np.array(cands, dtype=np.float32)
    return D, D.sum(axis=1), (D * D).sum(axis=1), coords


def compress_ficanrp_fast(img, config):
    """Quadtree FICANRP vectorisé (fit_affine en batch numpy)."""
    from tqdm import tqdm

    orig_h, orig_w = img.shape
    min_size = config.get("min_block_size", 4)
    max_size = config.get("max_block_size", 32)
    threshold = config.get("error_threshold", 50.0)
    d_step = config.get("domain_step", 2.0)

    h_pad = ((orig_h + max_size - 1) // max_size) * max_size
    w_pad = ((orig_w + max_size - 1) // max_size) * max_size
    if (h_pad, w_pad) != (orig_h, orig_w):
        padded = np.full((h_pad, w_pad), 128.0, dtype=np.float32)
        padded[:orig_h, :orig_w] = img
        img = padded
    h, w = img.shape

    pool_cache = {}

    def best_match(r_block, r_size):
        if r_size not in pool_cache:
            pool_cache[r_size] = _build_domain_pool(img, r_size, d_step)
        D, sD, ssD, coords = pool_cache[r_size]
        R = r_block.flatten().astype(np.float32)
        n = R.size
        den = n * ssD - sD * sD
        s = np.where(den != 0, (n * (D @ R) - R.sum() * sD) / den, 0.0)
        s = np.clip(s, -S_MAX, S_MAX)
        o = np.clip((R.sum() - s * sD) / n, -255.0, 510.0)
        mse = np.mean((R - (s[:, None] * D + o[:, None])) ** 2, axis=1)
        best = int(np.argmin(mse))
        dx, dy, sym = coords[best]
        return dx, dy, s[best], o[best], sym, float(mse[best])

    queue = [(x, y, max_size) for y in range(0, h, max_size) for x in range(0, w, max_size)]
    codes = []
    pbar = tqdm(desc="Compression", unit="blocs", dynamic_ncols=True)

    while queue:
        rx, ry, r_size = queue.pop(0)
        r_block = img[ry:ry + r_size, rx:rx + r_size]

        if np.max(r_block) - np.min(r_block) < 0.1:
            codes.append({"rx": rx, "ry": ry, "size": r_size, "dx": 0, "dy": 0,
                          "s": 0.0, "o": float(np.mean(r_block)), "sym": 0})
            pbar.update(1)
            continue

        dx, dy, s, o, sym, mse = best_match(r_block, r_size)
        if mse > threshold and r_size > min_size:
            half = r_size // 2
            queue += [(rx, ry, half), (rx + half, ry, half), (rx, ry + half, half), (rx + half, ry + half, half)]
        else:
            codes.append({"rx": rx, "ry": ry, "size": r_size, "dx": dx, "dy": dy,
                          "s": float(s), "o": float(o), "sym": sym})
            pbar.update(1)
            pbar.set_postfix({"queue": len(queue), "taille": r_size})

    pbar.close()
    return {"width": w, "height": h, "codes": codes, "orig_w": orig_w, "orig_h": orig_h}


def decompress_ficanrp(codes_data, max_it=30, rmse_tol=0.3, seed=42):
    """Itère l'opérateur de collage depuis un bruit, arrêt quand le pas RMSE est stable."""
    w, h = codes_data["width"], codes_data["height"]
    codes = codes_data["codes"]
    np.random.seed(seed)
    current_img = np.random.uniform(0, 255, (h, w)).astype(np.float32)

    states, timings = {0: current_img.copy()}, {0: 0.0}
    cumulative_ms, final_it = 0.0, max_it

    print(f"  Décompression ({w}x{h}, {len(codes)} blocs)...")
    for it in range(1, max_it + 1):
        t0 = time.time()
        prev_img = current_img.copy()
        next_img = np.zeros_like(current_img)
        for code in codes:
            size = code["size"]
            d_block = downsample_half(current_img[code["dy"]:code["dy"] + 2 * size, code["dx"]:code["dx"] + 2 * size])
            d_trans = get_symmetries_list(d_block)[code["sym"]]
            next_img[code["ry"]:code["ry"] + size, code["rx"]:code["rx"] + size] = code["s"] * d_trans + code["o"]
        current_img = np.clip(next_img, 0, 255)

        cumulative_ms += (time.time() - t0) * 1000.0
        delta = float(np.sqrt(np.mean((current_img - prev_img) ** 2)))
        states[it], timings[it] = current_img.copy(), cumulative_ms
        print(f"    Étape {it:2d}: cumul {cumulative_ms:.1f} ms, d={delta:.4f}")

        if it >= 3 and delta < rmse_tol:
            final_it = it
            print(f"    Convergence à l'étape {it} (d={delta:.4f} < {rmse_tol})")
            break

    return states, timings, final_it


def _draw_full_grid(ax, img, codes):
    h, w = img.shape
    for code in codes:
        ax.add_patch(mpatches.Rectangle(
            (code["rx"] - 0.5, code["ry"] - 0.5), code["size"], code["size"],
            linewidth=0.3, edgecolor="black", facecolor="white", zorder=3))
    ax.set_xlim(-0.5, w - 0.5)
    ax.set_ylim(h - 0.5, -0.5)


def _block_stats_str(codes):
    counts = Counter(c["size"] for c in codes)
    return "  ".join(f"{counts[s]}×{s}px" for s in sorted(counts))


def generate_grid_figure(examples_data, output_path):
    t0 = time.perf_counter()
    n = len(examples_data)
    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 4.5), dpi=300)
    if n == 1:
        axes = [axes]
    for ax, (name, orig, codes) in zip(axes, examples_data):
        h_i, w_i = orig.shape
        ax.imshow(np.full((h_i, w_i), 255, dtype=np.float32), cmap="gray", vmin=0, vmax=255)
        _draw_full_grid(ax, orig, codes)
        ax.axis("off")
        ax.set_title(name.capitalize(), fontsize=10, color="#1f2937")
    plt.tight_layout()
    timed_savefig(plt, output_path, t0, bbox_inches="tight", pad_inches=0.1)
    plt.close()


def generate_comparison_figure(orig, states, timings, final_it, codes, compress_time_s, output_path, cmap="gray"):
    t0 = time.perf_counter()
    h_img, w_img = orig.shape
    layout = [(0, 0, ("img", None)), (0, 1, ("img", final_it)), (1, 0, ("img", 1)),
              (1, 1, ("grid",)), (2, 0, ("img", 2)), (2, 1, ("img", 3))]

    fig, axes = plt.subplots(3, 2, figsize=(6.4, 9.6), dpi=300, gridspec_kw={"hspace": 0.35, "wspace": 0.05})
    for row, col, content in layout:
        ax = axes[row, col]
        if content[0] == "img":
            it = content[1]
            if it is None:
                ax.imshow(orig, cmap=cmap, vmin=0, vmax=255)
                ax.set_title(f"Originale  ({w_img}x{h_img}px)", fontsize=8, color="#1f2937", pad=3)
            elif it in states:
                ax.imshow(states[it], cmap=cmap, vmin=0, vmax=255)
                rmse = np.sqrt(np.mean((orig - states[it]) ** 2))
                t_ms = timings.get(it, 0.0)
                t_str = f"{t_ms / 1000.:.2f}s" if t_ms >= 1000. else f"{t_ms:.0f}ms"
                suffix = " (conv.)" if it == final_it else ""
                ax.set_title(f"Étape {it}{suffix}  RMSE {rmse:.1f}  {t_str}", fontsize=7, color="#1f2937", pad=3)
            else:
                ax.axis("off")
                continue
        else:
            ax.imshow(np.full((h_img, w_img), 255, dtype=np.float32), cmap="gray", vmin=0, vmax=255)
            _draw_full_grid(ax, orig, codes)
            ax.set_title("Arbre quaternaire", fontsize=8, color="#1f2937", pad=3)
        ax.axis("off")

    caption = (f"Compression : {compress_time_s:.1f}s  |  {len(codes)} blocs  {_block_stats_str(codes)}"
               "  |  temps indiqués : cumulés depuis étape 0")
    fig.text(0.5, 0.01, caption, ha="center", fontsize=7, color="#374151")
    timed_savefig(plt, output_path, t0, bbox_inches="tight", pad_inches=0.12)
    plt.close()


def _save_noise_figure(shape, output_path, seed=42):
    t0 = time.perf_counter()
    np.random.seed(seed)
    noise = np.random.uniform(0, 255, shape).astype(np.float32)
    fig, ax = plt.subplots(1, 1, figsize=(3.2, 3.2), dpi=300)
    ax.imshow(noise, cmap="gray", vmin=0, vmax=255)
    ax.axis("off")
    ax.set_title("Bruit initial (étape 0)", fontsize=9, color="#1f2937")
    fig.text(0.5, 0.01, f"Image aléatoire uniforme, résolution indicative ({shape[1]}×{shape[0]}px)",
             ha="center", fontsize=6.5, color="#6b7280")
    timed_savefig(plt, output_path, t0, bbox_inches="tight", pad_inches=0.1)
    plt.close()
    print(f"Sauvegarde bruit: {output_path}")


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    examples = [
        {"name": "fern", "file": SOURCES_DIR / "real_fern_source.png"},
        {"name": "sierpinski", "file": SOURCES_DIR / "sierpinski_source.png", "domain_step": 4.0},
        {"name": "cat", "file": SOURCES_DIR / "cat_source.png"},
    ]
    base_config = {"max_width": 256, "min_block_size": 2, "max_block_size": 128,
                   "error_threshold": 50.0, "domain_step": 2.0}

    grid_data = []
    noise_saved = False

    for ex in examples:
        name = ex["name"]
        config = {**base_config, "domain_step": ex.get("domain_step", base_config["domain_step"])}
        print(f"\nTraitement de {name}...")

        t0 = time.time()
        raw_img = load_image_grayscale(str(ex["file"]), config["max_width"])
        codes_data = compress_ficanrp_fast(raw_img, config)
        compress_time = time.time() - t0
        print(f"Compression en {compress_time:.2f}s, {len(codes_data['codes'])} blocs")

        oh, ow = codes_data.get("orig_h", raw_img.shape[0]), codes_data.get("orig_w", raw_img.shape[1])
        grid_data.append((name, raw_img.astype(np.float32), codes_data["codes"]))

        states, timings, final_it = decompress_ficanrp(codes_data)
        states = {k: v[:oh, :ow] for k, v in states.items()}

        if not noise_saved:
            _save_noise_figure(raw_img.shape, FIGURES_DIR / "decompression_bruit_initial.png")
            noise_saved = True

        out_img = FIGURES_DIR / f"decompression_{name}_comparison.png"
        generate_comparison_figure(raw_img.astype(np.float32), states, timings, final_it,
                                   codes_data["codes"], compress_time, out_img)
        print(f"Sauvegarde: {out_img}")
        print(f"  RMSE final (étape {final_it}): {np.sqrt(np.mean((raw_img.astype(np.float32) - states[final_it]) ** 2)):.4f}")

    generate_grid_figure(grid_data, FIGURES_DIR / "decompression_grilles_comparison.png")
    print("Sauvegarde grilles.")


if __name__ == "__main__":
    main()
