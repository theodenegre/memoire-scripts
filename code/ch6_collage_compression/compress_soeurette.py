"""Compression PIFS quadtree couleur des photos soeurette (soeurette1 et soeurette2).

L'image RGB est traitée comme trois images en niveaux de gris independantes
(I_R, I_G, I_B). Chaque canal est compresse puis decompresse par le meme PIFS
quadtree vectorise (FICANRP) que le cas gris, puis les trois canaux reconstruits
sont recombines.

Étapes :
1. Recadre chaque image à la plus haute puissance de 2 de chaque côté,
   en coupant équitablement à gauche/droite et haut/bas (crop centré).
2. Compresse chaque canal (R, G, B) par l'algorithme FICANRP vectorisé
   (quadtree adaptatif), seuil d'erreur maximale par carré : 150.
3. Décompresse chaque canal depuis un bruit aléatoire (théorème de Banach),
   puis recombine en image couleur.
4. Génère les figures de comparaison et d'arbre quaternaire.

Sorties :
    figures/ch6_collage_compression/soeurette1_quadtree_comparison.png
    figures/ch6_collage_compression/soeurette2_quadtree_comparison.png
    figures/ch6_collage_compression/soeurette_quadtrees_grid.png
    figures/sources/soeurette1_cropped.png
    figures/sources/soeurette2_cropped.png

Usage :
    python compress_soeurette.py
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
from PIL import Image

# ── Chemin vers pifs_core (même répertoire) ──────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pifs_core import S_MAX, downsample_half, get_symmetries_list
from figmeta import timed_savefig

# ── Chemins ──────────────────────────────────────────────────────────────────
ROOT          = Path(__file__).resolve().parents[2]
FIGURES_DIR   = ROOT / "figures" / "ch6_collage_compression"
SOURCES_DIR   = ROOT / "figures" / "sources"
MEMOIRE_DIR   = ROOT                          # contient les JPG source

IMG_PATHS = [
    SOURCES_DIR / "soeurette1.jpg",
    SOURCES_DIR / "soeurette2.jpg",
]
NAMES = ["soeurette1", "soeurette2"]

# ── Paramètres de compression ─────────────────────────────────────────────────
ERROR_THRESHOLD = 50.0
MIN_BLOCK_SIZE  = 4
# Résolution de travail : on recadre à une puissance de 2 puis on redimensionne
# à TARGET_SIZE x TARGET_SIZE (compromis qualité / temps de calcul).
TARGET_SIZE     = 512
# MAX_BLOCK_SIZE est calculé dynamiquement = plus grande puissance de 2 telle que
# les blocs domaines (2x) rentrent dans l'image : max_block_size = prev_p2(dim // 2)
DOMAIN_STEP     = 2.0      # pas de balayage des domaines (en fractions de taille)


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Recadrage à la plus haute puissance de 2
# ─────────────────────────────────────────────────────────────────────────────

def prev_power_of_two(n: int) -> int:
    """Plus haute puissance de 2 inferieure ou egale a n."""
    p = 1
    while p * 2 <= n:
        p *= 2
    return p


def max_block_for_image(h: int, w: int) -> int:
    """Plus grande taille de bloc telle que les domaines 2B x 2B rentrent dans l'image.
    Garantit : 2 * max_block_size <= min(h, w).
    """
    return prev_power_of_two(min(h, w) // 2)


def center_crop_to_power_of_two(img_path: Path, save_path: Path) -> np.ndarray:
    """Charge, recadre et sauvegarde l'image RGB recadrée. Renvoie le tableau (h, w, 3)."""
    pil = Image.open(img_path).convert("RGB")
    w, h = pil.size

    tw = prev_power_of_two(w)
    th = prev_power_of_two(h)

    left   = (w - tw) // 2
    top    = (h - th) // 2
    right  = left + tw
    bottom = top  + th

    cropped = pil.crop((left, top, right, bottom))
    if TARGET_SIZE is not None and (tw, th) != (TARGET_SIZE, TARGET_SIZE):
        cropped = cropped.resize((TARGET_SIZE, TARGET_SIZE), Image.Resampling.LANCZOS)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    cropped.save(str(save_path))
    print(f"  {img_path.name} : {w}x{h} -> recadrage centre {tw}x{th} -> "
          f"redim {cropped.size[0]}x{cropped.size[1]}")
    print(f"  Supprime : {w-tw}px horizontaux ({(w-tw)//2} de chaque cote), "
          f"{h-th}px verticaux ({(h-th)//2} de chaque cote)")
    print(f"  Sauvegarde : {save_path}")
    return np.array(cropped, dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Compression FICANRP vectorisée (reprise de pifs_quadtree_demo)
# ─────────────────────────────────────────────────────────────────────────────

def _build_domain_pool(img, r_size, d_step):
    """Pré-calcule tous les domaines candidats (réduits + isométries D4)."""
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
    """Quadtree FICANRP vectorisé."""
    try:
        from tqdm import tqdm
        use_tqdm = True
    except ImportError:
        use_tqdm = False

    orig_h, orig_w = img.shape
    min_size  = config.get("min_block_size",  MIN_BLOCK_SIZE)
    max_size  = config.get("max_block_size",  64)
    threshold = config.get("error_threshold", ERROR_THRESHOLD)
    d_step    = config.get("domain_step",     DOMAIN_STEP)

    # Padding pour que h et w soient multiples de max_size
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

    if use_tqdm:
        pbar = tqdm(desc="Compression", unit="blocs", dynamic_ncols=True, total=None, leave=True)
    else:
        pbar = None
    _pbar_count = [0]

    while queue:
        rx, ry, r_size = queue.pop(0)
        r_block = img[ry:ry + r_size, rx:rx + r_size]

        if np.max(r_block) - np.min(r_block) < 0.1:
            codes.append({"rx": rx, "ry": ry, "size": r_size, "dx": 0, "dy": 0,
                          "s": 0.0, "o": float(np.mean(r_block)), "sym": 0})
            if pbar is not None:
                pbar.update(1)
            continue

        dx, dy, s, o, sym, mse = best_match(r_block, r_size)
        if mse > threshold and r_size > min_size:
            half = r_size // 2
            queue += [(rx, ry, half), (rx + half, ry, half),
                      (rx, ry + half, half), (rx + half, ry + half, half)]
        else:
            codes.append({"rx": rx, "ry": ry, "size": r_size, "dx": dx, "dy": dy,
                          "s": float(s), "o": float(o), "sym": sym})
            if pbar is not None:
                pbar.update(1)
                pbar.set_postfix({"queue": len(queue), "taille": r_size})

    if pbar is not None: pbar.close()
    return {"width": w, "height": h, "codes": codes, "orig_w": orig_w, "orig_h": orig_h}


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Décompression itérative (Banach)
# ─────────────────────────────────────────────────────────────────────────────

def decompress_ficanrp(codes_data, max_it=30, rmse_tol=0.3, seed=42):
    """Itère l'opérateur de collage depuis un bruit uniforme."""
    w, h   = codes_data["width"], codes_data["height"]
    codes  = codes_data["codes"]
    np.random.seed(seed)
    current_img = np.random.uniform(0, 255, (h, w)).astype(np.float32)

    states   = {0: current_img.copy()}
    timings  = {0: 0.0}
    cumul_ms = 0.0
    final_it = max_it

    print(f"  Decompression ({w}x{h}, {len(codes)} blocs)...")
    for it in range(1, max_it + 1):
        t0       = time.time()
        prev_img = current_img.copy()
        next_img = np.zeros_like(current_img)
        for code in codes:
            size   = code["size"]
            d_blk  = downsample_half(current_img[code["dy"]:code["dy"] + 2*size,
                                                  code["dx"]:code["dx"] + 2*size])
            d_tr   = get_symmetries_list(d_blk)[code["sym"]]
            next_img[code["ry"]:code["ry"]+size, code["rx"]:code["rx"]+size] = \
                code["s"] * d_tr + code["o"]
        current_img = np.clip(next_img, 0, 255)

        cumul_ms += (time.time() - t0) * 1000.0
        delta     = float(np.sqrt(np.mean((current_img - prev_img) ** 2)))
        states[it]  = current_img.copy()
        timings[it] = cumul_ms
        print(f"    Etape {it:2d}: cumul {cumul_ms:.1f} ms, d={delta:.4f}")

        if it >= 3 and delta < rmse_tol:
            final_it = it
            print(f"    Convergence a l'etape {it} (d={delta:.4f} < {rmse_tol})")
            break

    return states, timings, final_it


# ─────────────────────────────────────────────────────────────────────────────
# 3c. Orchestration couleur : un canal R, G, B traite independamment
# ─────────────────────────────────────────────────────────────────────────────

CHANNEL_NAMES = ["R", "G", "B"]


def compress_rgb(rgb, config):
    """Compresse les 3 canaux (R, G, B). Renvoie une liste de 3 codes_data."""
    channels = []
    for c in range(3):
        print(f"  Canal {CHANNEL_NAMES[c]} ...")
        channels.append(compress_ficanrp_fast(rgb[:, :, c], config))
    return channels


def decompress_rgb(channels_data, max_it=30, rmse_tol=0.3, seed=42):
    """Decompresse les 3 canaux et recombine. Renvoie les etats RGB empiles.

    Renvoie (states_rgb, timings, final_it) ou states_rgb[it] est (h, w, 3) uint8.
    Les etats des 3 canaux sont alignes sur le nombre d'etapes du canal le plus lent.
    """
    per_channel = [decompress_ficanrp(cd, max_it=max_it, rmse_tol=rmse_tol, seed=seed)
                   for cd in channels_data]
    final_it = max(fit for _, _, fit in per_channel)

    states_rgb, timings = {}, {}
    for it in range(final_it + 1):
        chans = []
        cumul = 0.0
        for st, tm, fit in per_channel:
            key = it if it in st else fit          # canal converge plus tot : on fige
            chans.append(st[key])
            cumul = max(cumul, tm.get(key, tm.get(fit, 0.0)))
        states_rgb[it] = np.clip(np.stack(chans, axis=2), 0, 255).astype(np.uint8)
        timings[it] = cumul
    return states_rgb, timings, final_it


# ─────────────────────────────────────────────────────────────────────────────
# 3b. Sauvegarde binaire (.fif) et stats (.txt)
# ─────────────────────────────────────────────────────────────────────────────

import struct


def save_fif(filepath, codes_data):
    """Sauvegarde les codes PIFS en binaire compact (.fif, 18 octets/code)."""
    from pathlib import Path
    fp = Path(filepath)
    fp.parent.mkdir(parents=True, exist_ok=True)
    with open(fp, "wb") as f:
        f.write(b"FIF\x01")
        f.write(struct.pack("<HH", int(codes_data["width"]), int(codes_data["height"])))
        f.write(struct.pack("<I", len(codes_data["codes"])))
        for c in codes_data["codes"]:
            f.write(struct.pack("<HHHBHHff",
                                int(c["rx"]), int(c["ry"]), int(c["size"]),
                                int(c["sym"]), int(c["dx"]), int(c["dy"]),
                                float(c["s"]), float(c["o"])))
    return fp.stat().st_size


def save_stats(filepath, name, channels_data, orig_w, orig_h,
               compress_time_s, rmse_final, max_block):
    """Ecrit un rapport texte couleur (3 canaux) avec taux de compression."""
    from collections import Counter
    all_codes = [c for cd in channels_data for c in cd["codes"]]
    n_codes = len(all_codes)
    raw_bytes = orig_w * orig_h * 3          # RGB 8 bits par canal
    fif_bytes = 3 * (8 + 4) + n_codes * 19   # 3 en-tetes + codes (19 oct/code : HHHBHHff)
    ratio = raw_bytes / fif_bytes
    counts = Counter(c["size"] for c in all_codes)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(f"=== Rapport de compression PIFS quadtree couleur : {name} ===\n")
        f.write(f"Image (RGB)             : {orig_w}x{orig_h} px x 3 canaux\n")
        f.write(f"Taille brute            : {raw_bytes:,} octets ({raw_bytes/1024:.1f} Ko)\n")
        f.write(f"Taille compressee .fif  : {fif_bytes:,} octets ({fif_bytes/1024:.1f} Ko)\n")
        f.write(f"Taux de compression     : {ratio:.2f}:1\n")
        f.write(f"Nombre de codes PIFS    : {n_codes:,} (")
        f.write(", ".join(f"{CHANNEL_NAMES[i]}={len(cd['codes'])}"
                          for i, cd in enumerate(channels_data)) + ")\n")
        f.write(f"Taille max bloc racine  : {max_block} px\n")
        f.write(f"Seuil d'erreur          : {ERROR_THRESHOLD:.0f} (MSE)\n")
        f.write(f"Temps de compression    : {compress_time_s:.1f} s\n")
        f.write(f"RMSE final (RGB)        : {rmse_final:.4f}\n")
        f.write(f"\nRepartition des tailles de blocs (3 canaux) :\n")
        for sz in sorted(counts):
            pct = 100.0 * counts[sz] / n_codes
            f.write(f"  {sz:4d} px : {counts[sz]:6d} blocs ({pct:.1f}%)\n")
    print(f"  Stats : {filepath}")
    print(f"  Taux de compression : {ratio:.2f}:1  ({fif_bytes/1024:.1f} Ko / {raw_bytes/1024:.0f} Ko bruts)")



# ─────────────────────────────────────────────────────────────────────────────
# 4.  Génération des figures
# ─────────────────────────────────────────────────────────────────────────────

def _block_stats_str(codes):
    counts = Counter(c["size"] for c in codes)
    return "  ".join(f"{counts[s]}×{s}px" for s in sorted(counts))


def _draw_quadtree_grid(ax, shape_hw, codes):
    h, w = shape_hw
    for code in codes:
        ax.add_patch(mpatches.Rectangle(
            (code["rx"] - 0.5, code["ry"] - 0.5), code["size"], code["size"],
            linewidth=0.3, edgecolor="black", facecolor="white", zorder=3))
    ax.set_xlim(-0.5, w - 0.5)
    ax.set_ylim(h - 0.5, -0.5)


def generate_comparison_figure(name, orig, states, timings, final_it, codes,
                                compress_time_s, output_path):
    """Figure 3×2 : original | convergé | étapes 1,2,3 | grille quadtree (couleur)."""
    t0     = time.perf_counter()
    h_img, w_img = orig.shape[:2]

    layout = [
        (0, 0, ("img", None)),
        (0, 1, ("img", final_it)),
        (1, 0, ("img", 1)),
        (1, 1, ("grid",)),
        (2, 0, ("img", 2)),
        (2, 1, ("img", 3)),
    ]

    fig, axes = plt.subplots(3, 2, figsize=(6.4, 9.6), dpi=200,
                             gridspec_kw={"hspace": 0.35, "wspace": 0.05})
    fig.suptitle(f"Compression quadtree PIFS couleur - {name}\n"
                 f"seuil={ERROR_THRESHOLD:.0f}  |  {len(codes)} blocs (canal vert)  |  "
                 f"resolution {w_img}x{h_img}px",
                 fontsize=9, color="#1f2937", y=0.99)

    orig_u8 = np.clip(orig, 0, 255).astype(np.uint8)

    for row, col, content in layout:
        ax = axes[row, col]
        if content[0] == "img":
            it = content[1]
            if it is None:
                ax.imshow(orig_u8)
                ax.set_title(f"Originale ({w_img}x{h_img}px)", fontsize=8,
                             color="#1f2937", pad=3)
            elif it in states:
                ax.imshow(states[it])
                rmse  = np.sqrt(np.mean(
                    (orig.astype(np.float32) - states[it].astype(np.float32)) ** 2))
                t_ms  = timings.get(it, 0.0)
                t_str = f"{t_ms/1000.:.2f}s" if t_ms >= 1000. else f"{t_ms:.0f}ms"
                suf   = " (conv.)" if it == final_it else ""
                ax.set_title(f"Etape {it}{suf}  RMSE {rmse:.1f}  {t_str}",
                             fontsize=7, color="#1f2937", pad=3)
            else:
                ax.axis("off")
                continue
        else:  # grille quadtree
            ax.imshow(np.full((h_img, w_img), 255, dtype=np.float32),
                      cmap="gray", vmin=0, vmax=255)
            _draw_quadtree_grid(ax, (h_img, w_img), codes)
            ax.set_title("Arbre quaternaire (canal vert)", fontsize=8,
                         color="#1f2937", pad=3)
        ax.axis("off")

    caption = (f"Compression : {compress_time_s:.1f}s  |  {_block_stats_str(codes)}"
               "  |  temps : cumules depuis etape 0")
    fig.text(0.5, 0.005, caption, ha="center", fontsize=6.5, color="#374151")
    timed_savefig(plt, output_path, t0, bbox_inches="tight", pad_inches=0.12)
    plt.close()
    print(f"  Sauvegarde : {output_path}")


def generate_grid_overview(examples_data, output_path):
    """Figure côte à côte : arbre quaternaire des deux images."""
    t0 = time.perf_counter()
    n  = len(examples_data)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5), dpi=200)
    if n == 1:
        axes = [axes]
    for ax, (name, orig, codes) in zip(axes, examples_data):
        h_i, w_i = orig.shape[:2]
        ax.imshow(np.full((h_i, w_i), 255, dtype=np.float32), cmap="gray",
                  vmin=0, vmax=255)
        _draw_quadtree_grid(ax, (h_i, w_i), codes)
        ax.set_title(f"{name}  ({w_i}x{h_i})  {len(codes)} blocs",
                     fontsize=9, color="#1f2937")
        ax.axis("off")
    plt.suptitle(f"Partitions quadtree adaptatives (seuil={ERROR_THRESHOLD:.0f})",
                 fontsize=10, color="#1f2937")
    plt.tight_layout()
    timed_savefig(plt, output_path, t0, bbox_inches="tight", pad_inches=0.1)
    plt.close()
    print(f"  Sauvegarde : {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Pipeline principal
# ─────────────────────────────────────────────────────────────────────────────

def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    SOURCES_DIR.mkdir(parents=True, exist_ok=True)

    config = {
        "min_block_size":  MIN_BLOCK_SIZE,
        # max_block_size sera calcule apres recadrage
        "error_threshold": ERROR_THRESHOLD,
        "domain_step":     DOMAIN_STEP,
    }

    grid_data = []

    for img_path, name in zip(IMG_PATHS, NAMES):
        print(f"\n{'='*60}")
        print(f"  {name.upper()} - {img_path.name}")
        print(f"{'='*60}")

        # 1. Recadrage centre a la plus haute puissance de 2 (RGB)
        cropped_path = SOURCES_DIR / f"{name}_cropped.png"
        img_rgb = center_crop_to_power_of_two(img_path, cropped_path)
        orig_h, orig_w = img_rgb.shape[:2]

        # max_block_size dynamique : plus grande puissance de 2 t.q. domaines (2B) rentrent
        dyn_max = max_block_for_image(orig_h, orig_w)
        config["max_block_size"] = dyn_max
        print(f"  max_block_size calcule : {dyn_max} px (image {orig_w}x{orig_h})")

        # 2. Compression quadtree des 3 canaux
        import pickle
        ckpt_path = FIGURES_DIR / f"{name}_codes.pkl"
        if ckpt_path.exists():
            print(f"  Checkpoint trouve, chargement : {ckpt_path.name}")
            with open(ckpt_path, "rb") as f:
                channels_data, compress_time = pickle.load(f)
            n_codes = sum(len(cd["codes"]) for cd in channels_data)
            print(f"  {n_codes} blocs (3 canaux, depuis checkpoint)")
        else:
            print(f"\n  Compression couleur (seuil={ERROR_THRESHOLD:.0f}, max_bloc={dyn_max}px)...")
            t0 = time.time()
            channels_data = compress_rgb(img_rgb, config)
            compress_time = time.time() - t0
            n_codes = sum(len(cd["codes"]) for cd in channels_data)
            print(f"  {n_codes} blocs (3 canaux) en {compress_time:.2f}s")
            with open(ckpt_path, "wb") as f:
                pickle.dump((channels_data, compress_time), f)
            print(f"  Checkpoint sauvegarde : {ckpt_path.name}")

        # canal de reference (vert) pour les dimensions et la grille quadtree
        oh = channels_data[1].get("orig_h", orig_h)
        ow = channels_data[1].get("orig_w", orig_w)
        green_codes = channels_data[1]["codes"]
        grid_data.append((name, img_rgb[:oh, :ow], green_codes))

        # 2b. Sauvegarde .fif (un par canal) + stats
        for ci, cd in enumerate(channels_data):
            fif_path = FIGURES_DIR / f"{name}_{CHANNEL_NAMES[ci]}.fif"
            fif_size = save_fif(fif_path, cd)
            print(f"  FIF {CHANNEL_NAMES[ci]} : {fif_path.name}  ({fif_size/1024:.1f} Ko)")
        stats_path = FIGURES_DIR / f"{name}_stats.txt"

        # 3. Decompression couleur (Banach, 3 canaux recombines)
        print()
        states, timings, final_it = decompress_rgb(channels_data)
        states = {k: v[:oh, :ow] for k, v in states.items()}

        orig_crop = img_rgb[:oh, :ow]
        rmse_final = float(np.sqrt(np.mean(
            (orig_crop.astype(np.float32) - states[final_it].astype(np.float32)) ** 2)))
        print(f"  RMSE final RGB (etape {final_it}): {rmse_final:.4f}")

        save_stats(stats_path, name, channels_data, ow, oh,
                   compress_time, rmse_final, dyn_max)

        # 4. Figure de comparaison
        out_fig = FIGURES_DIR / f"{name}_quadtree_comparison.png"
        generate_comparison_figure(
            name, orig_crop, states, timings, final_it,
            green_codes, compress_time, out_fig)

    # Vue d'ensemble des deux arbres quaternaires
    generate_grid_overview(grid_data,
                           FIGURES_DIR / "soeurette_quadtrees_grid.png")

    print("\nTermine.")


if __name__ == "__main__":
    main()
