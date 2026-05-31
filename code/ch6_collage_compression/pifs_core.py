"""Cœur de la compression d'images par PIFS fractal (chapitre 6).

Implémente l'algorithme FICANRP (partitionnement adaptatif par quadtree), variante
de BFIC décrite dans le mémoire :

- Partition de l'image en blocs cibles (ranges), subdivision quadtree si l'erreur
  de collage dépasse un seuil.
- Pool de blocs domaines de taille 2B x 2B, sous-échantillonnés à B x B par moyenne
  2x2 (``downsample_half``), conformément à l'algorithme du mémoire.
- Pour chaque (range, domaine, isométrie de D4), ajustement photométrique s, o par
  moindres carrés (``fit_affine``), avec écrêtage |s| <= S_MAX < 1 garantissant la
  contractivité (lemme de contractivité de l'opérateur PIFS).
- Décodage : itération de l'opérateur de collage depuis une image grise/bruit
  jusqu'à l'attracteur (théorème de Banach).

Ce module est aussi importé par ``pifs_quadtree_demo.py``. Il fournit en plus une
CLI compress/decompress et des formats binaires .fif / .ezfif.

Usage :
    python pifs_core.py compress entree.png sortie.fif --error 50
    python pifs_core.py decompress sortie.fif sortie.png --iter 10
"""

import argparse
import json
import struct
import sys
import time
from collections import namedtuple
from pathlib import Path

import numpy as np
from PIL import Image

FractalCode = namedtuple("FractalCode", ["rx", "ry", "size", "dx", "dy", "s", "o", "sym"])

# Écrêtage de contraste strictement < 1 (contractivité de l'opérateur PIFS).
S_MAX = 0.999


def load_image_grayscale(path, max_width=None):
    raw = Image.open(path)
    if raw.mode in ("RGBA", "LA"):
        background = Image.new(raw.mode, raw.size, (255,) * len(raw.mode))
        background.paste(raw, mask=raw.split()[-1])
        img = background.convert("L")
    else:
        img = raw.convert("L")

    if max_width and img.width > max_width:
        ratio = max_width / img.width
        img = img.resize((max_width, int(img.height * ratio)), Image.Resampling.LANCZOS)

    w, h = img.size
    if max_width:
        # Forcer une image carrée max_width×max_width avec padding blanc.
        target = max_width
        if (w, h) != (target, target):
            canvas = Image.new("L", (target, target), 255)
            canvas.paste(img, (0, 0))
            img = canvas
    else:
        # Sans contrainte de taille : arrondir au multiple de 4 supérieur.
        w4 = w if w % 4 == 0 else w + (4 - w % 4)
        h4 = h if h % 4 == 0 else h + (4 - h % 4)
        if (w4, h4) != (w, h):
            canvas = Image.new("L", (w4, h4), 255)
            canvas.paste(img, (0, 0))
            img = canvas
    return np.array(img, dtype=np.float32)


def downsample_half(img):
    """Réduit une image de moitié par moyenne locale 2x2."""
    h, w = img.shape
    return img.reshape(h // 2, 2, w // 2, 2).mean(axis=(1, 3))


def get_symmetries_list(block):
    """Les 8 éléments du groupe diédral D4 appliqués au bloc."""
    return [
        block,
        np.rot90(block, 1),
        np.rot90(block, 2),
        np.rot90(block, 3),
        np.fliplr(block),
        np.flipud(block),
        block.T,
        np.flip(block.T),
    ]


def fit_affine(range_block, domain_block):
    """Ajuste s, o minimisant ||s*domain + o - range||^2 (moindres carrés)."""
    R = range_block.flatten()
    D = domain_block.flatten()
    n = R.size

    sum_R, sum_D = np.sum(R), np.sum(D)
    sum_RD, sum_DD = np.dot(R, D), np.dot(D, D)
    den = n * sum_DD - sum_D * sum_D

    if den == 0:
        s, o = 0.0, sum_R / n
    else:
        s = (n * sum_RD - sum_R * sum_D) / den
        o = (sum_R - s * sum_D) / n

    s = np.clip(s, -S_MAX, S_MAX)
    o = np.clip(o, -255.0, 510.0)
    mse = np.mean((R - (s * D + o)) ** 2)
    return s, o, mse


def _flat_code(rx, ry, size, mean_value):
    return {"rx": rx, "ry": ry, "size": size, "dx": 0, "dy": 0, "s": 0.0, "o": float(mean_value), "sym": 0}


def _best_match(img, r_block, rx, ry, r_size, domain_step_factor):
    """Cherche dans le pool de domaines 2B x 2B le meilleur (domaine, isométrie, s, o)."""
    h, w = img.shape
    d_size = 2 * r_size
    step = max(1, int(r_size * domain_step_factor))

    best_code, best_error = None, float("inf")
    for dy in range(0, h - d_size + 1, step):
        for dx in range(0, w - d_size + 1, step):
            d_block = downsample_half(img[dy:dy + d_size, dx:dx + d_size])
            for sym_idx, transformed_d in enumerate(get_symmetries_list(d_block)):
                s, o, error = fit_affine(r_block, transformed_d)
                if error < best_error:
                    best_error = error
                    best_code = FractalCode(rx, ry, r_size, dx, dy, s, o, sym_idx)
    return best_code, best_error


def compress_image(image_path, config):
    from tqdm import tqdm

    print(f"Chargement de {image_path}...")
    img = load_image_grayscale(image_path, config.get("max_width", 256))
    h, w = img.shape
    print(f"Taille : {w}x{h}")

    min_size = config.get("min_block_size", 4)
    max_size = config.get("max_block_size", 16)
    error_threshold = config.get("error_threshold", 20.0)
    domain_step_factor = config.get("domain_step", 1.0)

    queue = [
        (x, y, max_size)
        for y in range(0, h, max_size)
        for x in range(0, w, max_size)
        if y + max_size <= h and x + max_size <= w
    ]

    codes = []
    start_time = time.time()
    pbar = tqdm(desc="Compression", unit="blocs", dynamic_ncols=True)

    while queue:
        rx, ry, r_size = queue.pop(0)
        r_block = img[ry:ry + r_size, rx:rx + r_size]

        if np.max(r_block) - np.min(r_block) < 0.1:
            codes.append(_flat_code(rx, ry, r_size, np.mean(r_block)))
            pbar.update(1)
            continue

        best_code, best_error = _best_match(img, r_block, rx, ry, r_size, domain_step_factor)

        if best_error > error_threshold and r_size > min_size:
            half = r_size // 2
            queue += [(rx, ry, half), (rx + half, ry, half), (rx, ry + half, half), (rx + half, ry + half, half)]
            continue

        codes.append(best_code._asdict() if best_code else _flat_code(rx, ry, r_size, np.mean(r_block)))
        pbar.update(1)
        pbar.set_postfix({"queue": len(queue), "taille": r_size})

    pbar.close()
    print(f"Compression terminée. {len(codes)} codes en {time.time() - start_time:.2f}s")
    return {"width": w, "height": h, "codes": codes}


def decompress_image(data, iterations=10):
    w, h = data["width"], data["height"]
    codes = data["codes"]
    current_img = np.full((h, w), 128.0, dtype=np.float32)

    print(f"Décompression {w}x{h}, {len(codes)} codes, {iterations} itérations...")
    for i in range(iterations):
        print(f"Itération {i + 1}...")
        next_img = np.zeros_like(current_img)
        for code in codes:
            rx, ry, size = code["rx"], code["ry"], code["size"]
            d_block = downsample_half(current_img[code["dy"]:code["dy"] + 2 * size, code["dx"]:code["dx"] + 2 * size])
            d_transformed = get_symmetries_list(d_block)[code["sym"]]
            next_img[ry:ry + size, rx:rx + size] = code["s"] * d_transformed + code["o"]
        current_img = next_img

    return np.clip(current_img, 0, 255).astype(np.uint8)


def save_ezfif(filepath, data):
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write("# EZFIF Fractal Image Code\n")
        f.write(f"# Resolution: {int(data['width'])}x{int(data['height'])}\n")
        f.write("# Format: rx ry size sym dx dy s o\n")
        for c in data["codes"]:
            f.write(f"{c['rx']} {c['ry']} {c['size']} {c['sym']} {c['dx']} {c['dy']} {c['s']:.6f} {c['o']:.6f}\n")


def load_ezfif(filepath):
    width = height = 0
    codes = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("#"):
                if "Resolution:" in line:
                    w_str, h_str = line.split("Resolution:")[1].strip().split("x")
                    width, height = int(w_str), int(h_str)
                continue
            parts = line.split()
            if len(parts) < 8:
                continue
            codes.append({
                "rx": int(parts[0]), "ry": int(parts[1]), "size": int(parts[2]), "sym": int(parts[3]),
                "dx": int(parts[4]), "dy": int(parts[5]), "s": float(parts[6]), "o": float(parts[7]),
            })
    return {"width": width, "height": height, "codes": codes}


def save_fif(filepath, data):
    filepath = Path(filepath)
    if filepath.suffix.lower() == ".ezfif":
        save_ezfif(filepath, data)
        return

    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "wb") as f:
        f.write(b"FIF\x01")
        f.write(struct.pack("<HH", int(data["width"]), int(data["height"])))
        f.write(struct.pack("<I", len(data["codes"])))
        for c in data["codes"]:
            f.write(struct.pack("<HHBBHHff", int(c["rx"]), int(c["ry"]), int(c["size"]),
                                int(c["sym"]), int(c["dx"]), int(c["dy"]), float(c["s"]), float(c["o"])))


def load_fif(filepath):
    filepath = Path(filepath)
    with open(filepath, "rb") as f:
        if f.read(4) != b"FIF\x01":
            return load_ezfif(filepath)
        width, height = struct.unpack("<HH", f.read(4))
        (num_codes,) = struct.unpack("<I", f.read(4))
        codes = []
        for _ in range(num_codes):
            chunk = f.read(18)
            if len(chunk) < 18:
                break
            rx, ry, size, sym, dx, dy, s, o = struct.unpack("<HHBBHHff", chunk)
            codes.append({"rx": int(rx), "ry": int(ry), "size": int(size), "sym": int(sym),
                          "dx": int(dx), "dy": int(dy), "s": float(s), "o": float(o)})
    return {"width": width, "height": height, "codes": codes}


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def _parse_args(argv):
    parser = argparse.ArgumentParser(description="Compresseur d'images par fractales (PIFS quadtree).")
    sub = parser.add_subparsers(dest="command", required=True)

    comp = sub.add_parser("compress")
    comp.add_argument("input")
    comp.add_argument("output")
    comp.add_argument("--max-width", type=int, default=256)
    comp.add_argument("--error", type=float, default=50.0)

    decomp = sub.add_parser("decompress")
    decomp.add_argument("input")
    decomp.add_argument("output")
    decomp.add_argument("--iter", type=int, default=10)
    return parser.parse_args(argv)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if not argv:
        print("Compresseur d'images par fractales (CLI). Utilisez --help.")
        return

    args = _parse_args(argv)
    if args.command == "compress":
        config = {"max_width": args.max_width, "error_threshold": args.error,
                  "min_block_size": 4, "max_block_size": 32, "domain_step": 0.5}
        result = compress_image(args.input, config)
        output_path = Path(args.output)
        if output_path.suffix.lower() in (".fif", ".ezfif"):
            save_fif(output_path, result)
        else:
            with open(args.output, "w") as f:
                json.dump(result, f, cls=NumpyEncoder)
        print(f"Sauvegardé : {args.output}")

    elif args.command == "decompress":
        input_path = Path(args.input)
        if input_path.suffix.lower() in (".fif", ".ezfif"):
            data = load_fif(input_path)
        else:
            with open(args.input, "r") as f:
                data = json.load(f)
        Image.fromarray(decompress_image(data, args.iter)).save(args.output)
        print(f"Sauvegardé : {args.output}")


if __name__ == "__main__":
    main()
