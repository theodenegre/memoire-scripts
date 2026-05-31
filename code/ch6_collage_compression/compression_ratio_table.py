"""Tableau d'arbitrage taux de compression / fidélité selon le seuil (chapitre 6).

Compresse l'image du chat par l'algorithme FICANRP (``pifs_quadtree_demo``) pour
plusieurs seuils de tolérance, mesure pour chacun :

- le nombre de blocs ranges retenus,
- la taille réelle du fichier ``.fif`` sérialisé (``pifs_core.save_fif``),
- le taux de compression (octets bruts / octets ``.fif``),
- la RMSE finale entre l'original et l'attracteur reconstruit.

Le tableau LaTeX correspondant est écrit dans
``figures/ch6_collage_compression/compression_ratio_table.tex`` et inclus par
``\input`` dans le chapitre. Les valeurs proviennent donc directement des scripts
de compression, sans recopie manuelle.

Usage :
    python compression_ratio_table.py
"""

import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pifs_core import load_image_grayscale, save_fif
from pifs_quadtree_demo import compress_ficanrp_fast, decompress_ficanrp, SOURCES_DIR

OUTPUT_DIR = Path(__file__).resolve().parents[2] / "figures" / "ch6_collage_compression"
SEUILS = [20.0, 50.0, 100.0, 200.0, 400.0]
BASE_CONFIG = {"max_width": 256, "min_block_size": 2, "max_block_size": 128, "domain_step": 2.0}


def _fif_size_bytes(codes_data):
    """Taille réelle du fichier .fif sérialisé par le format binaire du mémoire."""
    with tempfile.NamedTemporaryFile(suffix=".fif", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        save_fif(tmp_path, codes_data)
        return tmp_path.stat().st_size
    finally:
        tmp_path.unlink(missing_ok=True)


def _fr(x, decimals):
    """Formate un nombre à la française (virgule décimale) pour le tableau."""
    return f"{x:.{decimals}f}".replace(".", "{,}")


def measure(img):
    rows = []
    for theta in SEUILS:
        config = {**BASE_CONFIG, "error_threshold": theta}
        data = compress_ficanrp_fast(img, config)
        codes = data["codes"]
        oh, ow = data.get("orig_h", img.shape[0]), data.get("orig_w", img.shape[1])

        raw_bytes = ow * oh
        fif_bytes = _fif_size_bytes(data)
        ratio = raw_bytes / fif_bytes

        states, _, final_it = decompress_ficanrp(data, max_it=20)
        rec = states[final_it][:oh, :ow]
        rmse = float(np.sqrt(np.mean((img[:oh, :ow] - rec) ** 2)))

        rows.append({"theta": theta, "blocs": len(codes), "ratio": ratio, "rmse": rmse})
        print(f"theta={theta:6.0f}  blocs={len(codes):5d}  fif={fif_bytes:6d}o  "
              f"ratio={ratio:5.2f}:1  RMSE={rmse:6.2f}")
    return rows


def write_table(rows, raw_bytes, output_path):
    lines = [
        "% Tableau généré par code/ch6_collage_compression/compression_ratio_table.py",
        "% Ne pas éditer à la main : relancer le script pour mettre à jour.",
        r"\begin{tabular}{|c|c|c|c|}",
        r"\hline",
        r"\textbf{Seuil $\theta$} & \textbf{Blocs $N$} & \textbf{Taux de compression} & \textbf{RMSE} \\",
        r"\hline",
    ]
    for r in rows:
        lines.append(
            f"${r['theta']:.0f}$ & ${r['blocs']}$ & ${_fr(r['ratio'], 2)} : 1$ & ${_fr(r['rmse'], 2)}$ \\\\"
        )
        lines.append(r"\hline")
    lines.append(r"\end{tabular}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nTableau écrit : {output_path}")


def main():
    img = load_image_grayscale(str(SOURCES_DIR / "cat_source.png"), BASE_CONFIG["max_width"])
    oh, ow = img.shape
    rows = measure(img)
    write_table(rows, ow * oh, OUTPUT_DIR / "compression_ratio_table.tex")


if __name__ == "__main__":
    main()
