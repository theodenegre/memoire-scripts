"""Génère les figures du mémoire.

Usage :
    python run.py --chapter ch5
    python run.py --chapter all
    python run.py --chapter all --dry-run
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Iterable

# Racine du dépôt (contenant run.py)
REPO_ROOT = Path(__file__).resolve().parent
CODE_DIR = REPO_ROOT / "code"

CHAPTER_SCRIPTS: dict[str, list[Path]] = {
    "ch3": [
        CODE_DIR / "ch3_ensemble_fractales" / "cantor_etapes.py",
        CODE_DIR / "ch3_ensemble_fractales" / "ifs_1d_etapes.py",
    ],
    "ch4": [
        CODE_DIR / "ch4_proprietes_hx" / "chemin_cantor_sierpinski.py",
    ],
    "ch5": [
        CODE_DIR / "ch5_transformations" / "ifs_labels.py",
        CODE_DIR / "ch5_transformations" / "fractales_etapes.py",
        CODE_DIR / "ch5_transformations" / "fougere_poly.py",
        CODE_DIR / "ch5_transformations" / "fougere_singleton.py",
        CODE_DIR / "ch5_transformations" / "fougere_condensation.py",
        CODE_DIR / "ch5_transformations" / "menger.py",
        CODE_DIR / "ch5_transformations" / "sierpinski_2d.py",
        CODE_DIR / "ch5_transformations" / "sierpinski_3d.py",
        CODE_DIR / "ch5_transformations" / "produit_cantor_koch.py",
        CODE_DIR / "ch5_transformations" / "sierpinski_nonlinear.py",
    ],
    "ch6": [
        CODE_DIR / "ch6_collage_compression" / "collage_L.py",
        CODE_DIR / "ch6_collage_compression" / "pifs_quadtree_demo.py",
        CODE_DIR / "ch6_collage_compression" / "compression_ratio_table.py",
        CODE_DIR / "ch6_collage_compression" / "pifs_core.py",
        CODE_DIR / "ch6_collage_compression" / "capybara_binaire.py",
        CODE_DIR / "ch6_collage_compression" / "pifs_couleur.py",
        CODE_DIR / "ch6_collage_compression" / "pifs_decode_pentagone.py",
        CODE_DIR / "ch6_collage_compression" / "isometries_d4.py",
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Générer les figures par chapitre.")
    parser.add_argument(
        "--chapter",
        choices=["ch3", "ch4", "ch5", "ch6", "all"],
        default="all",
        help="Chapitre à régénérer (défaut : all).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Affiche les scripts sans les exécuter.",
    )
    return parser.parse_args()


def iter_scripts(chapter: str) -> Iterable[Path]:
    if chapter == "all":
        for key in ["ch3", "ch4", "ch5", "ch6"]:
            yield from CHAPTER_SCRIPTS[key]
    else:
        yield from CHAPTER_SCRIPTS[chapter]


def run_script(script_path: Path, dry_run: bool) -> None:
    if not script_path.exists():
        print(f"  [SKIP] Introuvable : {script_path.relative_to(REPO_ROOT)}")
        return

    cmd = [sys.executable, str(script_path)]
    print(f"\n==> {' '.join(cmd)}")
    if dry_run:
        return

    result = subprocess.run(cmd, cwd=REPO_ROOT, check=False, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Erreur pour {script_path.name} (code {result.returncode})")


def main() -> None:
    args = parse_args()
    for script in iter_scripts(args.chapter):
        run_script(script, args.dry_run)
    print("\nTerminé.")


if __name__ == "__main__":
    main()
