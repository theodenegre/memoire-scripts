"""Génère les figures du mémoire.

Usage :
    python scripts/run.py --chapter ch5
    python scripts/run.py --chapter all
    python scripts/run.py --chapter all --dry-run
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Iterable

# Racine du dépôt (parent de scripts/)
REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = Path(__file__).resolve().parent

CHAPTER_SCRIPTS: dict[str, list[Path]] = {
    "ch3": [
        SCRIPTS_DIR / "ch3_ensemble_fractales" / "cantor_etapes.py",
        SCRIPTS_DIR / "ch3_ensemble_fractales" / "ifs_1d_etapes.py",
    ],
    "ch4": [
        SCRIPTS_DIR / "ch4_proprietes_hx" / "chemin_cantor_sierpinski.py",
    ],
    "ch5": [
        SCRIPTS_DIR / "ch5_transformations" / "ifs_labels.py",
        SCRIPTS_DIR / "ch5_transformations" / "fractales_etapes.py",
        SCRIPTS_DIR / "ch5_transformations" / "fougere_poly.py",
        SCRIPTS_DIR / "ch5_transformations" / "fougere_singleton.py",
        SCRIPTS_DIR / "ch5_transformations" / "fougere_condensation.py",
        SCRIPTS_DIR / "ch5_transformations" / "menger.py",
        SCRIPTS_DIR / "ch5_transformations" / "sierpinski_2d.py",
        SCRIPTS_DIR / "ch5_transformations" / "sierpinski_3d.py",
        SCRIPTS_DIR / "ch5_transformations" / "produit_cantor_koch.py",
        SCRIPTS_DIR / "ch5_transformations" / "sierpinski_nonlinear.py",
    ],
    "ch6": [
        SCRIPTS_DIR / "ch6_collage_compression" / "collage_L.py",
        SCRIPTS_DIR / "ch6_collage_compression" / "pifs_quadtree_demo.py",
        SCRIPTS_DIR / "ch6_collage_compression" / "compression_ratio_table.py",
        SCRIPTS_DIR / "ch6_collage_compression" / "pifs_core.py",
        SCRIPTS_DIR / "ch6_collage_compression" / "capybara_binaire.py",
        SCRIPTS_DIR / "ch6_collage_compression" / "pifs_couleur.py",
        SCRIPTS_DIR / "ch6_collage_compression" / "pifs_decode_pentagone.py",
        SCRIPTS_DIR / "ch6_collage_compression" / "isometries_d4.py",
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
        print(f"  [SKIP] Introuvable : {script_path.relative_to(SCRIPTS_DIR)}")
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
