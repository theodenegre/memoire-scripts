# Scripts du mémoire pour les fractales et la compression d'images

Ce dossier est **autonome** : il contient tous les scripts Python
nécessaires à la génération des figures du mémoire, ainsi que les
bibliothèques communes, sans dépendance au reste du projet LaTeX.

## Structure

```
scripts/
├── requirements.txt          # Dépendances Python
├── run.py                    # Lanceur global (tous chapitres ou un seul)
├── figmeta.py                # Utilitaire de sauvegarde avec métadonnées PNG
├── ifs_common/               # Bibliothèque partagée (AffineMap, Hutchinson, 3D)
├── ch3_ensemble_fractales/   # Cantor, IFS 1D
├── ch4_proprietes_hx/        # Chemin Cantor-Sierpinski
├── ch5_transformations/      # Fougère, Sierpinski, Menger, Koch…
└── ch6_collage_compression/  # PIFS, compression fractale, capybara
```

Les figures sont générées dans `Mémoire/figures/<chapitre>/` par rapport
à la racine du dépôt (détection automatique via `Path(__file__)`).

## Installation

```bash
# Depuis la racine du dépôt (Memoire/)
python -m venv .venv
.venv\Scripts\activate          # Windows
# ou : source .venv/bin/activate  (Linux/macOS)
pip install -r scripts/requirements.txt
```

## Générer les figures

```bash
# Toutes les figures
python scripts/run.py --chapter all

# Un seul chapitre
python scripts/run.py --chapter ch5

# Voir la liste sans exécuter
python scripts/run.py --chapter all --dry-run
```

## Chapitres

| Dossier                  | Contenu                                            |
|--------------------------|----------------------------------------------------|
| `ch3_ensemble_fractales` | Ensemble de Cantor, IFS 1D par étapes              |
| `ch4_proprietes_hx`      | Chemin fractal Cantor × Sierpinski                 |
| `ch5_transformations`    | Fougère, Sierpinski 2D/3D, Menger, Koch, arbre…   |
| `ch6_collage_compression`| PIFS quadtree, compression couleur, capybara       |
