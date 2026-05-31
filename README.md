# Scripts du mémoire pour les fractales et la compression d'images

## Structure du dépôt

```
├── requirements.txt          # Dépendances Python
├── run.py                    # Lanceur global de génération des figures
├── figures/
│   └── sources/              # Images sources requises (chat, fougère, Sierpinski)
└── code/
    ├── figmeta.py            # Métadonnées PNG
    ├── ifs_common/           # Bibliothèque partagée (AffineMap, Hutchinson)
    ├── ch3_ensemble_fractales/   # Cantor, IFS 1D par étapes
    ├── ch4_proprietes_hx/        # Chemin Cantor-Sierpinski
    ├── ch5_transformations/      # Fougère, Sierpinski 2D/3D, Menger, Koch
    └── ch6_collage_compression/  # PIFS, compression couleur, capybara
```

Les figures générées sont enregistrées dans un dossier `figures/` créé à la racine du dépôt.

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
# ou : source .venv/bin/activate  (Linux/macOS)
pip install -r requirements.txt
```

## Génération des figures

```bash
# Générer toutes les figures
python run.py --chapter all

# Générer les figures d'un chapitre spécifique
python run.py --chapter ch5

# Lister les scripts sans les exécuter
python run.py --chapter all --dry-run
```

## Description des chapitres

| Dossier | Contenu |
| :--- | :--- |
| `code/ch3_ensemble_fractales` | Construction de l'ensemble de Cantor et illustrations d'IFS 1D |
| `code/ch4_proprietes_hx` | Chemin de transition de Cantor vers Sierpinski |
| `code/ch5_transformations` | Fougère de Barnsley (méthode polygonale déterministe et condensée), triangle et tétraèdre de Sierpinski, éponge de Menger, produit Cantor-Koch |
| `code/ch6_collage_compression` | Compression fractale (PIFS) quadtree (chat, fougère, Sierpinski), compression couleur multicanale, partitionnement binaire géométrique sur le capybara |
