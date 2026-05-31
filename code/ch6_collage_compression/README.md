# Scripts - Chapitre 6 (Collage et compression)

## Usage rapide

```bash
C:/Users/theod/Desktop/coding/Memoire/.venv/Scripts/python.exe Memoire/code/ch6_collage_compression/collage_L.py
C:/Users/theod/Desktop/coding/Memoire/.venv/Scripts/python.exe Memoire/code/ch6_collage_compression/pifs_core.py compress <input> <output.fif>
C:/Users/theod/Desktop/coding/Memoire/.venv/Scripts/python.exe Memoire/code/ch6_collage_compression/pifs_core.py decompress <input.fif> <output.png>
```

## Sorties

Les images sont écrites dans:
- Memoire/figures/ch6_collage_compression/

## Scripts

- pifs_core.py: Cœur PIFS (FICANRP quadtree) + CLI compress/decompress + formats .fif/.ezfif.
- pifs_quadtree_demo.py: Décompression de fougère, Sierpiński et chat (figures du mémoire).
- pifs_couleur.py: PIFS couleur canal par canal (R/G/B).
- pifs_decode_pentagone.py: Décodage PIFS du pentagone de Sierpiński depuis un bruit.
- collage_L.py: Théorème du collage (parfait vs imparfait) sur une forme en L.
- isometries_d4.py: Les 8 isométries du groupe diédral D4.
- capybara_binaire.py: Partitionnement PIFS binaire (capybara 32x32).
- compression_ratio_table.py: Génère le tableau des taux de compression (.tex).
