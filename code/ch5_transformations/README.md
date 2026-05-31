# Scripts - Chapitre 5 (Transformations)

## Usage rapide

```bash
C:/Users/theod/Desktop/coding/Memoire/.venv/Scripts/python.exe Memoire/code/ch5_transformations/ifs_labels.py
C:/Users/theod/Desktop/coding/Memoire/.venv/Scripts/python.exe Memoire/code/ch5_transformations/fractales_etapes.py
C:/Users/theod/Desktop/coding/Memoire/.venv/Scripts/python.exe Memoire/code/ch5_transformations/fougere_poly.py
```

## Sorties

Les images sont écrites dans:
- Memoire/figures/ch5_transformations/
- Memoire/figures/ch5_transformations/steps/
- Memoire/figures/ch5_transformations/fern_steps_1_14/
- Memoire/figures/ch5_transformations/fern_singleton/

## Scripts (tous déterministes : itération de l'opérateur de Hutchinson)

- ifs_labels.py: IFS simples annotés (carré, union de deux intervalles).
- fractales_etapes.py: Cantor, Koch, Sierpiński, tapis, arbre (étapes).
- fougere_poly.py: Fougère de Barnsley par itération polygonale (figure du mémoire).
- fougere_singleton.py: Fougère à partir d'un singleton (n=0..14).
- fougere_condensation.py: Comparaison fougère avec / sans condensation.
- sierpinski_2d.py: Triangle de Sierpiński (deux ensembles initiaux).
- sierpinski_nonlinear.py: Variante non linéaire du Sierpiński (figure sierpinski_nonlinear_n10).
- menger.py: Éponge de Menger (R^3).
- sierpinski_3d.py: Tétraèdre de Sierpiński (R^3).
- produit_cantor_koch.py: Produit cartésien Cantor x Koch (R^3).
