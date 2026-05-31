# Scripts - Chapitre 5 (Transformations)

## Usage rapide

```bash
C:/Users/theod/Desktop/coding/Memoire/.venv/Scripts/python.exe Memoire/code/ch5_transformations/ifs_labels.py
C:/Users/theod/Desktop/coding/Memoire/.venv/Scripts/python.exe Memoire/code/ch5_transformations/fractales_etapes.py
C:/Users/theod/Desktop/coding/Memoire/.venv/Scripts/python.exe Memoire/code/ch5_transformations/fougere_poly.py
```

## Sorties

Les images sont ecrites dans:
- Memoire/figures/ch5_transformations/
- Memoire/figures/ch5_transformations/steps/
- Memoire/figures/ch5_transformations/fern_steps_1_14/
- Memoire/figures/ch5_transformations/fern_singleton/

## Scripts (tous deterministes : iteration de l'operateur de Hutchinson)

- ifs_labels.py: IFS simples annotes (carre, union de deux intervalles).
- fractales_etapes.py: Cantor, Koch, Sierpinski, tapis, arbre (etapes).
- fougere_poly.py: Fougere de Barnsley par iteration polygonale (figure du memoire).
- fougere_singleton.py: Fougere a partir d'un singleton (n=0..14).
- fougere_condensation.py: Comparaison fougere avec / sans condensation.
- sierpinski_2d.py: Triangle de Sierpinski (deux ensembles initiaux).
- sierpinski_nonlinear.py: Variante non lineaire du Sierpinski (figure sierpinski_nonlinear_n10).
- menger.py: Eponge de Menger (R^3).
- sierpinski_3d.py: Tetraedre de Sierpinski (R^3).
- produit_cantor_koch.py: Produit cartesien Cantor x Koch (R^3).
