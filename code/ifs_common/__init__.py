"""Code commun aux scripts d'illustration des SFI (chapitres 3 à 6).

Regroupe les briques réutilisées par plusieurs scripts :

- ``affine``   : transformation affine 2D ``AffineMap`` et représentation par dict.
- ``hutchinson`` : itération déterministe de l'opérateur de Hutchinson
  (polygones 2D, nuages de points 2D, solides 3D).
- ``plotting3d`` : rendu commun des solides 3D colorés par hauteur.

Les scripts ajoutent ``code/`` au ``sys.path`` (comme pour ``figmeta``) puis
importent, par exemple, ``from ifs_common.affine import AffineMap``.
"""
