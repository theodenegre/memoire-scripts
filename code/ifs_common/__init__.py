"""Code commun aux scripts d'illustration des SFI (chapitres 3 a 6).

Regroupe les briques reutilisees par plusieurs scripts :

- ``affine``   : transformation affine 2D ``AffineMap`` et representation par dict.
- ``hutchinson`` : iteration deterministe de l'operateur de Hutchinson
  (polygones 2D, nuages de points 2D, solides 3D).
- ``plotting3d`` : rendu commun des solides 3D colores par hauteur.

Les scripts ajoutent ``code/`` au ``sys.path`` (comme pour ``figmeta``) puis
importent, par exemple, ``from ifs_common.affine import AffineMap``.
"""
