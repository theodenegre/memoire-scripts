"""Utilitaire commun : sauvegarde de figures avec métadonnées de génération.

Inscrit dans chaque PNG le temps de génération de la figure ainsi que des champs
descriptifs, via les chunks tEXt PNG (lisibles avec ``PIL.Image.open(p).text`` ou
``exiftool``). Clés écrites :

    Software        : "Memoire IFS"
    Generated       : date ISO 8601 (heure locale)
    Generation-Time : "<secondes> s" (temps écoulé depuis ``started_at``)

Exemple :
    import time
    from figmeta import timed_savefig
    t0 = time.perf_counter()
    fig, ax = plt.subplots()
    ...
    timed_savefig(fig, chemin, t0, dpi=200, bbox_inches="tight")
"""

import time
from datetime import datetime

SOFTWARE = "Memoire IFS"


def generation_metadata(started_at, extra=None):
    """Construit le dict de métadonnées PNG pour une figure démarrée à ``started_at``.

    ``started_at`` est une valeur ``time.perf_counter()`` prise au début de la
    construction de la figure.
    """
    metadata = {
        "Software": SOFTWARE,
        "Generated": datetime.now().isoformat(timespec="seconds"),
        "Generation-Time": f"{time.perf_counter() - started_at:.3f} s",
    }
    if extra:
        metadata.update(extra)
    return metadata


def pnginfo(started_at, extra=None):
    """Construit un ``PIL.PngImagePlugin.PngInfo`` portant les mêmes métadonnées.

    À passer à ``Image.save(path, pnginfo=...)`` pour les images produites via PIL.
    """
    from PIL import PngImagePlugin

    info = PngImagePlugin.PngInfo()
    for key, value in generation_metadata(started_at, extra=extra).items():
        info.add_text(key, value)
    return info


def timed_savefig(target, path, started_at, *, extra=None, **savefig_kwargs):
    """Sauvegarde ``target`` (Figure ou module pyplot) dans ``path`` avec métadonnées.

    Renvoie le temps de génération en secondes.
    """
    metadata = generation_metadata(started_at, extra=extra)
    user_meta = savefig_kwargs.pop("metadata", None)
    if user_meta:
        metadata.update(user_meta)
    target.savefig(path, metadata=metadata, **savefig_kwargs)
    return time.perf_counter() - started_at
