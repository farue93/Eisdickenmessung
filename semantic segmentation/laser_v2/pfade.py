# -*- coding: utf-8 -*-
"""
pfade.py - zentrale, RELATIVE Pfad-Aufloesung (push-tauglich, keine G:/-Pfade).

Findet die Repo-Wurzel automatisch (der Ordner, der 'frame differencing' UND
'pre processing' enthaelt), egal wo das Repo geklont wurde. Alle laser_v2-Skripte
(auch die in sam/ und unet/) importieren dieses Modul und fragen darueber die
Serie, die Frame-0-Geometrie und die Ergebnis-JSONs ab.

Voraussetzung: die Pipeline (run.py -> laser_pipeline, serie_eis, canny) wurde
einmal ausgefuehrt, damit Geometrie (pre processing/output) und die
gecroppten Frames (frame differencing/serie_*/frames) existieren.
"""
import os, sys, glob, json
import numpy as np


def _repo_root(start):
    d = os.path.abspath(start)
    for _ in range(8):                                   # max. 8 Ebenen nach oben
        if os.path.isdir(os.path.join(d, "frame differencing")) and \
           os.path.isdir(os.path.join(d, "pre processing")):
            return d
        nd = os.path.dirname(d)
        if nd == d:                                      # Dateisystem-Wurzel erreicht
            break
        d = nd
    raise RuntimeError("Repo-Wurzel nicht gefunden (erwarte 'frame differencing' + 'pre processing').")


REPO      = _repo_root(os.path.dirname(os.path.abspath(__file__)))
FRAMEDIFF = os.path.join(REPO, "frame differencing")
CANNY     = os.path.join(REPO, "canny ice detection")
GERUEST   = os.path.join(REPO, "pre processing", "output")
INPUT     = os.path.join(REPO, "input")
if FRAMEDIFF not in sys.path:
    sys.path.insert(0, FRAMEDIFF)                        # damit 'import serie_eis' geht


def serie_dir():
    """Erste 'serie_*' unter 'frame differencing/' mit frames-Unterordner."""
    c = [d for d in sorted(glob.glob(os.path.join(FRAMEDIFF, "serie_*")))
         if os.path.isdir(os.path.join(d, "frames"))]
    if not c:
        raise RuntimeError("Keine 'serie_*' in 'frame differencing/'. Erst run.py (serie_eis) laufen lassen.")
    return c[0]


def frames_dir():   return os.path.join(serie_dir(), "frames")
def suffix():       return os.path.basename(serie_dir()).split("serie_", 1)[-1]
def framediff_json(): return os.path.join(serie_dir(), "data.json")
def canny_json():   return os.path.join(CANNY, "serie_" + suffix(), "data.json")


def stem():
    """Frame-0-Stamm (Original-Dateiname) aus der serie-data.json ableiten."""
    d = json.load(open(framediff_json()))
    return d["frames"][0]["name"]


def geometrie():
    """Laser-Geometrie in CROP-Koordinaten: dict(x,y,outx,outy,s,bbox)."""
    st = stem()
    L = np.load(os.path.join(GERUEST, st + "_laserlinie.npz"))
    R = np.load(os.path.join(GERUEST, st + "_roi.npz"))
    y0, y1, x0, x1 = (int(v) for v in R["bbox"]); vz = float(R["aussen_vorzeichen"])
    return dict(x=L["x"] - x0, y=L["y"] - y0, outx=vz * L["nx"], outy=vz * L["ny"],
                s=L["s"], bbox=(x0, y0, x1, y1))
