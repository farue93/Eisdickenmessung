# -*- coding: utf-8 -*-
"""
mittellinie.py - aus den dick gemalten Handlabels (masks/) die MITTELLINIE ziehen.

Pro Maske: kleine Luecken schliessen -> groesste Komponente -> auf 1 px
skelettieren -> auf gleichmaessige Breite (~3 px) aufdicken = Trainingsziel.
Schreibt masks_train/ (Ziel) + Kontroll-Overlays (Mittellinie gruen auf Bild).
Die Handlabels in masks/ bleiben unangetastet.
"""
import os, glob, cv2
import numpy as np
from skimage.morphology import skeletonize

BASE = os.path.dirname(os.path.abspath(__file__))
FR   = os.path.join(BASE, "frames")
MH   = os.path.join(BASE, "masks")          # Handlabels (dick)
MT   = os.path.join(BASE, "masks_train")    # Trainingsziel (Mittellinie, gleichmaessig)
OVL  = os.path.join(BASE, "kontrolle")
os.makedirs(MT, exist_ok=True); os.makedirs(OVL, exist_ok=True)
ZIEL_BREITE = 3                              # Breite der Trainingslinie [px]
LUECKE      = 15                             # Schliess-Kernel gegen Mini-Luecken [px]
MIN_AREA    = 150                            # Komponenten kleiner als das = Specks (weg)


def echte_linien(m):
    """ALLE hinreichend grossen Komponenten behalten (beide Aeste), Specks weg."""
    n, lab, st, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8))
    keep = np.zeros_like(m, np.uint8)
    for k in range(1, n):
        if st[k, cv2.CC_STAT_AREA] >= MIN_AREA:
            keep[lab == k] = 1
    return keep


for f in sorted(glob.glob(os.path.join(MH, "*.png"))):
    name = os.path.basename(f)
    mb = (cv2.imread(f, 0) > 0).astype(np.uint8)
    if mb.sum() == 0:
        continue
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (LUECKE, LUECKE))
    closed = cv2.morphologyEx(mb, cv2.MORPH_CLOSE, ker)     # kleine Luecken ueberbruecken
    big = echte_linien(closed)                               # beide Aeste behalten, Specks weg
    sk = skeletonize(big > 0).astype(np.uint8)               # 1-px-Mittellinie
    ziel = cv2.dilate(sk, np.ones((ZIEL_BREITE, ZIEL_BREITE), np.uint8))  # gleichmaessig ~3 px
    cv2.imwrite(os.path.join(MT, name), (ziel * 255).astype(np.uint8))

    # Kontroll-Overlay: Hand (dunkelrot) vs. Mittellinie (gruen)
    img = cv2.imread(os.path.join(FR, name), 0)
    vis = cv2.cvtColor(np.clip(img * 1.8, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    vis[mb > 0] = (0, 0, 120)                                # Handlabel dezent rot
    vis[cv2.dilate(sk, np.ones((3, 3), np.uint8)) > 0] = (0, 255, 0)  # Mittellinie gruen
    sc = 1100 / vis.shape[1]
    vis = cv2.resize(vis, (1100, int(vis.shape[0] * sc)), interpolation=cv2.INTER_AREA)
    cv2.imwrite(os.path.join(OVL, f"mid_{name}"), vis)
    print(f"  {name}: Skelett {int(sk.sum())} px -> Ziel {int((ziel>0).sum())} px")

print(f"-> masks_train/ (Trainingsziel) + Overlays mid_*.png")
