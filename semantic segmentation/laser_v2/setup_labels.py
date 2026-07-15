# -*- coding: utf-8 -*-
"""
setup_labels.py - Label-Workspace vorbereiten (relative Pfade via pfade.py).

  1. 5 Frames der Serie auswaehlen (1x eisfrei + 4 sinnvolle).
  2. Die gecroppten PNGs nach laser_v2/frames/ kopieren.
  3. Vorlabel-Maske je Frame: nur die allerbesten Laserpunkte (pro Station der
     hellste Punkt entlang der Laser-Normale, sofern hell UND klarer Peak).
  4. split.json schreiben (alle 5 = train; bewusst 'auswendig lernen').
"""
import os, json, shutil
import numpy as np
import cv2

import pfade                       # relative Repo-Aufloesung
import serie_eis as se            # (pfade hat 'frame differencing' bereits in den Pfad gelegt)

BASE = os.path.dirname(os.path.abspath(__file__))
SRC  = pfade.frames_dir()          # gecroppte PNGs der Serie
G    = pfade.geometrie()           # Laser-Geometrie in Crop-Koordinaten
x, y, outx, outy = G["x"], G["y"], G["outx"], G["outy"]

SEL      = [0, 12, 28, 40, 47]     # 0=eisfrei, dann Onset/Mitte/spaet/sehr-spaet
SEED_MIN = 205                     # nur SEHR helle Punkte (nahe Saettigung)
MIN_KON  = 60                      # klarer Peak: Helligkeit ueber Hintergrund

os.makedirs(os.path.join(BASE, "frames"), exist_ok=True)
os.makedirs(os.path.join(BASE, "masks"),  exist_ok=True)


def best_punkte(crop):
    """Sparse Maske: pro Station der hellste, eindeutige Laserpunkt (>=SEED_MIN)."""
    t = np.arange(-15, 55, 1.0)
    X = x[:, None] + outx[:, None] * t[None, :]
    Y = y[:, None] + outy[:, None] * t[None, :]
    P = se.bilinear(crop, X, Y)
    m = np.zeros(crop.shape, np.uint8); innen = t < 0; gesetzt = 0
    for i in range(P.shape[0]):
        prof = P[i]
        bg = np.median(prof[innen]) if innen.any() else float(prof.min())
        pk = int(np.argmax(prof))
        if prof[pk] >= SEED_MIN and prof[pk] - bg >= MIN_KON:
            xi = int(round(x[i] + outx[i] * t[pk])); yi = int(round(y[i] + outy[i] * t[pk]))
            if 0 <= xi < crop.shape[1] and 0 <= yi < crop.shape[0]:
                m[yi, xi] = 255; gesetzt += 1
    return m, gesetzt


namen = []
for k in SEL:
    src = os.path.join(SRC, f"{k:04d}.png"); dst = os.path.join(BASE, "frames", f"{k:04d}.png")
    shutil.copy(src, dst)
    crop = cv2.imread(dst, cv2.IMREAD_GRAYSCALE).astype(np.float32)
    m, n = best_punkte(crop)
    cv2.imwrite(os.path.join(BASE, "masks", f"{k:04d}.png"), m)
    namen.append(f"{k:04d}.png")
    print(f"  Frame {k:02d}: {n:4d} Best-Punkte vorbelegt")

json.dump({"train": namen, "val": []}, open(os.path.join(BASE, "split.json"), "w"), indent=1)
print(f"\n{len(namen)} Frames -> laser_v2/frames + vorbelegte Masken in laser_v2/masks")
print("Auswahl:", namen, "(0000 = eisfrei)")
