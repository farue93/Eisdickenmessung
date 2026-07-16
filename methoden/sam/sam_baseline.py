# -*- coding: utf-8 -*-
"""
sam_baseline.py - SAM (Segment Anything) als bestehende KI-Baseline, ZERO-SHOT
(ohne Training) auf den gecroppten Frames der Serie 174444.

Liegt im Unterordner laser_v2/sam/ (nur SAM-Code). Ausgaben (data.json, Overlays)
landen HIER im selben Ordner. Labels/Frames werden NICHT gebraucht (zero-shot).

Drei Betriebsarten:
  A) automatisch  - SAM segmentiert "alles" -> viele Flaechen-Bloecke (Overlay).
  B) Punkt-Prompts - Prompt-Punkte ENTLANG der bekannten Laserlinie -> eine Maske
     -> gleiche Nachbearbeitung wie U-Net (Schwerpunkt entlang Normalen) -> data.json.
  C) enger Crop   - kleiner Ausschnitt um ein Laserstueck (SAMs beste Chance).

Warum SAM scheitert: interne ~1024px-Skalierung (1px-Linie wird sub-pixel) +
Flaechen-, nicht Kurven-Decoder.
"""
import os, sys, glob, json
import numpy as np
import cv2
import torch
from ultralytics import SAM

BASE = os.path.dirname(os.path.abspath(__file__))                   # = laser_v2/sam
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(BASE)), "gemeinsam"))                          # laser_v2 (fuer pfade)
import pfade                                                        # relative Repo-Aufloesung
import serie_eis as se                                             # (pfade legt 'methoden/frame_differencing' in den Pfad)

FRAMES = pfade.frames_dir()
OUT    = BASE                                                       # Ausgaben hierher
N_PROMPT = 25
PX_PER_MM = 13.9

G = pfade.geometrie()
x, y, s = G["x"], G["y"], G["s"]
outx, outy = G["outx"], G["outy"]
x0, y0, x1, y1 = G["bbox"]
idx = np.linspace(0, len(x) - 1, N_PROMPT).astype(int)

model = SAM(os.path.join(BASE, "mobile_sam.pt"))                    # Gewichte liegen hier


def sam_union(res):
    if not res or res[0].masks is None:
        return None
    md = res[0].masks.data.cpu().numpy()
    return ((md.sum(0) > 0).astype(np.uint8) * 255)


def maske_prompt(imgpath):
    pts = [[float(x[i]), float(y[i])] for i in idx]
    res = model(imgpath, points=pts, labels=[1] * len(pts), verbose=False)
    return sam_union(res)


def detektiere(mask):
    t = np.arange(-25, 61, 1.0)
    X = x[:, None] + outx[:, None] * t[None, :]
    Y = y[:, None] + outy[:, None] * t[None, :]
    P = se.bilinear(mask.astype(np.float32), X, Y)
    d = np.full(len(x), np.nan)
    for i in range(len(x)):
        m = P[i] > 127
        if m.any():
            w = P[i][m]; d[i] = float(np.sum(t[m] * w) / np.sum(w))
    return d


# ---- B) Punkt-Prompts ueber die Serie ----
frames = sorted(glob.glob(os.path.join(FRAMES, "*.png")))
sub = slice(None, None, 2); s_exp = se.nan_liste(s[sub])
d0 = None; fdJSON = []
print(f"SAM Punkt-Prompts ueber {len(frames)} Frames ({N_PROMPT} Prompts/Frame) ...")
for k, fp in enumerate(frames):
    m = maske_prompt(fp)
    if m is None:
        m = np.zeros(cv2.imread(fp, 0).shape, np.uint8)
    d = detektiere(m)
    if d0 is None:
        d0 = d.copy()
    dicke = se.median_glatt(d - d0, 9)
    gut = np.isfinite(d)
    ix = np.where(gut, x + outx * np.nan_to_num(d), np.nan)
    iy = np.where(gut, y + outy * np.nan_to_num(d), np.nan)
    fdJSON.append({"file": f"frames/{k:04d}.png", "name": os.path.basename(fp).replace(".png", ""),
                   "ix": se.nan_liste(ix[sub]), "iy": se.nan_liste(iy[sub]), "dicke": se.nan_liste(dicke[sub])})
    if k % 12 == 0:
        print(f"  Frame {k:2d}: SAM-Maske deckt {100.0*(m>0).mean():4.1f}% des Crops  (Laser <1%!)")
json.dump({"crop_w": x1-x0, "crop_h": y1-y0, "px_per_mm": PX_PER_MM, "s": s_exp, "frames": fdJSON},
          open(os.path.join(OUT, "data.json"), "w"))
print(f"-> {OUT}/data.json")


# ---- A) automatisch (Anschauung, 2 Frames) ----
def overlay_auto(fp, name):
    res = model(fp, verbose=False)
    img = cv2.cvtColor(cv2.imread(fp, 0), cv2.COLOR_GRAY2BGR)
    nreg = 0
    if res and res[0].masks is not None:
        md = res[0].masks.data.cpu().numpy(); nreg = md.shape[0]
        rng = np.random.default_rng(0)
        for j in range(nreg):
            img[md[j] > 0] = (0.5 * img[md[j] > 0] + rng.integers(60, 255, 3)).astype(np.uint8)
    cv2.imwrite(os.path.join(OUT, f"auto_{name}.png"), img)
    print(f"  auto_{name}.png: {nreg} SAM-Regionen")

print("SAM automatisch (Anschauung) ...")
for k in (0, 40):
    overlay_auto(frames[k], f"{k:04d}")


# ---- C) enger Crop ----
def overlay_engcrop(fp, name):
    img = cv2.imread(fp, 0); i = len(x) // 2
    cxp, cyp = int(x[i]), int(y[i]); h = 160
    xa, xb = max(0, cxp-h), min(img.shape[1], cxp+h); ya, yb = max(0, cyp-h), min(img.shape[0], cyp+h)
    tile = img[ya:yb, xa:xb]; tp = os.path.join(OUT, f"tile_{name}.png"); cv2.imwrite(tp, tile)
    pts = [[float(x[j]-xa), float(y[j]-ya)] for j in range(len(x)) if xa <= x[j] < xb and ya <= y[j] < yb]
    pts = pts[::max(1, len(pts)//15)]
    res = model(tp, points=pts, labels=[1]*len(pts), verbose=False)
    vis = cv2.cvtColor(tile, cv2.COLOR_GRAY2BGR); mu = sam_union(res)
    if mu is not None:
        vis[mu > 0] = (0.4 * vis[mu > 0] + np.array([0, 0, 180])).astype(np.uint8)
    for p in pts:
        cv2.circle(vis, (int(p[0]), int(p[1])), 2, (0, 255, 0), -1)
    cv2.imwrite(os.path.join(OUT, f"engcrop_{name}.png"), vis)
    print(f"  engcrop_{name}.png: SAM-Maske deckt {0.0 if mu is None else 100.0*(mu>0).mean():4.1f}% des Fensters")

print("SAM enger Crop (beste Chance) ...")
overlay_engcrop(frames[40], "0040")
print("\nFertig. sam/data.json + auto_*.png + engcrop_*.png")
