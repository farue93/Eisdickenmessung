# -*- coding: utf-8 -*-
"""
apply_unet.py - trainiertes U-Net (unet.pt) ueber die ganze Serie 174444 anwenden.
Liegt in laser_v2/unet/. Maske -> groesste Komponente -> Schwerpunkt entlang der
Laser-Normalen -> Eisdicke. Schreibt data.json HIER (unet/) fuer den Vergleich.
"""
import os, sys, glob, json
import numpy as np
import cv2
import torch

BASE = os.path.dirname(os.path.abspath(__file__))                  # laser_v2/unet
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(BASE)), "gemeinsam"))
import pfade                                                        # relative Repo-Aufloesung
import serie_eis as se
from modell import UNet

FRAMES = pfade.frames_dir()
OUT    = BASE                                                       # data.json hierher
PX_PER_MM = 13.9
dev = "cuda" if torch.cuda.is_available() else "cpu"

G = pfade.geometrie()
x, y, s = G["x"], G["y"], G["s"]
outx, outy = G["outx"], G["outy"]
x0, y0, x1, y1 = G["bbox"]

net = UNet().to(dev); net.load_state_dict(torch.load(os.path.join(BASE, "unet.pt"))); net.eval()


def echte_linien(m, min_area=100):
    """ALLE hinreichend grossen Komponenten behalten (beide Aeste), Specks weg."""
    n, lab, st, _ = cv2.connectedComponentsWithStats((m > 0).astype(np.uint8))
    keep = np.zeros_like(m, np.uint8)
    for k in range(1, n):
        if st[k, cv2.CC_STAT_AREA] >= min_area:
            keep[lab == k] = 255
    return keep


def unet_maske(img):
    H, W = img.shape; Hp, Wp = H + (-H) % 8, W + (-W) % 8
    Xt = torch.from_numpy(np.pad(img, ((0, Hp-H), (0, Wp-W)))[None, None] / 255.).float().to(dev)
    with torch.no_grad():
        pr = (torch.sigmoid(net(Xt)) > 0.5)[0, 0].cpu().numpy()[:H, :W].astype(np.uint8) * 255
    return echte_linien(pr)


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


frames = sorted(glob.glob(os.path.join(FRAMES, "*.png")))
sub = slice(None, None, 2); s_exp = se.nan_liste(s[sub])
d0 = None; fr = []
print(f"U-Net ueber {len(frames)} Frames, Geraet {dev}")
for k, fp in enumerate(frames):
    img = cv2.imread(fp, cv2.IMREAD_GRAYSCALE).astype(np.float32)
    d = detektiere(unet_maske(img))
    if d0 is None:
        d0 = d.copy()
    dicke = se.median_glatt(d - d0, 9)
    gut = np.isfinite(d)
    ix = np.where(gut, x + outx * np.nan_to_num(d), np.nan)
    iy = np.where(gut, y + outy * np.nan_to_num(d), np.nan)
    fr.append({"file": f"frames/{k:04d}.png", "name": os.path.basename(fp).replace(".png", ""),
               "ix": se.nan_liste(ix[sub]), "iy": se.nan_liste(iy[sub]), "dicke": se.nan_liste(dicke[sub])})
json.dump({"crop_w": x1-x0, "crop_h": y1-y0, "px_per_mm": PX_PER_MM, "s": s_exp, "frames": fr},
          open(os.path.join(OUT, "data.json"), "w"))
print(f"-> {OUT}/data.json ({len(fr)} Frames)")
