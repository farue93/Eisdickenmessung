# -*- coding: utf-8 -*-
"""
train.py - U-Net auf die 5 gelabelten laser_v2-Frames trainieren (from scratch).
Bewusst 'auswendig lernen' (wenig Daten). Speichert unet.pt.
Loss = BCE(pos_weight) + Dice. Bewertung auf den Trainingsframes (Memorierung).
"""
import os
import numpy as np
import cv2
import torch
import torch.nn as nn

from modell import UNet
import daten_lader as dl

BASE = os.path.dirname(os.path.abspath(__file__))
PATCH, BATCH, ITERS, LR, POSW = 256, 8, 500, 1e-3, 10.0
torch.manual_seed(0); np.random.seed(0)
dev = "cuda" if torch.cuda.is_available() else "cpu"

tr, leer = dl.lade_paare("train")
if leer:
    print("[!] Noch ungelabelt (leere Maske):", leer)
if not tr:
    print("[!] Keine gelabelten Frames. Erst mit label.bat labeln."); raise SystemExit

net = UNet().to(dev)
opt = torch.optim.Adam(net.parameters(), lr=LR)
bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(POSW, device=dev))


def dice_loss(logit, y):
    p = torch.sigmoid(logit); inter = (p * y).sum()
    return 1 - (2 * inter + 1) / (p.sum() + y.sum() + 1)


def _pad8(a):
    H, W = a.shape; return np.pad(a, ((0, (-H) % 8), (0, (-W) % 8)))


@torch.no_grad()
def eval_train():
    net.eval(); ious, dices = [], []
    for img, mask in tr:
        X = torch.from_numpy(_pad8(img)[None, None] / 255.).float().to(dev)
        pred = (torch.sigmoid(net(X)) > 0.5).float()[0, 0, :img.shape[0], :img.shape[1]].cpu().numpy()
        inter = (pred * mask).sum(); uni = pred.sum() + mask.sum() - inter
        ious.append(inter / (uni + 1e-6)); dices.append(2 * inter / (pred.sum() + mask.sum() + 1e-6))
    return float(np.mean(ious)), float(np.mean(dices))


print(f"Geraet {dev} | train {len(tr)} Frames | Netz {sum(p.numel() for p in net.parameters()):,} Param")
best = 0.0
for it in range(1, ITERS + 1):
    net.train()
    X, Y = dl.batch(tr, BATCH, PATCH, dev)
    opt.zero_grad(); lo = net(X)
    loss = bce(lo, Y) + dice_loss(lo, Y)
    loss.backward(); opt.step()
    if it == 1 or it % 50 == 0:
        ti, td = eval_train(); flag = ""
        if td > best:
            best = td; torch.save(net.state_dict(), os.path.join(BASE, "unet.pt")); flag = "  *bestes*"
        print(f"  it {it:3d} | loss {loss.item():.3f} | train IoU {ti:.3f} Dice {td:.3f}{flag}")
print(f"Bestes train-Dice: {best:.3f}  -> unet.pt")
