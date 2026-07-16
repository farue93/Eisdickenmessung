# -*- coding: utf-8 -*-
"""
daten_lader.py - Trainingsdaten fuer das U-Net (laser_v2/unet/).
Frames/Masken/split.json liegen im ELTERNORDNER laser_v2/ (gemeinsam mit dem
Label-Tool). Wenige Labels -> viele Beispiele durch Patches + Augmentierung.
"""
import os, json
import numpy as np
import cv2
import torch

BASE = os.path.dirname(os.path.abspath(__file__))    # laser_v2/unet
LAB  = os.path.join(os.path.dirname(os.path.dirname(BASE)), "gemeinsam")                          # laser_v2  (frames/ masks/ split.json)
# Trainingsziel: die Mittellinien (masks_train/, von mittellinie.py) - sonst Handlabels
MASKDIR = "masks_train" if os.path.isdir(os.path.join(LAB, "masks_train")) else "masks"


def lade_paare(gruppe):
    sp = json.load(open(os.path.join(LAB, "split.json")))
    paare, leer = [], []
    for name in sp.get(gruppe, []):
        img = cv2.imread(os.path.join(LAB, "frames", name), cv2.IMREAD_GRAYSCALE)
        m = cv2.imread(os.path.join(LAB, MASKDIR, name), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        if m is None or int((m > 0).sum()) == 0:
            leer.append(name); continue
        paare.append((img.astype(np.float32), (m > 0).astype(np.float32)))
    return paare, leer


def sample_patch(img, maske, size):
    H, W = img.shape
    ys, xs = np.where(maske > 0)
    if len(xs) and np.random.rand() < 0.8:
        k = np.random.randint(len(xs))
        cy = ys[k] + np.random.randint(-size // 4, size // 4 + 1)
        cx = xs[k] + np.random.randint(-size // 4, size // 4 + 1)
        y0 = int(np.clip(cy - size // 2, 0, H - size)); x0 = int(np.clip(cx - size // 2, 0, W - size))
    else:
        y0 = np.random.randint(0, H - size + 1); x0 = np.random.randint(0, W - size + 1)
    return img[y0:y0+size, x0:x0+size], maske[y0:y0+size, x0:x0+size]


def augment(im, ma):
    if np.random.rand() < 0.5: im, ma = im[:, ::-1], ma[:, ::-1]
    if np.random.rand() < 0.5: im, ma = im[::-1, :], ma[::-1, :]
    im = np.clip(im * (0.8 + 0.4 * np.random.rand()) + (np.random.rand() - 0.5) * 20, 0, 255)
    return np.ascontiguousarray(im), np.ascontiguousarray(ma)


def batch(paare, n, size, dev):
    ims, mas = [], []
    for _ in range(n):
        img, maske = paare[np.random.randint(len(paare))]
        pi, pm = augment(*sample_patch(img, maske, size)); ims.append(pi); mas.append(pm)
    X = torch.from_numpy(np.stack(ims)[:, None] / 255.0).float().to(dev)
    Y = torch.from_numpy(np.stack(mas)[:, None]).float().to(dev)
    return X, Y
