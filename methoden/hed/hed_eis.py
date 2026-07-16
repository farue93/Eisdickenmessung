# -*- coding: utf-8 -*-
"""
hed_eis.py - Eisreihe mit HED (Holistically-Nested Edge Detection, KI-Kanten)
statt Canny. Laser nur fuers Croppen; Referenz/Nulllinie = HED-Kante Frame 0.
Arbeitet auf den bereits gecroppten PNGs (Crop-Koordinaten) -> kein TIF-Nachladen.
"""
import os, sys, glob, json, shutil
import numpy as np
import cv2

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "frame_differencing"))
sys.path.insert(0, os.path.join(_HERE, "..", "..", "preprocessing"))
import serie_eis as se
import laser_pipeline as lp
from skimage.morphology import skeletonize

_FD  = os.path.normpath(os.path.join(_HERE, "..", "frame_differencing"))
_GER = os.path.normpath(os.path.join(_HERE, "..", "..", "preprocessing", "output"))


def _erste_serie():
    import glob as _g
    for d in sorted(_g.glob(os.path.join(_FD, "serie_*"))):
        if os.path.isdir(os.path.join(d, "frames")):
            return d
    return None


_SER = _erste_serie()
FRAMES_IN = os.path.join(_SER, "frames") if _SER else ""
_STEM = (json.load(open(os.path.join(_SER, "data.json")))["frames"][0]["name"]
         if _SER and os.path.exists(os.path.join(_SER, "data.json")) else "")
ROI = os.path.join(_GER, _STEM + "_roi.npz") if _STEM else ""
OUT_DIR = ("serie_" + os.path.basename(_SER).split("serie_", 1)[-1]) if _SER else "serie_hed"
FRAMES_OUT = os.path.join(OUT_DIR, "frames")
MEAN = (104.00698793, 116.66876762, 122.67891434)

LUECKEN = 6
HED_BIN = 40          # Schwelle auf HED-Wahrscheinlichkeit*255 -> binaere Kante
SUCH_INNEN, SUCH_AUSSEN = 25, 60
EDGE_SCHWELLE = 60
MAX_JUMP = 5          # px: max. Aenderung der Kante pro Frame (Eis waechst langsam)
GLATT_S = 9
EXPORT_STEP = 2
PX_PER_MM = 13.9   # CAD-kalibriert (war 23.4 provisorisch)
MAX_FRAMES = None

bilinear = se.bilinear; median_glatt = se.median_glatt; nan_liste = se.nan_liste


class CropLayer:
    def __init__(self, p, b): self.s = [0, 0, 0, 0]
    def getMemoryShapes(self, inputs):
        inp, tgt = inputs[0], inputs[1]; b, c, H, W = inp; h, w = tgt[2], tgt[3]
        ys = (H-h)//2; xs = (W-w)//2; self.s = [ys, ys+h, xs, xs+w]
        return [[b, c, h, w]]
    def forward(self, inputs):
        ys, ye, xs, xe = self.s; return [inputs[0][:, :, ys:ye, xs:xe]]


def init_hed():
    try:
        cv2.dnn_registerLayer("Crop", CropLayer)
    except cv2.error:
        pass
    return cv2.dnn.readNetFromCaffe(
        os.path.join(os.path.dirname(__file__) or ".", "deploy.prototxt"),
        os.path.join(os.path.dirname(__file__) or ".", "hed.caffemodel"))


def hed_karte(net, crop_bgr, mk):
    H, W = crop_bgr.shape[:2]; w, h = (W//16*16, H//16*16)
    blob = cv2.dnn.blobFromImage(crop_bgr, 1.0, (w, h), MEAN, swapRB=False, crop=False)
    net.setInput(blob)
    edge = cv2.resize(net.forward()[0, 0], (W, H)) * 255.0
    bin_ = (edge > HED_BIN) & (mk > 0)
    return (skeletonize(bin_).astype(np.uint8) * 255)        # duenne 1px-Kante


def referenzlinie(edges):
    ys, xs = np.where(edges > 0)
    segs = lp.skelett_teilstuecke(xs, ys, LUECKEN)
    if not segs:
        raise RuntimeError("HED-Referenz: keine ordenbare Kante.")
    denses = [lp.fit_teilstueck(s.astype(float), np.ones(len(s))) for s in segs]
    PTS, _ = lp.verbinde(denses); voll, _ = lp.endverlaengerung(PTS, 0, lp.TANGENTEN_FIT)
    s, nrm = lp.bogenlaenge_und_normalen(voll)
    cx, cy = voll[:, 0].mean(), voll[:, 1].mean()
    proj = np.mean(nrm[:, 0]*(cx-voll[:, 0]) + nrm[:, 1]*(cy-voll[:, 1]))
    sgn = -1.0 if proj > 0 else 1.0
    return voll[:, 0], voll[:, 1], s, sgn*nrm[:, 0], sgn*nrm[:, 1]


def detektiere(edges, x, y, outx, outy, prev):
    t = np.arange(-SUCH_INNEN, SUCH_AUSSEN+0.5, 0.5)
    X = x[:, None]+outx[:, None]*t[None, :]; Y = y[:, None]+outy[:, None]*t[None, :]
    P = bilinear(edges.astype(np.float32), X, Y)
    d = np.full(len(x), np.nan)
    for i in range(P.shape[0]):
        m = P[i] > EDGE_SCHWELLE
        if not m.any():
            continue
        idx = np.where(m)[0]; gr = np.split(idx, np.where(np.diff(idx) > 1)[0]+1)
        kreuz = np.array([t[g].mean() for g in gr])
        nah = kreuz[np.argmin(np.abs(kreuz - prev[i]))]
        if abs(nah - prev[i]) <= MAX_JUMP:                   # kein Sprung auf ferne Kante
            d[i] = float(nah)
    return d


def schreibe_html(daten):
    js = "const DATA = " + json.dumps(daten) + ";"
    html = se.HTML_TEMPLATE.replace("/*DATA*/", js).replace("260402-174444", "260402-174444 (HED)")
    open(os.path.join(OUT_DIR, "viewer.html"), "w", encoding="utf-8").write(html)


def main():
    global FRAMES_IN, ROI, OUT_DIR, FRAMES_OUT
    import argparse
    ap = argparse.ArgumentParser(description="HED-Kantendetektion: Eisdicke ueber eine (gecroppte) Bilderserie.")
    ap.add_argument("--frames", default=FRAMES_IN, help="Ordner mit gecroppten Frames (aus serie_eis)")
    ap.add_argument("--geruest", default=None, help="Vorverarbeitungs-Output (fuer <stem>_roi.npz)")
    ap.add_argument("--stem", default=None, help="Frame-0-Stamm (fuer die roi.npz)")
    ap.add_argument("--out", default=None, help="Ausgabeordner")
    a = ap.parse_args()
    FRAMES_IN = a.frames
    if a.geruest and a.stem:
        ROI = os.path.join(a.geruest, a.stem + "_roi.npz")
    if a.out:
        OUT_DIR = a.out
    FRAMES_OUT = os.path.join(OUT_DIR, "frames")
    os.makedirs(FRAMES_OUT, exist_ok=True)
    R = np.load(ROI); y0, y1, x0, x1 = (int(v) for v in R["bbox"])
    mk = cv2.erode(R["maske"][y0:y1, x0:x1].astype(np.uint8), np.ones((9, 9), np.uint8))
    net = init_hed()

    frames = sorted(glob.glob(os.path.join(FRAMES_IN, "*.png")))
    if MAX_FRAMES:
        frames = frames[:MAX_FRAMES]

    f0 = cv2.imread(frames[0], cv2.IMREAD_COLOR)
    rx, ry, s, ox, oy = referenzlinie(hed_karte(net, f0, mk))
    print(f"{len(frames)} Frames, Referenzlinie {len(rx)} Punkte")

    prev = np.zeros(len(rx)); d0 = None
    sub = slice(None, None, EXPORT_STEP); s_exp = nan_liste(s[sub]); fd = []
    for k, fp in enumerate(frames):
        crop = cv2.imread(fp, cv2.IMREAD_COLOR)
        shutil.copy(fp, os.path.join(FRAMES_OUT, f"{k:04d}.png"))
        edges = hed_karte(net, crop, mk)
        d = detektiere(edges, rx, ry, ox, oy, prev)
        prev = np.where(np.isfinite(d), d, prev)
        if d0 is None:
            d0 = d.copy()
        dicke = median_glatt(d - d0, GLATT_S)
        gut = np.isfinite(d)
        ix = np.where(gut, rx + ox*np.nan_to_num(d), np.nan)
        iy = np.where(gut, ry + oy*np.nan_to_num(d), np.nan)
        fd.append({"file": f"frames/{k:04d}.png",
                   "name": os.path.basename(fp).replace(".png", ""),
                   "ix": nan_liste(ix[sub]), "iy": nan_liste(iy[sub]), "dicke": nan_liste(dicke[sub])})
        eis = np.isfinite(dicke) & (dicke > 1.0)
        mx = np.nanmax(dicke) if np.isfinite(dicke).any() else float("nan")
        print(f"  Frame {k:2d}: Eis {int(eis.sum())/len(s)*100:4.0f}%  max {mx:5.1f}px  "
              f"median {np.nanmedian(dicke[eis]) if eis.any() else 0:4.1f}px")

    daten = {"bbox": [x0, y0, x1, y1], "crop_w": x1-x0, "crop_h": y1-y0,
             "px_per_mm": PX_PER_MM, "s": s_exp, "frames": fd}
    json.dump(daten, open(os.path.join(OUT_DIR, "data.json"), "w"))
    schreibe_html(daten)
    print(f"-> {OUT_DIR}/viewer.html ({len(fd)} Frames)")


if __name__ == "__main__":
    main()
