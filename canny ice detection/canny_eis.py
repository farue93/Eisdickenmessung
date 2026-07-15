# -*- coding: utf-8 -*-
"""
canny_eis.py - Eisreihe auswerten mit CANNY-Kantendetektion (Methodenvergleich
zu frame_diff/serie_eis). Laserlinie wird NUR fuers Croppen verwendet; die
Referenz-/Nulllinie ist die Canny-Kante von Frame 0.

Ablauf:
  1. ROI-Band (aus laser_pipeline/crop_roi) NUR zum Croppen laden.
  2. Frame 0: Canny -> dominante Kante als geordnete, geglaettete Referenzlinie
     -> Bogenlaenge s + (auswaerts orientierte) Normalen = NULLLINIE.
  3. Jeder Frame: Canny -> entlang der Referenz-Normalen die Kantenkreuzung
     (zeitlich getrackt, naechste zur Vorframe-Lage) -> Versatz d_N(s).
  4. Eisdicke(s,N) = d_N(s) - d_0(s). Frame 0 = 0.
  5. cropped PNGs + data.json + viewer.html (Film + Kurve + umschaltbare Linie).

Liest nur; schreibt nur nach OUT_DIR.
"""
import os, sys, glob, json, argparse
import numpy as np
import cv2

# Helfer aus dem Schwester-Skript (HTML-Template, bilinear, ...) + laser_pipeline;
# Pfade relativ zum Repo (Schwesterordner) -> push-tauglich
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "frame differencing"))
sys.path.insert(0, os.path.join(HERE, "..", "pre processing"))
import serie_eis as se
import laser_pipeline as lp

# ---- Pfade (CLI-überschreibbar; Defaults relativ zum Repo) ----
GERUEST_DIR = os.path.normpath(os.path.join(HERE, "..", "pre processing", "output"))
def _erste_serie(here):
    _inp = os.path.normpath(os.path.join(here, "..", "input"))          # Serien-Ordner (../input)
    for d in sorted(glob.glob(os.path.join(_inp, "*"))):
        if os.path.isdir(d) and glob.glob(os.path.join(d, "*.tif")):
            return d
    return _inp
SERIE_DIR   = _erste_serie(HERE)   # erste Bilderserie in ../input (dorthin die Serien ablegen)
STEM        = "2026-04-02_17-51-15-328_image0000000"      # Frame 0 (Default: erstes *.tif der Serie)
OUT_DIR     = "serie_260402-174444"
FRAMES_DIR  = os.path.join(OUT_DIR, "frames")

# ---- Canny ----
BLUR        = 3      # Gauss-Glaettung vor Canny (gegen Speckle); ungerade
CANNY_LO    = 40     # unterer Hysterese-Schwellwert (schwache Kanten nur bei Anschluss)
CANNY_HI    = 120    # oberer Hysterese-Schwellwert (sichere Kanten)
LUECKEN     = 6      # px Dilation zum Schliessen kleiner Kantenluecken (Ordnung)

# ---- Detektion entlang Normalen ----
SUCH_INNEN   = 25    # px Suchweite nach innen entlang der Normale
SUCH_AUSSEN  = 60    # px Suchweite nach aussen (eng gegen ferne Reflexkanten)
EDGE_SCHWELLE = 60   # Schwelle auf der (interpolierten) Canny-Karte
GLATT_S      = 9     # Median-Fenster zur Glaettung der Dicke ueber s
EXPORT_STEP  = 2     # nur jeder n-te Punkt in die data.json
PX_PER_MM    = 13.9   # CAD-kalibriert (war 23.4 provisorisch)
MAX_FRAMES   = None  # None = alle Frames; Zahl zum Testen

bilinear     = se.bilinear      # Subpixel-Abtastung (aus serie_eis wiederverwendet)
median_glatt = se.median_glatt  # Median-Glaettung ueber s
nan_liste    = se.nan_liste     # Array JSON-tauglich machen (NaN -> None)
lade_grau    = se.lade_grau     # robustes Graustufen-Laden


def canny_karte(frame, maske):
    """Canny-Kanten, auf das ROI-Band beschraenkt (uint8 0/255)."""
    g = frame.astype(np.uint8)                     # Canny erwartet 8-Bit
    if BLUR >= 3:
        g = cv2.GaussianBlur(g, (BLUR, BLUR), 0)   # vorglaetten (gegen Sensor-Speckle)
    e = cv2.Canny(g, CANNY_LO, CANNY_HI)           # 1px-Kanten (Gradient+NMS+Hysterese), 0/255
    return e & (maske.astype(np.uint8) * 255)      # nur Kanten im ROI-Band behalten


def referenzlinie(frame0, maske):
    """Frame-0-Canny -> geordnete, geglaettete Referenzlinie + s + Aussen-Normalen."""
    e = canny_karte(frame0, maske)                 # Canny-Kanten von Frame 0 (im Band)
    ys, xs = np.where(e > 0)                        # Koordinaten aller Kantenpixel
    segs = lp.skelett_teilstuecke(xs, ys, LUECKEN)  # zu geordneten 1px-Teilstuecken (skeleton + BFS)
    if not segs:
        raise RuntimeError("Canny-Referenz: keine ordenbare Kante gefunden.")
    denses = [lp.fit_teilstueck(seg.astype(float), np.ones(len(seg))) for seg in segs]  # je Stueck ein Spline
    PTS, _ = lp.verbinde(denses)                    # Teilstuecke per Hermite-Bruecke verbinden
    voll, _ = lp.endverlaengerung(PTS, 0, lp.TANGENTEN_FIT)   # keine Verlaengerung (laenge=0)
    s, nrm = lp.bogenlaenge_und_normalen(voll)      # Bogenlaenge s + Einheits-Normalen je Punkt
    cx, cy = voll[:, 0].mean(), voll[:, 1].mean()   # Schwerpunkt der Kurve
    proj = np.mean(nrm[:, 0] * (cx - voll[:, 0]) + nrm[:, 1] * (cy - voll[:, 1]))  # zeigt Normale zum Schwerpunkt?
    sgn = -1.0 if proj > 0 else 1.0                          # auswaerts = weg vom Schwerpunkt
    return voll[:, 0], voll[:, 1], s, sgn * nrm[:, 0], sgn * nrm[:, 1]  # x, y, s + AUSSEN-Normalen


def detektiere(edges, x, y, outx, outy, prev):
    """Canny-Kantenkreuzung entlang jeder Aussennormale, getrackt (naechste zu prev)."""
    t = np.arange(-SUCH_INNEN, SUCH_AUSSEN + 0.5, 0.5)  # Offsets entlang der Normale (0.5px-Schritt)
    X = x[:, None] + outx[:, None] * t[None, :]     # (N,T) Abtast-x
    Y = y[:, None] + outy[:, None] * t[None, :]     # (N,T) Abtast-y
    P = bilinear(edges.astype(np.float32), X, Y)             # ~0..255 (Kante) je Station & Offset
    d = np.full(len(x), np.nan)                     # Ergebnis (N,), zunaechst NaN
    for i in range(P.shape[0]):                     # jede Station einzeln
        m = P[i] > EDGE_SCHWELLE                     # wo kreuzt eine Kante die Normale?
        if not m.any():
            continue                                # keine Kante -> d[i] bleibt NaN
        idx = np.where(m)[0]                          # Indizes der Kanten-Treffer
        gruppen = np.split(idx, np.where(np.diff(idx) > 1)[0] + 1)  # zusammenhaengende Kanten gruppieren
        kreuz = np.array([t[g].mean() for g in gruppen])           # Mitte (Offset) je Kantengruppe
        d[i] = float(kreuz[np.argmin(np.abs(kreuz - prev[i]))])     # naechste zur Vorframe-Lage (Tracking)
    return d


def schreibe_html(daten):
    """Messdaten ins gemeinsame HTML-Template einbetten -> viewer.html (Titel: Canny)."""
    js = "const DATA = " + json.dumps(daten) + ";"     # Daten als JS-Variable
    html = se.HTML_TEMPLATE.replace("/*DATA*/", js).replace("260402-174444",
            "260402-174444 (Canny)")                    # gemeinsames Template, Titel um "(Canny)" ergaenzt
    with open(os.path.join(OUT_DIR, "viewer.html"), "w", encoding="utf-8") as f:
        f.write(html)                                  # eigenstaendige viewer.html schreiben


def main():
    global SERIE_DIR, STEM, GERUEST_DIR, OUT_DIR, FRAMES_DIR
    ap = argparse.ArgumentParser(description="Canny-Kantendetektion: Eisdicke ueber eine Bilderserie.")
    ap.add_argument("serie", nargs="?", default=SERIE_DIR, help="Ordner mit der Bilderserie (*.tif)")
    ap.add_argument("--geruest", default=GERUEST_DIR, help="Vorverarbeitungs-Output (roi npz, nur zum Croppen)")
    ap.add_argument("--stem", default=None, help="Frame-0-Stamm (Default: erstes *.tif der Serie)")
    ap.add_argument("--out", default=None, help="Ausgabeordner (Default: serie_<Ordner-Suffix>)")
    a = ap.parse_args()
    SERIE_DIR, GERUEST_DIR = a.serie, a.geruest
    _fr = sorted(glob.glob(os.path.join(SERIE_DIR, "*.tif")))
    STEM = a.stem or (os.path.splitext(os.path.basename(_fr[0]))[0] if _fr else STEM)
    OUT_DIR = a.out or ("serie_" + os.path.basename(os.path.normpath(SERIE_DIR)).split("_")[-1])
    FRAMES_DIR = os.path.join(OUT_DIR, "frames")
    os.makedirs(FRAMES_DIR, exist_ok=True)             # Ausgabeordner frames/ anlegen
    R = np.load(os.path.join(GERUEST_DIR, STEM + "_roi.npz"))  # ROI-Band (nur zum Croppen)
    maske = R["maske"]; bbox = tuple(int(v) for v in R["bbox"]); y0, y1, x0, x1 = bbox  # Band + Crop-Grenzen

    frames = sorted(glob.glob(os.path.join(SERIE_DIR, "*.tif")))  # alle Frame-Dateien
    if MAX_FRAMES:
        frames = frames[:MAX_FRAMES]                   # ggf. begrenzen (Test)

    # Referenzlinie = Canny(Frame 0)
    f0 = lade_grau(frames[0])                          # Frame 0 laden
    rx, ry, s, outx, outy = referenzlinie(f0, maske)   # Nulllinie aus Canny(Frame0) + s + Aussennormalen
    print(f"{len(frames)} Frames, Crop {y1-y0}x{x1-x0}, Referenzlinie {len(rx)} Punkte")

    prev = np.zeros(len(rx))                           # Vorframe-Lage je Station (Start 0) fuers Tracking
    d0 = None                                          # Nulllinie d_0 (wird in Frame 0 gesetzt)
    sub = slice(None, None, EXPORT_STEP)               # Ausduennung fuer die data.json
    s_exp = nan_liste(s[sub])                          # ausgeduennte s-Achse
    frame_daten = []                                   # sammelt je Frame ein Dict

    for k, fp in enumerate(frames):                    # jeder Frame der Serie
        img = lade_grau(fp)                            # Vollbild laden
        if img is None:
            print(f"  Frame {k}: defekt, uebersprungen"); continue  # defektes Bild ueberspringen

        masked = np.zeros_like(img); masked[maske] = img[maske]  # nur das ROI-Band behalten (Rest schwarz)
        cv2.imwrite(os.path.join(FRAMES_DIR, f"{k:04d}.png"),
                    masked[y0:y1, x0:x1].astype(np.uint8))  # Band-Crop als PNG (fuer den Film)

        edges = canny_karte(img, maske)                # Canny-Kanten dieses Frames
        d = detektiere(edges, rx, ry, outx, outy, prev)  # Kantenkreuzung je Normale (getrackt) = d_N(s)
        prev = np.where(np.isfinite(d), d, prev)             # Tracking aktualisieren (letzte gute Lage)
        if d0 is None:
            d0 = d.copy()                              # erster Frame -> Nulllinie
        dicke = median_glatt(d - d0, GLATT_S)          # Eisdicke = d_N - d_0, geglaettet

        gut = np.isfinite(d)                           # wo wurde eine Kante gefunden?
        ix = np.where(gut, rx + outx * np.nan_to_num(d) - x0, np.nan)  # Eiskante-x in Crop-Koordinaten
        iy = np.where(gut, ry + outy * np.nan_to_num(d) - y0, np.nan)  # Eiskante-y in Crop-Koordinaten
        frame_daten.append({
            "file": f"frames/{k:04d}.png",
            "name": os.path.basename(fp).replace(".tif", ""),
            "ix": nan_liste(ix[sub]), "iy": nan_liste(iy[sub]),
            "dicke": nan_liste(dicke[sub]),
        })
        eis = np.isfinite(dicke) & (dicke > 1.0)       # echte Eisstellen (>1px)
        mx = np.nanmax(dicke) if np.isfinite(dicke).any() else float("nan")  # max Dicke
        print(f"  Frame {k:2d}: Eis auf {int(eis.sum())/len(s)*100:4.0f}%  "
              f"max {mx:5.1f}px  median {np.nanmedian(dicke[eis]) if eis.any() else 0:4.1f}px")

    daten = {"bbox": [x0, y0, x1, y1], "crop_w": x1-x0, "crop_h": y1-y0,   # Crop-Lage/-Groesse
             "px_per_mm": PX_PER_MM, "s": s_exp, "frames": frame_daten}     # Kalibrierung, s-Achse, Frames
    with open(os.path.join(OUT_DIR, "data.json"), "w") as f:
        json.dump(daten, f)                            # alle Messdaten speichern
    schreibe_html(daten)                               # Viewer bauen
    print(f"-> {OUT_DIR}/viewer.html ({len(frame_daten)} Frames)")


if __name__ == "__main__":
    main()
